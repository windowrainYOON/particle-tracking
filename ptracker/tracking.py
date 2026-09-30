"""4단계: 입자 트래킹 (v2 알고리즘).

예측 위치 = HT 세포 흐름(optical flow) + Cy5 패턴 매칭(입자와 주변 입자 배치를 템플릿으로 다음 프레임에서 탐색)
비용 = 예측 거리 + 세포 불일치 + intensity 차이 + 크기 차이 + 이웃 일관성(주변 입자 이동 중앙값과의 차이, 2-pass)
→ LAP(헝가리안) 1:1 연결 → 1프레임 gap closing
"""
from __future__ import annotations
import numpy as np, pandas as pd
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from skimage import morphology
from skimage.feature import match_template
from .config import TrackingParams, DetectionParams
from .detection import background_subtracted
from .registration import fl_to_ht


# ------------------------------------------------------------------ preparation
def prepare_detections(P, reg, S, CID, FP, CS, FLOW, HT) -> pd.DataFrame:
    df = P.copy()
    T, NH, NW = CID.shape
    df["y_ht"], df["x_ht"] = fl_to_ht(df.y.values, df.x.values, df.frame.values, reg, S)
    yi = np.clip(df.y_ht.round().astype(int), 0, NH - 1).values; xi = np.clip(df.x_ht.round().astype(int), 0, NW - 1).values
    fi = df.frame.values - 1
    df["cell_id"] = CID[fi, yi, xi]; df["inside_cell"] = df.cell_id > 0
    df["on_footprint"] = FP[fi, yi, xi]
    df["region"] = np.where(df.inside_cell, "cytoplasm", np.where(df.on_footprint, "spread_membrane", "background"))
    mit = set(map(tuple, CS[CS.mitotic][["cell_id", "frame"]].values))
    df["in_mitotic_cell"] = [(c, f) in mit for c, f in zip(df.cell_id, df.frame)]
    tv = np.zeros(len(df), np.float32); samples = []
    for z in range(T):   # HT에서도 밝은 점(입자)인지 확인
        th = ndi.maximum_filter(morphology.white_tophat(HT[z].astype(np.float32), morphology.disk(4)), 3)
        m = fi == z; tv[m] = th[yi[m], xi[m]]; e = min(50, NH // 8); samples.append(th[e:-e:7, e:-e:7].ravel())
    df["ht_tophat"] = tv; df["ht_confirmed"] = df.ht_tophat > np.percentile(np.concatenate(samples), 99)
    fz = np.clip(fi, 0, max(T - 2, 0))
    if len(FLOW):
        df["flow_y"] = np.where(df.inside_cell, FLOW[fz, 0, yi, xi], 0.0); df["flow_x"] = np.where(df.inside_cell, FLOW[fz, 1, yi, xi], 0.0)
    else:
        df["flow_y"] = 0.0; df["flow_x"] = 0.0
    df["predF_y"] = df.y_ht + df.flow_y; df["predF_x"] = df.x_ht + df.flow_x
    df["idx"] = np.arange(len(df))
    return df


def pattern_predictions(df, A, reg, S, det: DetectionParams, prm: TrackingParams, progress=None) -> pd.DataFrame:
    """Cy5 패턴 매칭으로 다음 프레임 위치 예측 (predP_y/x, pat_score)."""
    T = A.shape[0]; h, R = prm.pattern_half, prm.pattern_search
    py = np.full(len(df), np.nan); px = np.full(len(df), np.nan); ps = np.full(len(df), np.nan)
    o = h + R + 2; S1 = background_subtracted(A[0], det).clip(0)
    for z in range(T - 1):
        S0 = S1; S1 = background_subtracted(A[z + 1], det).clip(0); pad0 = np.pad(S0, o); pad1 = np.pad(S1, o)
        for i in np.nonzero((df.frame == z + 1).values)[0]:
            y, x = df.y.values[i], df.x.values[i]; yc, xc = int(round(y)), int(round(x))
            cyi, cxi = int(round(y + df.flow_y.values[i] / S)), int(round(x + df.flow_x.values[i] / S))
            tpl = pad0[yc + o - h:yc + o + h + 1, xc + o - h:xc + o + h + 1]
            srch = pad1[cyi + o - h - R:cyi + o + h + R + 1, cxi + o - h - R:cxi + o + h + R + 1]
            if tpl.shape != (2 * h + 1,) * 2 or srch.shape != (2 * (h + R) + 1,) * 2 or tpl.std() < 1e-6 or srch.std() < 1e-6:
                continue
            res = match_template(srch, tpl); k = np.unravel_index(res.argmax(), res.shape); sc = res[k]
            dy, dx = k[0] - R, k[1] - R
            if 0 < k[0] < res.shape[0] - 1 and 0 < k[1] < res.shape[1] - 1:     # parabolic subpixel
                a, b, c = res[k[0] - 1, k[1]], res[k], res[k[0] + 1, k[1]]; dn = a - 2 * b + c; dy += 0.5 * (a - c) / dn if dn != 0 else 0
                a, b, c = res[k[0], k[1] - 1], res[k], res[k[0], k[1] + 1]; dn = a - 2 * b + c; dx += 0.5 * (a - c) / dn if dn != 0 else 0
            py[i] = cyi + dy; px[i] = cxi + dx; ps[i] = sc
        if progress: progress((z + 1) / (T - 1), f"패턴 매칭 {z+1}/{T-1}")
    df = df.copy(); df["pat_score"] = ps
    nxt = np.clip(df.frame.values + 1, 1, T)          # 예측 위치는 다음 프레임의 HT 좌표계
    pat_y, pat_x = fl_to_ht(py, px, nxt, reg, S)
    w = np.nan_to_num(np.where(ps >= prm.pattern_score_hi, 1.0, np.where(ps >= prm.pattern_score_lo, 0.5, 0.0)))
    df["predP_y"] = np.where(w > 0, w * np.nan_to_num(pat_y) + (1 - w) * df.predF_y, df.predF_y)
    df["predP_x"] = np.where(w > 0, w * np.nan_to_num(pat_x) + (1 - w) * df.predF_x, df.predF_x)
    return df


# ------------------------------------------------------------------ linking
def cost_matrix(Aa, Bb, prm: TrackingParams, py_, px_, nb=None, gate=None):
    gate = prm.gate if gate is None else gate
    d = np.linalg.norm(np.stack([Aa[py_].values, Aa[px_].values], 1)[:, None] - Bb[["y_ht", "x_ht"]].values[None], axis=2)
    c = d.copy()
    c += prm.w_cell * (Aa.cell_id.values[:, None] != Bb.cell_id.values[None])
    c += prm.w_int * np.abs(np.log(Aa.mean_intensity.values[:, None] / Bb.mean_intensity.values[None]))
    c += prm.w_area * np.abs(np.log(Aa.area_px.values[:, None] / Bb.area_px.values[None]))
    if nb is not None:
        disp = Bb[["y_ht", "x_ht"]].values[None] - Aa[["y_ht", "x_ht"]].values[:, None]
        dev = np.linalg.norm(disp - nb[:, None, :], axis=2); ok = ~np.isnan(nb[:, 0])
        c[ok] += prm.w_nb * dev[ok]
    c[d > gate] = np.inf
    return c, d


def lap(c, gate):
    """종료/시작 비용(gate+2)을 포함한 확장 행렬로 1:1 최적 할당."""
    na, nb = c.shape; big = 1e6; nc = gate + 2
    M = np.full((na + nb, nb + na), big); M[:na, :nb] = np.where(np.isinf(c), big, c)
    M[:na, nb:] = np.where(np.eye(na) > 0, nc, big); M[na:, :nb] = np.where(np.eye(nb) > 0, nc, big)
    M[na:, nb:] = np.where(M[:na, :nb].T < big, 0, big)
    r, cl = linear_sum_assignment(M)
    return [(i, j) for i, j in zip(r, cl) if i < na and j < nb and M[i, j] < big]


def neighbour_median(Aa, disp_map, prm: TrackingParams):
    xy = Aa[["y", "x"]].values; out = np.full((len(Aa), 2), np.nan)
    if len(Aa) == 0: return out
    tr = cKDTree(xy)
    for i, nbs in enumerate(tr.query_ball_point(xy, prm.nb_radius)):
        nbs = [j for j in nbs if j != i and Aa.idx.values[j] in disp_map and Aa.cell_id.values[j] == Aa.cell_id.values[i]]
        if len(nbs) >= prm.nb_min: out[i] = np.median([disp_map[Aa.idx.values[j]] for j in nbs], 0)
    return out


PRED = ("predP_y", "predP_x")   # pattern_predictions가 만든 예측 위치 (HT 좌표)


def link_frames(df, prm: TrackingParams, progress=None):
    py_, px_ = PRED; T = int(df.frame.max()); links = []; amb = []
    for t in range(1, T):
        Aa = df[df.frame == t]; Bb = df[df.frame == t + 1]
        if len(Aa) == 0 or len(Bb) == 0: continue
        c, d = cost_matrix(Aa, Bb, prm, py_, px_); L1 = lap(c, prm.gate)          # 1차 연결
        dm = {Aa.idx.values[i]: Bb[["y_ht", "x_ht"]].values[j] - Aa[["y_ht", "x_ht"]].values[i] for i, j in L1}
        nb = neighbour_median(Aa, dm, prm); c, d = cost_matrix(Aa, Bb, prm, py_, px_, nb); L1 = lap(c, prm.gate)   # 이웃 일관성 반영
        for i, j in L1:
            links.append((Aa.idx.values[i], Bb.idx.values[j], c[i, j], d[i, j], *nb[i]))
        srt = np.sort(np.where(np.isfinite(c), c, np.inf), 1)
        amb.append(np.mean(np.isfinite(srt[:, 0]) & (srt[:, 1] < srt[:, 0] + 1.0)) if c.shape[1] > 1 else 0)
        if progress: progress(t / (T - 1), f"연결 {t}/{T-1}")
    return pd.DataFrame(links, columns=["a", "b", "cost", "pred_dist", "nb_dy", "nb_dx"]), np.array(amb)


def build_tracks(df, links, prm: TrackingParams):
    """프레임 간 연결 → 트랙 ID. gap closing(1프레임 누락) 포함. 반환 (tracks, n_gap_closings)."""
    py_, px_ = PRED
    nx = dict(zip(links.a, links.b)); hp = set(links.b); tid = np.full(len(df), -1); k = 0
    for s0 in df.idx.values:
        if s0 in hp: continue
        cur = s0
        while True:
            tid[cur] = k
            if cur in nx: cur = nx[cur]
            else: break
        k += 1
    d = df.copy(); d["track_id"] = tid; mp = {}
    if prm.gap_closing:
        T = int(d.frame.max())
        ends = d.loc[d.groupby("track_id").frame.idxmax()]; sts = d.loc[d.groupby("track_id").frame.idxmin()]
        for t in range(1, T - 1):
            E = ends[ends.frame == t]; Sx = sts[sts.frame == t + 2]
            if len(E) == 0 or len(Sx) == 0: continue
            E2 = E.assign(gy=2 * E[py_] - E.y_ht, gx=2 * E[px_] - E.x_ht)
            c, _ = cost_matrix(E2, Sx, prm, "gy", "gx", gate=prm.gate * 1.3); c = np.where(np.isinf(c), 1e6, c)
            for i, j in zip(*linear_sum_assignment(c)):
                if c[i, j] < 1e6: mp[Sx.track_id.values[j]] = E.track_id.values[i]

    def root(x):
        while x in mp: x = mp[x]
        return x
    d["track_id"] = pd.factorize(d.track_id.map(root), sort=True)[0] + 1
    return d.sort_values(["track_id", "frame"]).reset_index(drop=True), len(mp)


def add_motion_columns(tr, dt_min):
    g = tr.groupby("track_id")
    for c in ["y_ht", "x_ht", "predF_y", "predF_x", "frame"]: tr["prev_" + c] = g[c].shift()
    one = np.where(tr.frame - tr.prev_frame == 1, 1, np.nan)
    tr["step_ht"] = np.hypot(tr.y_ht - tr.prev_y_ht, tr.x_ht - tr.prev_x_ht)
    tr["cell_carried_ht"] = np.hypot(tr.prev_predF_y - tr.prev_y_ht, tr.prev_predF_x - tr.prev_x_ht) * one
    tr["residual_ht"] = np.hypot(tr.y_ht - tr.prev_predF_y, tr.x_ht - tr.prev_predF_x) * one
    tr["time_min"] = (tr.frame - 1) * dt_min
    return tr.drop(columns=[c for c in tr.columns if c.startswith("prev_")])

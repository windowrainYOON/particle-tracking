"""4단계: 입자 트래킹 (v2 알고리즘 + 속도 예측·2단계 패턴 매칭).

예측 위치 = HT 세포 흐름(optical flow) + Cy5 패턴 매칭(입자와 주변 입자 배치를 템플릿으로 다음 프레임에서 탐색)
    + (pattern_refine) 패턴 점수가 낮으면 작은 템플릿으로 넓게 재탐색
    + (velocity_model) 패턴이 불확실하면 트랙 자체 속도(α-β/Kalman)로 보정, 트랙별 적응형 gate
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


def _match(pad0, pad1, o, y, x, cy, cx, h, R, prior_sigma=None):
    """(y,x) 주변 (2h+1)² 템플릿을 (cy,cx) 중심 ±R에서 NCC로 탐색. 반환 (y', x', score) 또는 None.
    prior_sigma가 있으면 탐색 중심에서 멀수록 점수를 가우시안으로 낮춰 가장 가까운 비슷한 봉우리를 고릅니다(점수는 원래 NCC)."""
    yc, xc = int(round(y)), int(round(x)); cyi, cxi = int(round(cy)), int(round(cx))
    tpl = pad0[yc + o - h:yc + o + h + 1, xc + o - h:xc + o + h + 1]
    srch = pad1[cyi + o - h - R:cyi + o + h + R + 1, cxi + o - h - R:cxi + o + h + R + 1]
    if tpl.shape != (2 * h + 1,) * 2 or srch.shape != (2 * (h + R) + 1,) * 2 or tpl.std() < 1e-6 or srch.std() < 1e-6:
        return None
    res = match_template(srch, tpl); sel = res
    if prior_sigma:
        g = np.arange(-R, R + 1); sel = res * np.exp(-(g[:, None] ** 2 + g[None] ** 2) / (2 * prior_sigma ** 2))
    k = np.unravel_index(sel.argmax(), sel.shape); sc = res[k]; dy, dx = k[0] - R, k[1] - R
    if 0 < k[0] < res.shape[0] - 1 and 0 < k[1] < res.shape[1] - 1:     # parabolic subpixel
        a, b, c = res[k[0] - 1, k[1]], res[k], res[k[0] + 1, k[1]]; dn = a - 2 * b + c; dy += 0.5 * (a - c) / dn if dn != 0 else 0
        a, b, c = res[k[0], k[1] - 1], res[k], res[k[0], k[1] + 1]; dn = a - 2 * b + c; dx += 0.5 * (a - c) / dn if dn != 0 else 0
    return cyi + dy, cxi + dx, sc


def pattern_raw(df, A, S, det: DetectionParams, prm: TrackingParams, progress=None):
    """패턴 매칭 원시 결과 (형광 좌표). 1단계: 넓은 템플릿(입자+주변 배치).
    2단계(pattern_refine): 1단계 점수가 낮은 입자만 작은 템플릿(입자 자체)으로 더 넓게 다시 탐색 — 이웃이 제각각 움직이는 밀집 지역·빠른 이동 대비.
    반환 dict(py, px, ps, ry, rx, rs)."""
    T = A.shape[0]; h, R = prm.pattern_half, prm.pattern_search; h2, R2 = prm.refine_half, prm.refine_search
    n = len(df); out = {k: np.full(n, np.nan) for k in ["py", "px", "ps", "ry", "rx", "rs"]}
    o = max(h + R, h2 + R2) + 2; S1 = background_subtracted(A[0], det).clip(0)
    fy, fx = df.flow_y.values / S, df.flow_x.values / S
    for z in range(T - 1):
        S0 = S1; S1 = background_subtracted(A[z + 1], det).clip(0); pad0 = np.pad(S0, o); pad1 = np.pad(S1, o)
        for i in np.nonzero((df.frame == z + 1).values)[0]:
            y, x = df.y.values[i], df.x.values[i]
            m = _match(pad0, pad1, o, y, x, y + fy[i], x + fx[i], h, R)
            if m: out["py"][i], out["px"][i], out["ps"][i] = m
            if prm.pattern_refine and not (m and m[2] >= prm.pattern_score_hi):
                m2 = _match(pad0, pad1, o, y, x, y + fy[i], x + fx[i], h2, R2, prior_sigma=R2 / 2)
                if m2: out["ry"][i], out["rx"][i], out["rs"][i] = m2
        if progress: progress((z + 1) / (T - 1), f"패턴 매칭 {z+1}/{T-1}")
    return out


def apply_pattern(df, raw, reg, S, prm: TrackingParams) -> pd.DataFrame:
    """원시 패턴 결과 → 패턴 위치(pat_y/x, HT 좌표)와 신뢰 가중치(pat_w), 흐름과 섞은 예측(predP_y/x)."""
    T = int(df.frame.max()); ps = raw["ps"]; rs = raw["rs"]
    w = np.nan_to_num(np.where(ps >= prm.pattern_score_hi, 1.0, np.where(ps >= prm.pattern_score_lo, 0.5, 0.0)))
    use2 = (w < 1) & (np.nan_to_num(rs) >= prm.refine_min_score) if prm.pattern_refine else np.zeros(len(df), bool)
    py = np.where(use2, raw["ry"], raw["py"]); px = np.where(use2, raw["rx"], raw["px"]); w = np.where(use2, prm.refine_weight, w)
    nxt = np.clip(df.frame.values + 1, 1, T)          # 예측 위치는 다음 프레임의 HT 좌표계
    pat_y, pat_x = fl_to_ht(py, px, nxt, reg, S)
    df = df.copy(); df["pat_score"] = ps; df["pat2_score"] = rs; df["pat_refined"] = use2
    df["pat_y"], df["pat_x"], df["pat_w"] = np.nan_to_num(pat_y), np.nan_to_num(pat_x), w
    df["predP_y"] = np.where(w > 0, w * df.pat_y + (1 - w) * df.predF_y, df.predF_y)
    df["predP_x"] = np.where(w > 0, w * df.pat_x + (1 - w) * df.predF_x, df.predF_x)
    return df


def pattern_predictions(df, A, reg, S, det: DetectionParams, prm: TrackingParams, progress=None) -> pd.DataFrame:
    """Cy5 패턴 매칭으로 다음 프레임 위치 예측 (pat_*, predP_y/x)."""
    return apply_pattern(df, pattern_raw(df, A, S, det, prm, progress), reg, S, prm)


# ------------------------------------------------------------------ linking
def cost_matrix(Aa, Bb, prm: TrackingParams, py_, px_, nb=None, gate=None):
    """비용 행렬. gate는 스칼라 또는 행(트랙)별 배열. 행별 gate가 넓으면 거리 비용을 gate 비율만큼 줄여
    (불확실한 트랙일수록 같은 거리의 벌점이 작음) 종료 비용과의 균형을 유지합니다."""
    gate = prm.gate if gate is None else gate
    g = np.broadcast_to(np.asarray(gate, float), (len(Aa),))[:, None]
    d = np.linalg.norm(np.stack([Aa[py_].values, Aa[px_].values], 1)[:, None] - Bb[["y_ht", "x_ht"]].values[None], axis=2)
    c = d * (prm.gate / g) if np.ndim(gate) else d.copy()
    c += prm.w_cell * (Aa.cell_id.values[:, None] != Bb.cell_id.values[None])
    c += prm.w_int * np.abs(np.log(Aa.mean_intensity.values[:, None] / Bb.mean_intensity.values[None]))
    c += prm.w_area * np.abs(np.log(Aa.area_px.values[:, None] / Bb.area_px.values[None]))
    if nb is not None:
        disp = Bb[["y_ht", "x_ht"]].values[None] - Aa[["y_ht", "x_ht"]].values[:, None]
        dev = np.linalg.norm(disp - nb[:, None, :], axis=2); ok = ~np.isnan(nb[:, 0])
        c[ok] += prm.w_nb * dev[ok]
    c[d > g] = np.inf
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


def link_frames(df, prm: TrackingParams, progress=None):
    """프레임 간 연결. 반환 (links, 프레임별 모호한 연결 비율, 검출별 예측 표 pred_y/pred_x/gate_used).

    velocity_model: 트랙마다 '세포 흐름을 뺀 자체 이동 속도'를 α-β(정상상태 Kalman) 필터로 추정해
    패턴 신뢰도가 낮을 때 흐름 예측 대신 (흐름 + 자체 속도)를 쓰고, 그 트랙의 예측 오차(RMS)에 비례해 gate를 넓힙니다.
    끄면 기존 v2와 같습니다(예측 = predP, gate 고정)."""
    T = int(df.frame.max()); n = len(df); links = []; amb = []
    pos = df[["y_ht", "x_ht"]].values; flow = df[["flow_y", "flow_x"]].values
    u = np.zeros((n, 2)); has_u = np.zeros(n, bool); rms = np.full(n, prm.gate / max(prm.gate_k, 1e-6))
    pred = df[["predP_y", "predP_x"]].values.copy(); gate_used = np.full(n, float(prm.gate))
    for t in range(1, T):
        Aa = df[df.frame == t]; Bb = df[df.frame == t + 1]
        if len(Aa) == 0 or len(Bb) == 0: continue
        ia = Aa.idx.values
        if prm.velocity_model:
            pv = df[["predF_y", "predF_x"]].values[ia] + np.where(has_u[ia, None], u[ia], 0)
            w = df.pat_w.values[ia][:, None]; pat = df[["pat_y", "pat_x"]].values[ia]
            pred[ia] = np.where(w > 0, w * pat + (1 - w) * pv, pv)
            gate_used[ia] = np.where(has_u[ia], np.clip(prm.gate_k * rms[ia], prm.gate, prm.gate_max), prm.gate)
        Aa = Aa.assign(pred_y=pred[ia, 0], pred_x=pred[ia, 1]); g = gate_used[ia] if prm.velocity_model else None
        c, d = cost_matrix(Aa, Bb, prm, "pred_y", "pred_x", gate=g); L1 = lap(c, prm.gate)          # 1차 연결
        dm = {ia[i]: pos[Bb.idx.values[j]] - pos[ia[i]] for i, j in L1}
        nb = neighbour_median(Aa, dm, prm); c, d = cost_matrix(Aa, Bb, prm, "pred_y", "pred_x", nb, gate=g); L1 = lap(c, prm.gate)   # 이웃 일관성 반영
        for i, j in L1:
            a, b = ia[i], Bb.idx.values[j]; links.append((a, b, c[i, j], d[i, j], *nb[i]))
            if prm.velocity_model:          # 연결된 검출로 속도·예측 오차 상태 전달
                uo = pos[b] - pos[a] - flow[a]; al = prm.vel_alpha
                u[b] = al * uo + (1 - al) * u[a] if has_u[a] else al * uo; has_u[b] = True
                rms[b] = np.sqrt(al * d[i, j] ** 2 + (1 - al) * rms[a] ** 2)
        srt = np.sort(np.where(np.isfinite(c), c, np.inf), 1)
        amb.append(np.mean(np.isfinite(srt[:, 0]) & (srt[:, 1] < srt[:, 0] + 1.0)) if c.shape[1] > 1 else 0)
        if progress: progress(t / (T - 1), f"연결 {t}/{T-1}")
    P = pd.DataFrame({"idx": df.idx.values, "pred_y": pred[:, 0], "pred_x": pred[:, 1], "gate_used": gate_used})
    return pd.DataFrame(links, columns=["a", "b", "cost", "pred_dist", "nb_dy", "nb_dx"]), np.array(amb), P


def build_tracks(df, links, prm: TrackingParams, pred=None):
    """프레임 간 연결 → 트랙 ID. gap closing(1프레임 누락) 포함. 반환 (tracks, n_gap_closings).
    pred: link_frames가 돌려준 검출별 예측 표(없으면 predP와 고정 gate 사용)."""
    d = df.copy()
    if pred is not None: d = d.drop(columns=[c for c in ["pred_y", "pred_x", "gate_used"] if c in d]).merge(pred, on="idx")
    else: d["pred_y"], d["pred_x"], d["gate_used"] = d.predP_y, d.predP_x, float(prm.gate)
    nx = dict(zip(links.a, links.b)); hp = set(links.b); tid = np.full(len(d), -1); k = 0
    pos_of = {v: i for i, v in enumerate(d.idx.values)}
    for s0 in d.idx.values:
        if s0 in hp: continue
        cur = s0
        while True:
            tid[pos_of[cur]] = k
            if cur in nx: cur = nx[cur]
            else: break
        k += 1
    d["track_id"] = tid; mp = {}
    if prm.gap_closing:
        T = int(d.frame.max())
        ends = d.loc[d.groupby("track_id").frame.idxmax()]; sts = d.loc[d.groupby("track_id").frame.idxmin()]
        for t in range(1, T - 1):
            E = ends[ends.frame == t]; Sx = sts[sts.frame == t + 2]
            if len(E) == 0 or len(Sx) == 0: continue
            E2 = E.assign(gy=2 * E.pred_y - E.y_ht, gx=2 * E.pred_x - E.x_ht)
            gg = E.gate_used.values * 1.3 if prm.velocity_model else prm.gate * 1.3
            c, _ = cost_matrix(E2, Sx, prm, "gy", "gx", gate=gg); c = np.where(np.isinf(c), 1e6, c)
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

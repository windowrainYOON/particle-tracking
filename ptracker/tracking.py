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


# ------------------------------------------------------------------ appearance (외형 특징)
APPEARANCE = {   # 이름: (열, 변환) — 비교량: log이면 |ln(a/b)|, 아니면 |a−b|
    "area": ("area_px", "log"), "cy5": ("mean_intensity", "log"), "ht_max": ("ht_ri_max", "lin"),
    "ht_mean": ("ht_ri_mean", "lin"), "ecc": ("eccentricity", "lin")}


def add_ht_features(df, LAB, HT, reg, S):
    """트래킹 전에 검출마다 ROI의 HT RI 평균·최대 (×10⁴ 단위 그대로)를 붙입니다. 측정 단계와 같은 좌표 변환."""
    from scipy import ndimage as ndi
    T = LAB.shape[0]; NH, NW = HT.shape[1:]; rec = {}
    for z in range(T):
        lab = LAB[z]; ht = HT[z]; dy, dx = reg.dy_ht.values[z], reg.dx_ht.values[z]
        for k, sl in enumerate(ndi.find_objects(lab), 1):
            if sl is None: continue
            yy, xx = np.nonzero(lab[sl] == k); yy = yy + sl[0].start; xx = xx + sl[1].start
            hy = np.clip(np.round((yy + 0.5) * S - 0.5 + dy).astype(int), 0, NH - 1); hx = np.clip(np.round((xx + 0.5) * S - 0.5 + dx).astype(int), 0, NW - 1)
            v = ht[hy, hx].astype(float); rec[(z + 1, k)] = (v.mean(), v.max())
    H = pd.DataFrame([(f, l, a, b) for (f, l), (a, b) in rec.items()], columns=["frame", "label", "ht_ri_mean", "ht_ri_max"])
    return df.drop(columns=[c for c in ["ht_ri_mean", "ht_ri_max"] if c in df]).merge(H, on=["frame", "label"], how="left")


def _feat_diff(a, b, kind):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.abs(np.log(a / b)) if kind == "log" else np.abs(a - b)


def learn_appearance(df, links, prm: TrackingParams):
    """1차 연결 중 확실한 연결(패턴 상관 ≥ 0.9, 비용 2위와 차이 ≥ 2)에서 '같은 입자의 변화'와
    '가장 가까운 다른 후보와의 차이'의 지수분포 척도(s_same, s_other)를 추정 → 특징별 가중치
    w = (1/s_same − 1/s_other) (로그 가능도비). 거리도 같은 방식으로 추정해 w_dist로 나눠 px 단위 비용으로 맞춥니다."""
    pos = df[["y_ht", "x_ht"]].values; fr = df.frame.values; tree = {}
    L = links[(df.pat_score.values[links.a.values] >= 0.9) & (links.margin.values >= 2)] if len(links) else links
    if len(L) < 50: return {}, {}
    same = {k: [] for k in APPEARANCE}; other = {k: [] for k in APPEARANCE}; dsame, dother = [], []
    for f in np.unique(fr): tree[f] = (cKDTree(pos[fr == f]), np.nonzero(fr == f)[0])
    for a, b, pd_ in zip(L.a.values, L.b.values, L.pred_dist.values):
        tr, ids = tree[fr[b]]; d, ii = tr.query(pos[b], k=2); o = ids[ii[1]] if len(ii) > 1 and np.isfinite(d[1]) else None
        if o is None: continue
        dsame.append(pd_); dother.append(np.linalg.norm(pos[o] - pos[b]) + pd_)
        for k, (col, kind) in APPEARANCE.items():
            same[k].append(_feat_diff(df[col].values[a], df[col].values[b], kind)); other[k].append(_feat_diff(df[col].values[a], df[col].values[o], kind))
    s = lambda x: max(float(np.nanmean(x)), 1e-6)
    w_dist = 1 / s(dsame) - 1 / s(dother)
    if w_dist <= 0: return {}, {}
    W = {k: max(0.0, (1 / s(same[k]) - 1 / s(other[k])) / w_dist) for k in APPEARANCE}
    stats = {k: dict(s_same=s(same[k]), s_other=s(other[k]), weight_px=W[k]) for k in APPEARANCE}
    stats["distance"] = dict(s_same=s(dsame), s_other=s(dother)); stats["n_links"] = len(dsame)
    return W, stats


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
def constellation_matrix(Aa, Bb, prm: TrackingParams, finite):
    """주변 배치 비용 (na, nb). Aa의 각 입자에 대해 같은 프레임의 가까운 이웃 con_k개(반경 con_radius)의
    '예측 위치 기준 상대 벡터'를 구하고, 후보 j 위치에 그 벡터를 더한 곳에서 Bb의 가장 가까운 검출까지 거리(상한 con_cap)의 평균.
    이웃이 2개 미만이면 0 (정보 없음). finite: 계산할 (i, j) 쌍 (gate 안)."""
    out = np.zeros(finite.shape)
    if len(Aa) < 3 or len(Bb) == 0 or not finite.any(): return out
    P = Aa[["y_ht", "x_ht"]].values; Q = Aa[["pred_y", "pred_x"]].values; B = Bb[["y_ht", "x_ht"]].values
    ta = cKDTree(P); tb = cKDTree(B); d, nn = ta.query(P, k=min(prm.con_k + 1, len(P)), distance_upper_bound=prm.con_radius)
    nn = nn[:, 1:]; ok = np.isfinite(d[:, 1:])                                   # 첫 번째는 자기 자신
    rel = np.where(ok[..., None], Q[np.where(ok, nn, 0)] - Q[:, None, :], np.nan)  # 예측 위치 기준 상대 벡터 (na, k, 2)
    I, J = np.nonzero(finite); valid = ok[I].sum(1) >= 2
    if not valid.any(): return out
    I, J = I[valid], J[valid]; q = B[J][:, None, :] + rel[I]; m = ok[I]
    e = np.full(m.shape, np.nan); dist, _ = tb.query(q[m]); e[m] = np.minimum(dist, prm.con_cap)
    out[I, J] = np.nanmean(e, 1); return out


# ------------------------------------------------------------------ 정답으로 학습한 연결 비용 (learn.py)
LEARN_FEATURES = ["d", "d2", "area", "cy5", "ht_max", "ht_mean", "ecc", "cell", "nb", "con"]


def pair_features(Aa, Bb, d, nb=None, con=None):
    """학습 비용용 특징 행렬 (이름 → (na, nb)). d = 예측 위치와의 거리(HT px). 트래킹과 학습이 같은 함수를 씁니다."""
    def diff(col, kind, scale=1.0):
        if col not in Aa or col not in Bb: return np.zeros(d.shape)
        return np.nan_to_num(_feat_diff(Aa[col].values[:, None].astype(float), Bb[col].values[None].astype(float), kind), nan=0.0) / scale
    F = {"d": d, "d2": (d / 10) ** 2, "area": diff("area_px", "log"), "cy5": diff("mean_intensity", "log"),
         "ht_max": diff("ht_ri_max", "lin", 100), "ht_mean": diff("ht_ri_mean", "lin", 100), "ecc": diff("eccentricity", "lin"),
         "cell": (Aa.cell_id.values[:, None] != Bb.cell_id.values[None]).astype(float), "nb": np.zeros(d.shape), "con": np.zeros(d.shape)}
    if nb is not None:
        disp = Bb[["y_ht", "x_ht"]].values[None] - Aa[["y_ht", "x_ht"]].values[:, None]
        dev = np.linalg.norm(disp - np.nan_to_num(nb)[:, None, :], axis=2); F["nb"] = np.where(np.isnan(nb[:, :1]), 0.0, dev)
    if con is not None: F["con"] = con
    return F


def load_learned(prm: TrackingParams):
    """prm.learned_model 파일의 가중치(px 단위, d = 1). 없거나 비어 있으면 None."""
    import json, os
    p = (prm.learned_model or "").strip()
    if not p or not os.path.exists(p): return None
    m = json.load(open(p, encoding="utf-8")); return {**m["weights_px"], "__scale__": float(m.get("scale", 1.0))}


def cost_matrix(Aa, Bb, prm: TrackingParams, py_, px_, nb=None, gate=None, app_w=None, extra=None, learned=None, con_raw=None):
    """비용 행렬. gate는 스칼라 또는 행(트랙)별 배열. 행별 gate가 넓으면 거리 비용을 gate 비율만큼 줄여
    (불확실한 트랙일수록 같은 거리의 벌점이 작음) 종료 비용과의 균형을 유지합니다."""
    gate = prm.gate if gate is None else gate
    g = np.broadcast_to(np.asarray(gate, float), (len(Aa),))[:, None]
    d = np.linalg.norm(np.stack([Aa[py_].values, Aa[px_].values], 1)[:, None] - Bb[["y_ht", "x_ht"]].values[None], axis=2)
    c = d * (prm.gate / g) if np.ndim(gate) else d.copy()
    if learned:          # 정답으로 학습한 비용 (거리 항은 위의 적응형 gate 비율 그대로)
        F = pair_features(Aa, Bb, d, nb, con_raw)
        for k, w in learned.items():
            if k not in ("d", "__scale__") and w > 0: c = c + w * F[k]
        c = c * learned.get("__scale__", 1.0); c[d > g] = np.inf
        return c, d
    c += prm.w_cell * (Aa.cell_id.values[:, None] != Bb.cell_id.values[None])
    c += prm.w_int * np.abs(np.log(Aa.mean_intensity.values[:, None] / Bb.mean_intensity.values[None]))
    c += prm.w_area * np.abs(np.log(Aa.area_px.values[:, None] / Bb.area_px.values[None]))
    if app_w:          # 학습된 외형 가중치 (이때 w_int·w_area 대신 사용)
        c -= prm.w_int * np.abs(np.log(Aa.mean_intensity.values[:, None] / Bb.mean_intensity.values[None]))
        c -= prm.w_area * np.abs(np.log(Aa.area_px.values[:, None] / Bb.area_px.values[None]))
        for k, w in app_w.items():
            col, kind = APPEARANCE[k]
            if w > 0: c += prm.appearance_scale * w * np.nan_to_num(_feat_diff(Aa[col].values[:, None], Bb[col].values[None], kind), nan=0.0)
    if extra is not None: c = c + extra
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
    """외형 특징을 쓰면 2-pass: 1차 연결로 외형 가중치를 학습한 뒤 다시 연결. 반환은 _link_frames와 같고, prm에는 영향 없음."""
    link_frames.last_appearance = None
    learned = load_learned(prm)
    if learned: return _link_frames(df, prm, progress, learned=learned)
    if not prm.appearance or "ht_ri_max" not in df:
        return _link_frames(df, prm, progress)
    L1, A1, P1 = _link_frames(df, prm, (lambda f, m: progress(f * .5, "1차 " + m)) if progress else None)
    W, stats = learn_appearance(df, L1, prm); link_frames.last_appearance = stats or None
    if not W: return L1, A1, P1
    return _link_frames(df, prm, (lambda f, m: progress(.5 + f * .5, "2차(외형) " + m)) if progress else None, app_w=W)


def _link_frames(df, prm: TrackingParams, progress=None, app_w=None, learned=None):
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
        con = con_raw = None
        if (prm.constellation and prm.w_con > 0) or (learned and learned.get("con", 0) > 0):
            c0, _ = cost_matrix(Aa, Bb, prm, "pred_y", "pred_x", gate=g); con_raw = constellation_matrix(Aa, Bb, prm, np.isfinite(c0))
            if not learned: con = prm.w_con * con_raw
        kw = dict(gate=g, app_w=app_w, extra=con, learned=learned, con_raw=con_raw)
        c, d = cost_matrix(Aa, Bb, prm, "pred_y", "pred_x", **kw); L1 = lap(c, prm.gate)          # 1차 연결
        dm = {ia[i]: pos[Bb.idx.values[j]] - pos[ia[i]] for i, j in L1}
        nb = neighbour_median(Aa, dm, prm); c, d = cost_matrix(Aa, Bb, prm, "pred_y", "pred_x", nb, **kw); L1 = lap(c, prm.gate)   # 이웃 일관성 반영
        srt2 = np.sort(np.where(np.isfinite(c), c, 1e6), 1)
        for i, j in L1:
            a, b = ia[i], Bb.idx.values[j]; mg = (srt2[i, 1] - c[i, j]) if c.shape[1] > 1 else 1e6
            links.append((a, b, c[i, j], d[i, j], *nb[i], mg))
            if prm.velocity_model:          # 연결된 검출로 속도·예측 오차 상태 전달
                uo = pos[b] - pos[a] - flow[a]; al = prm.vel_alpha
                u[b] = al * uo + (1 - al) * u[a] if has_u[a] else al * uo; has_u[b] = True
                rms[b] = np.sqrt(al * d[i, j] ** 2 + (1 - al) * rms[a] ** 2)
        srt = np.sort(np.where(np.isfinite(c), c, np.inf), 1)
        amb.append(np.mean(np.isfinite(srt[:, 0]) & (srt[:, 1] < srt[:, 0] + 1.0)) if c.shape[1] > 1 else 0)
        if progress: progress(t / (T - 1), f"연결 {t}/{T-1}")
    P = pd.DataFrame({"idx": df.idx.values, "pred_y": pred[:, 0], "pred_x": pred[:, 1], "gate_used": gate_used})
    return pd.DataFrame(links, columns=["a", "b", "cost", "pred_dist", "nb_dy", "nb_dx", "margin"]), np.array(amb), P


def build_tracks(df, links, prm: TrackingParams, pred=None, app_w=None):
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
    if prm.gap_closing and prm.gap_max > 1:
        mp = close_gaps(d, prm, app_w)
    elif prm.gap_closing:
        T = int(d.frame.max())
        ends = d.loc[d.groupby("track_id").frame.idxmax()]; sts = d.loc[d.groupby("track_id").frame.idxmin()]
        for t in range(1, T - 1):
            E = ends[ends.frame == t]; Sx = sts[sts.frame == t + 2]
            if len(E) == 0 or len(Sx) == 0: continue
            E2 = E.assign(gy=2 * E.pred_y - E.y_ht, gx=2 * E.pred_x - E.x_ht)
            gg = E.gate_used.values * 1.3 if prm.velocity_model else prm.gate * 1.3
            c, _ = cost_matrix(E2, Sx, prm, "gy", "gx", gate=gg, app_w=app_w, learned=load_learned(prm)); c = np.where(np.isinf(c), 1e6, c)
            for i, j in zip(*linear_sum_assignment(c)):
                if c[i, j] < 1e6: mp[Sx.track_id.values[j]] = E.track_id.values[i]

    def root(x):
        while x in mp: x = mp[x]
        return x
    d["track_id"] = pd.factorize(d.track_id.map(root), sort=True)[0] + 1
    return d.sort_values(["track_id", "frame"]).reset_index(drop=True), len(mp)


def close_gaps(d, prm: TrackingParams, app_w=None):
    """끊긴 조각 다시 잇기 (u-track식). 조각 끝 e(프레임 te)와 조각 시작 s(ts, 2 ≤ ts−te ≤ gap_max+1) 후보마다
      예측 거리 = 끝에서 앞으로 외삽(te의 다음 위치 예측으로 속도 추정)과 시작에서 뒤로 외삽(첫 이동)의 평균 오차,
      gate = gate_used(te) × √(누락+1) (최대 gate_max×1.5), 비용 = 거리×(gate/gate_g) + 세포 불일치 + 외형 차이(끝 3시점 평균 ↔ 시작 3시점 평균)
              + 누락 프레임당 벌점. 연결되는 후보끼리 묶음(연결 성분)마다 종료 비용(gate+2)을 둔 LAP로 1:1 최적 연결.
    반환 {시작 조각 track_id: 끝 조각 track_id}."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    g = d.sort_values(["track_id", "frame"]).groupby("track_id"); first = g.head(3); last = g.tail(3)
    E = g.tail(1).set_index("track_id"); S = g.head(1).set_index("track_id")
    fe = last.groupby("track_id"); fs = first.groupby("track_id")
    feat = {}
    for k, (col, kind) in APPEARANCE.items():
        if col in d: feat[k] = (fe[col].mean(), fs[col].mean(), kind)
    # 시작 조각의 첫 이동 (뒤로 외삽용)
    s2 = g.nth(1).set_index("track_id") if (g.size() > 1).any() else S.iloc[:0]
    vs = (s2[["y_ht", "x_ht"]] - S.loc[s2.index, ["y_ht", "x_ht"]].values).div(s2.frame - S.loc[s2.index, "frame"].values, axis=0)
    ve = E[["pred_y", "pred_x"]].values - E[["y_ht", "x_ht"]].values
    T = int(d.frame.max()); tree = {}; Sf = S.frame.values; Sp = S[["y_ht", "x_ht"]].values; Sid = S.index.values
    for f in np.unique(Sf): tree[f] = (cKDTree(Sp[Sf == f]), np.nonzero(Sf == f)[0])
    edges = []
    for ei, (eid, e) in enumerate(E.iterrows()):
        te = int(e.frame)
        for gap in range(2, prm.gap_max + 2):
            ts = te + gap
            if ts > T or ts not in tree: continue
            gg = min(e.gate_used * np.sqrt(gap), prm.gate_max * 1.5); fwd = np.array([e.y_ht, e.x_ht]) + gap * ve[ei]
            kt, ids = tree[ts]
            for j in kt.query_ball_point(fwd, gg + 5):
                si = ids[j]; sid = Sid[si]
                if sid == eid: continue
                ps = Sp[si]; df_ = np.linalg.norm(ps - fwd)
                if sid in vs.index and np.isfinite(vs.loc[sid]).all():
                    df_ = 0.5 * (df_ + np.linalg.norm(ps - gap * vs.loc[sid].values - np.array([e.y_ht, e.x_ht])))
                if df_ > gg: continue
                c = df_ * (prm.gate / gg) + prm.w_cell * (e.cell_id != S.cell_id.values[si]) + prm.gap_penalty * (gap - 1)
                if app_w:
                    for k, w in app_w.items():
                        if k in feat and w > 0:
                            a_, b_, kind = feat[k]; c += prm.appearance_scale * w * np.nan_to_num(_feat_diff(a_[eid], b_[sid], kind))
                else:
                    c += prm.w_int * abs(np.log(feat["cy5"][0][eid] / feat["cy5"][1][sid])) + prm.w_area * abs(np.log(feat["area"][0][eid] / feat["area"][1][sid]))
                edges.append((eid, sid, c))
    if not edges: return {}
    Ed = pd.DataFrame(edges, columns=["e", "s", "c"]).sort_values("c").drop_duplicates(["e", "s"])
    ue, us = {v: i for i, v in enumerate(Ed.e.unique())}, {v: i for i, v in enumerate(Ed.s.unique())}
    ne, ns = len(ue), len(us); r = Ed.e.map(ue).values; cidx = Ed.s.map(us).values + ne
    G = coo_matrix((np.ones(len(Ed)), (r, cidx)), shape=(ne + ns, ne + ns)); _, comp = connected_components(G, directed=False)
    Ed["comp"] = comp[r]; mp = {}; nc = prm.gate + 2
    for _, grp in Ed.groupby("comp"):
        es, ss = list(dict.fromkeys(grp.e)), list(dict.fromkeys(grp.s)); a, b = len(es), len(ss)
        M = np.full((a + b, b + a), 1e6); ie = {v: i for i, v in enumerate(es)}; is_ = {v: i for i, v in enumerate(ss)}
        for x in grp.itertuples(): M[ie[x.e], is_[x.s]] = x.c
        M[:a, b:] = np.where(np.eye(a) > 0, nc, 1e6); M[a:, :b] = np.where(np.eye(b) > 0, nc, 1e6); M[a:, b:] = np.where(M[:a, :b].T < 1e6, 0, 1e6)
        for i, j in zip(*linear_sum_assignment(M)):
            if i < a and j < b and M[i, j] < 1e6: mp[ss[j]] = es[i]
    return mp


def add_motion_columns(tr, dt_min):
    g = tr.groupby("track_id")
    for c in ["y_ht", "x_ht", "predF_y", "predF_x", "frame"]: tr["prev_" + c] = g[c].shift()
    one = np.where(tr.frame - tr.prev_frame == 1, 1, np.nan)
    tr["step_ht"] = np.hypot(tr.y_ht - tr.prev_y_ht, tr.x_ht - tr.prev_x_ht)
    tr["cell_carried_ht"] = np.hypot(tr.prev_predF_y - tr.prev_y_ht, tr.prev_predF_x - tr.prev_x_ht) * one
    tr["residual_ht"] = np.hypot(tr.y_ht - tr.prev_predF_y, tr.x_ht - tr.prev_predF_x) * one
    tr["time_min"] = (tr.frame - 1) * dt_min
    return tr.drop(columns=[c for c in tr.columns if c.startswith("prev_")])

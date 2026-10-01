"""정답(ground_truth.csv)으로 연결 비용 학습.

정답의 연속 연결(ok → ok)마다 '그 입자 다음 프레임 후보들'(예측 위치 근처 검출) 중 정답을 고르는 문제로 보고,
조건부 로지스틱(후보 중 하나를 고르는 softmax)으로 특징 가중치를 학습합니다.
  비용(후보) = Σ β_k · 특징_k     P(정답 = 후보) ∝ exp(−비용)
  특징은 tracking.pair_features와 같습니다 (거리, 거리², 크기·Cy5 밝기·HT RI·모양 차이, 세포 불일치, 이웃 이동과의 차이, 주변 배치).
가중치는 0 이상으로 제한하고, 거리 가중치로 나눠 px 단위(트래킹의 종료 비용 gate+2와 같은 척도)로 저장합니다.

정답이 적을 때 흔들리지 않도록 트래커의 확실한 연결(패턴 상관 ≥ 0.9, 2위 후보와 차이 큼)을 약한 가중치(합계가 정답의 절반)로 함께 씁니다.
정답 궤적 하나씩 빼고 학습해 빠진 궤적에서 정답 후보를 1순위로 고르는 비율(교차검증)을 현재 비용과 비교해 보고합니다.
"""
from __future__ import annotations
import json, pickle
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from .config import TrackingParams
from . import tracking
from .gt import load_gt

MODEL_DIR = Path.home() / ".particletracker"
DEFAULT_MODEL = MODEL_DIR / "learned_tracking.json"
FEATS = tracking.LEARN_FEATURES


def detections(folder):
    """결과 폴더 → 검출 표 (트래킹 df와 같은 열 이름) + 트래커 연결의 다음 검출 위치."""
    folder = Path(folder); P = pd.read_csv(folder / "tracks_points_A_B_HT.csv")
    D = pd.DataFrame(dict(frame=P.frame, label=P.roi_label, track_id=P.track_id, y=P.y_fl, x=P.x_fl, y_ht=P.y_ht, x_ht=P.x_ht,
                          area_px=P.area_px, mean_intensity=P.I_A_raw, eccentricity=P.eccentricity, cell_id=P.cell_id,
                          ht_ri_max=P.HT_RI_max * 1e4, ht_ri_mean=P.HT_RI_mean * 1e4, pat_score=P.get("pat_score", np.nan)))
    raw = folder / "_cache" / "tracks_raw.pkl"
    if raw.exists():
        R = pickle.load(open(raw, "rb"))
        cols = [c for c in ["pred_y", "pred_x", "gate_used"] if c in R]
        D = D.merge(R[["frame", "label"] + cols], on=["frame", "label"], how="left")
    for c, src in (("pred_y", "y_ht"), ("pred_x", "x_ht")):
        if c not in D: D[c] = D[src]
        D[c] = D[c].fillna(D[src])
    D = D.sort_values(["frame", "label"]).reset_index(drop=True); D["idx"] = np.arange(len(D))
    t = D.sort_values(["track_id", "frame"]); nx = t.groupby("track_id").idx.shift(-1); nf = t.groupby("track_id").frame.shift(-1)
    ok = (nf - t.frame == 1).values; D["next_idx"] = -1; D.loc[t.idx.values[ok], "next_idx"] = nx.values[ok].astype(int)
    return D


def _samples_for(D, pairs, prm: TrackingParams, weight, group):
    """pairs: [(a_idx, b_idx)] → [(특징 (n_cand, K), 정답 위치, 가중치, 그룹)]."""
    out = []; R = max(prm.gate_max, prm.gate) * 1.25
    byf = {f: g for f, g in D.groupby("frame")}; trees = {f: cKDTree(g[["y_ht", "x_ht"]].values) for f, g in byf.items()}
    nb_cache = {}
    for a, b in pairs:
        fa = int(D.frame[a]); fb = fa + 1
        if fb not in byf: continue
        Ba = byf[fa]; Bb = byf[fb]; pa = D.loc[a, ["pred_y", "pred_x"]].values.astype(float)
        cand = list(trees[fb].query_ball_point(pa, R)); pos_b = Bb.index.get_loc(b)
        if pos_b not in cand: cand.append(pos_b)
        if len(cand) < 2: continue
        Bc = Bb.iloc[cand]; Aa = D.loc[[a]]
        d = np.linalg.norm(Bc[["y_ht", "x_ht"]].values - pa, axis=1)[None]
        if fa not in nb_cache:
            dm = {i: D.loc[j, ["y_ht", "x_ht"]].values - D.loc[i, ["y_ht", "x_ht"]].values for i, j in zip(Ba.idx, Ba.next_idx) if j >= 0}
            nb_cache[fa] = dict(zip(Ba.idx, tracking.neighbour_median(Ba, dm, prm)))
        nb = np.array([nb_cache[fa][a]])
        fin = np.zeros((len(Ba), len(Bb)), bool); fin[Ba.index.get_loc(a), cand] = True
        con = tracking.constellation_matrix(Ba, Bb, prm, fin)[Ba.index.get_loc(a), cand][None]
        F = tracking.pair_features(Aa, Bc, d, nb, con)
        X = np.stack([F[k][0] for k in FEATS], 1); out.append((X, cand.index(pos_b), weight, group))
    return out


def build_samples(folders, prm: TrackingParams, self_ratio=0.5, max_self=1500, seed=0):
    """정답 폴더들 → 학습 표본. 반환 (표본 목록, 요약)."""
    S = []; info = []
    for fo in folders:
        gt = load_gt(fo); gt = gt[gt.status != "auto"]
        if not len(gt) or not (Path(fo) / "tracks_points_A_B_HT.csv").exists(): continue
        D = detections(fo); key = {(int(f), int(l)): i for i, f, l in zip(D.idx, D.frame, D.label)}; n0 = len(S)
        for gid, s in gt.sort_values(["gt_id", "frame"]).groupby("gt_id"):
            s = s.reset_index(drop=True); pairs = []
            for i in range(len(s) - 1):
                r0, r1 = s.iloc[i], s.iloc[i + 1]
                if r0.status == "ok" and r1.status == "ok" and r1.frame - r0.frame == 1 and r0.roi_label >= 0 and r1.roi_label >= 0:
                    a, b = key.get((int(r0.frame), int(r0.roi_label))), key.get((int(r1.frame), int(r1.roi_label)))
                    if a is not None and b is not None: pairs.append((a, b))
            S += _samples_for(D, pairs, prm, 1.0, f"{fo}#{gid}")
        n_gt = len(S) - n0
        conf = D[(D.next_idx >= 0) & (D.pat_score >= 0.9)]
        rng = np.random.default_rng(seed); take = conf.iloc[rng.permutation(len(conf))[:max_self]]
        ss = _samples_for(D, list(zip(take.idx, take.next_idx)), prm, 1.0, "self")
        ss = [x for x in ss if True][: max_self]
        w = self_ratio * max(n_gt, 1) / max(len(ss), 1); S += [(X, y, w, g) for X, y, _, g in ss]
        info.append(dict(folder=str(fo), gt_samples=n_gt, self_samples=len(ss)))
    return S, info


def fit(samples, l2=1e-2):
    """조건부 로지스틱, β ≥ 0. 반환 β (FEATS 순서, 원래 척도)."""
    Xs = np.concatenate([X for X, _, _, _ in samples]); sc = Xs.std(0); sc[sc < 1e-9] = 1.0
    data = [(X / sc, y, w) for X, y, w, _ in samples]; W = sum(w for _, _, w in data)

    def f(b):
        L = 0.0; G = np.zeros_like(b)
        for X, y, w in data:
            c = X @ b; m = c.min(); e = np.exp(-(c - m)); p = e / e.sum()
            L += w * (c[y] - m + np.log(e.sum())); G += w * (X[y] - p @ X)
        return L / W + l2 * b @ b, G / W + 2 * l2 * b
    b0 = np.zeros(len(FEATS)); b0[FEATS.index("d")] = 1.0
    r = minimize(f, b0, jac=True, method="L-BFGS-B", bounds=[(0, None)] * len(FEATS))
    return r.x / sc


def calibrate_scale(samples, w, bb, q=0.9, mult=2.0):
    """조건부 로지스틱은 후보끼리의 '순위'만 학습하고 '연결할지 말지'(종료 비용 gate+2와의 비교)는 정하지 못합니다.
    순위는 그대로 두고 비용 전체에 곱할 배율을, 정답·확실한 연결의 비용 분포(상위 q 분위)가 외형 특징 없는 현재 비용과 같아지도록 맞춘 뒤
    mult(=2, 첫 정답 14궤적에서 궤적별 교차검증 트래킹으로 정함: ×1.5 91.6%, ×2 92.2%, ×3 89.4%)를 곱합니다."""
    tl = np.array([X[y] @ w for X, y, _, _ in samples]); tb = np.array([X[y] @ bb for X, y, _, _ in samples])
    return float(mult * np.quantile(tb, q) / max(np.quantile(tl, q), 1e-9))


def top1(samples, beta):
    ok = [int(np.argmin(X @ beta) == y) for X, y, _, g in samples if g != "self"]
    return float(np.mean(ok)) if ok else np.nan, len(ok)


def baseline_beta(prm: TrackingParams):
    """현재 손으로 정한 비용(외형 특징 제외)을 같은 특징 공간으로 표현 — 비교 기준."""
    b = np.zeros(len(FEATS))
    for k, v in dict(d=1.0, cell=prm.w_cell, cy5=prm.w_int, area=prm.w_area, nb=prm.w_nb).items(): b[FEATS.index(k)] = v
    return b


def train(folders, prm: TrackingParams, out_path=DEFAULT_MODEL, log=print):
    """학습 → 교차검증 → 모델 저장. 반환 보고 dict."""
    S, info = build_samples(folders, prm)
    gt_groups = sorted({g for *_, g in S if g != "self"})
    if len(gt_groups) < 3: raise ValueError(f"학습할 정답이 부족합니다 (정답 궤적 {len(gt_groups)}개, 3개 이상 필요)")
    log(f"표본: 정답 연결 {sum(1 for *_, g in S if g != 'self')}개 (궤적 {len(gt_groups)}개), 확실한 트래커 연결 {sum(1 for *_, g in S if g == 'self')}개")
    hit_l = hit_b = n = 0; bb = baseline_beta(prm)
    for g in gt_groups:                     # 궤적 하나씩 빼고 학습
        tr = [s for s in S if s[3] != g]; te = [s for s in S if s[3] == g]
        if not te: continue
        beta = fit(tr); a, k = top1(te, beta); b_, _ = top1(te, bb)
        hit_l += a * k; hit_b += b_ * k; n += k
    beta = fit(S); d = beta[FEATS.index("d")]
    if d <= 1e-9: raise ValueError("거리 가중치가 0으로 학습됐습니다 — 정답이 더 필요합니다")
    wpx = {k: float(v / d) for k, v in zip(FEATS, beta)}; acc_all, n_all = top1(S, beta); acc_base, _ = top1(S, bb)
    scale = calibrate_scale(S, np.array([wpx[k] for k in FEATS]), bb)
    rep = dict(trained=datetime.now().isoformat(timespec="seconds"), folders=info, n_gt_links=n_all, n_trajectories=len(gt_groups),
               cv_top1_learned=hit_l / n if n else np.nan, cv_top1_current=hit_b / n if n else np.nan,
               fit_top1_learned=acc_all, fit_top1_current=acc_base, weights_px=wpx, scale=scale)
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    hist = []
    if out_path.exists():
        try: old = json.load(open(out_path, encoding="utf-8")); hist = old.get("history", []) + [{k: old[k] for k in ("trained", "n_gt_links", "cv_top1_learned", "cv_top1_current") if k in old}]
        except Exception: hist = []   # noqa: BLE001
    rep["history"] = hist[-20:]
    out_path.write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    return rep


def report_text(rep) -> str:
    f = lambda v: "-" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v * 100:.1f}%"
    w = rep["weights_px"]; names = dict(d="거리", d2="거리²", area="크기", cy5="Cy5 밝기", ht_max="HT RI 최대", ht_mean="HT RI 평균", ecc="모양",
                                         cell="세포 불일치", nb="이웃 이동 차이", con="주변 배치")
    t = (f"정답 궤적 {rep['n_trajectories']}개 · 정답 연결 {rep['n_gt_links']}개로 학습\n"
         f"교차검증(궤적 하나씩 빼고): 정답 후보를 1순위로 고른 비율  학습 모델 {f(rep['cv_top1_learned'])}  vs  현재 비용 {f(rep['cv_top1_current'])}\n"
         "학습된 가중치 (거리 1 px 기준): " + ", ".join(f"{names[k]} {v:.2f}" for k, v in w.items() if v > 1e-3))
    if rep.get("history"):
        t += "\n이전 학습: " + " → ".join(f"{h.get('n_gt_links', '?')}연결 {f(h.get('cv_top1_learned'))}" for h in rep["history"][-3:])
    return t

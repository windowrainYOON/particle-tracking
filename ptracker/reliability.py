"""특이 입자(및 트랙 구간)의 신뢰도: 결정 구간이 끝까지 같은 입자로 이어졌을 확률.

트래커는 연결마다 '2위 후보와의 비용 차이(마진)'를 남깁니다 (link_margin). 정답에서 마진이 작을수록 오연결이 많았습니다
(첫 정답 P01 179연결: 마진 <2 40%, 2–5 33%, 5–10 7%, ≥10 0%; 오류 구분 AUC 0.95).
구간의 연결마다 '맞을 확률' = 1 − (마진 구간의 오류율)을 곱하고, 누락 프레임을 건너뛴 연결과 합쳐짐 시점은 따로 깎습니다.
오류율은 정답(ground_truth.csv)이 있으면 그것으로 다시 계산하고, 없으면 첫 정답의 값을 씁니다.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd

BINS = [(-np.inf, 2, "<2"), (2, 5, "2–5"), (5, 10, "5–10"), (10, np.inf, "≥10")]
PRIOR = {"<2": (10, 4), "2–5": (9, 3), "5–10": (29, 2), "≥10": (125, 0)}      # (연결 수, 오류 수) — P01 첫 정답, 2026-10-01


def bin_of(m):
    for lo, hi, nm in BINS:
        if lo <= m < hi: return nm
    return BINS[-1][2]


def _rates(counts):
    return {k: (e + 1) / (n + 2) for k, (n, e) in counts.items()}          # 라플라스 보정


def link_outcomes(points: pd.DataFrame, gt: pd.DataFrame) -> pd.DataFrame:
    """정답 연속 연결(ok→ok)마다 트래커 연결의 마진과 오류 여부."""
    from .gt import _assign_tracks
    if "link_margin" not in points or not len(gt): return pd.DataFrame(columns=["margin", "err"])
    g = gt[gt.status != "auto"].sort_values(["gt_id", "frame"]).copy(); g["tid"] = _assign_tracks(points, g)
    mk = points.set_index(["frame", "roi_label"]).link_margin; mk = mk[~mk.index.duplicated()]; rows = []
    for _, s in g.groupby("gt_id"):
        s = s.reset_index(drop=True); ok = s[(s.status == "ok") & (s.tid >= 0)]
        for a, b in zip(ok.index[:-1], ok.index[1:]):
            if s.frame[b] - s.frame[a] == 1 and b - a == 1:
                m = mk.get((int(s.frame[a]), int(s.roi_label[a])), np.nan)
                if np.isfinite(m): rows.append(dict(margin=m, err=int(s.tid[a] != s.tid[b])))
    return pd.DataFrame(rows, columns=["margin", "err"])


def calibrate(folders=(), min_links=40):
    """정답이 있는 결과 폴더들 → 마진 구간별 오류율. 정답 연결이 min_links 미만이면 기본값(PRIOR). 반환 (오류율, 근거 설명)."""
    from .gt import load_gt
    parts = []
    for fo in folders or []:
        p = Path(fo) / "tracks_points_A_B_HT.csv"; gt = load_gt(fo)
        if p.exists() and len(gt):
            pts = pd.read_csv(p)
            if "link_margin" in pts: parts.append(link_outcomes(pts, gt))
    L = pd.concat(parts) if parts else pd.DataFrame(columns=["margin", "err"])
    if len(L) < min_links: return _rates(PRIOR), f"기본 오류율 (첫 정답 179연결; 현재 결과로 확인된 정답 연결 {len(L)}개)"
    counts = {nm: (int((L.margin.map(bin_of) == nm).sum()), int(L[L.margin.map(bin_of) == nm].err.sum())) for _, _, nm in BINS}
    return _rates(counts), f"현재 결과의 정답 연결 {len(L)}개로 계산"


def window_reliability(t: pd.DataFrame, f0, f1, perr, gap_err=0.5, merged_err=0.3):
    """트랙 t의 [f0, f1) 구간에서 시작하는 연결들의 '모두 맞을 확률'. 반환 dict."""
    t = t.sort_values("frame"); nf = t.frame.shift(-1); w = (t.frame >= f0) & (t.frame < f1) & nf.notna()
    p = 1.0; risky = gaps = 0; mins = np.inf
    ver = t.verified.astype(bool) if "verified" in t else pd.Series(False, index=t.index); vnext = ver.shift(-1, fill_value=False)
    for m, gap, v in zip(t.link_margin[w] if "link_margin" in t else [np.nan] * w.sum(), (nf - t.frame)[w], (ver & vnext)[w]):
        if v: continue                                       # 사용자가 교정·확인한 두 점 사이 연결은 확실
        if gap > 1 or not np.isfinite(m): p *= 1 - gap_err; gaps += 1; continue
        p *= 1 - perr[bin_of(m)]; risky += m < 5; mins = min(mins, m)
    nm = int(t.merged[(t.frame >= f0) & (t.frame <= f1)].astype(bool).sum()) if "merged" in t else 0
    p *= (1 - merged_err) ** nm
    return dict(reliability=p, risky_links=risky, gap_links=gaps, merged_frames=nm, min_margin=mins if np.isfinite(mins) else np.nan)

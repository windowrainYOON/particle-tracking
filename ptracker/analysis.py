"""6단계: 여러 데이터셋 통합, 특이 입자(I(pHrodo)/I(Cy5) 한 번 증가 후 감소·유지) 색인."""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
from .config import RiseFallParams


def wilson_ci(x, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    p = x / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def combine(dataset_dirs: dict) -> pd.DataFrame:
    """{name: 결과 폴더} → 통합 point 표 (dataset, track_uid 열 추가)."""
    parts = []
    for name, d in dataset_dirs.items():
        f = Path(d) / "tracks_points_A_B_HT.csv"
        if not f.exists(): continue
        t = pd.read_csv(f); t.insert(0, "dataset", name); t.insert(1, "track_uid", f"{name}::" + t.track_id.astype(str)); parts.append(t)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def classify_track(t0, prm: RiseFallParams, dt_min=20.0) -> dict:
    t = t0.dropna(subset=["ratio_BA"]).sort_values("frame"); x = t.ratio_BA.values; f = t.frame.values; n = len(x)
    o = dict(n_valid=n, base=np.nan, peak=np.nan, end=np.nan, peak_frame=np.nan, drop_frame=np.nan, decline_frame=np.nan,
             rise=False, abrupt=False, fall=False, category="no_rise", n_post=np.nan, max_post_over_peak=np.nan,
             frac_post_raw_high=np.nan, rebound=False, one_frame_spike=False, pre_peak_oscillation=False, keep=False)
    if n < prm.min_points: return o
    s = pd.Series(x).rolling(3, center=True, min_periods=1).median().values
    base = np.mean(s[:3]); ip = int(np.argmax(s)); pk = s[ip]; end = np.mean(s[-3:])
    o.update(base=base, peak=pk, end=end, peak_frame=f[ip])
    if not (ip >= 2 and pk >= prm.rise_x * max(base, prm.floor) and pk - base >= prm.rise_delta and n - ip >= 4): return o
    o["rise"] = True; o["category"] = "rise_sustained"
    for k in range(ip, n - 2):   # 급격한 감소
        if x[k] >= .6 * pk and f[k + 1] - f[k] == 1 and x[k + 1] <= prm.drop_frac * x[k] and np.mean(x[k + 1:k + 4]) <= prm.drop_frac * x[k] \
                and x[k] - np.mean(x[k + 1:k + 4]) >= prm.rise_delta:
            o["abrupt"] = True; o["drop_frame"] = f[k + 1]; break
    o["fall"] = (end <= prm.fall_frac * pk) and (pk - end >= prm.rise_delta)
    if o["abrupt"]: o["category"] = "abrupt_drop"
    elif o["fall"]: o["category"] = "gradual_decline"
    else: return o
    idec = list(f).index(o["drop_frame"]) if o["abrupt"] else next((k for k in range(ip + 1, n) if s[k] <= prm.fall_frac * pk), n)
    ps, px = s[idec:], x[idec:]; npost = len(ps)
    o["decline_frame"] = f[idec] if idec < n else np.nan; o["n_post"] = npost
    o["max_post_over_peak"] = ps.max() / pk if npost else np.nan
    o["frac_post_raw_high"] = float(np.mean(px > prm.rebound_frac * pk)) if npost else np.nan
    o["rebound"] = bool(npost and ps.max() > prm.rebound_frac * pk)
    mid = base + .5 * (pk - base); best = run = 0; prev = None
    for hh, ff in zip(x[:idec] >= mid, f[:idec]):
        run = run + 1 if (hh and prev is not None and ff - prev == 1 and run > 0) else (1 if hh else 0); prev = ff; best = max(best, run)
    pre = x[:idec]; pf = f[:idec]; im = int(np.argmax(pre))
    nbv = [pre[j] for j in (im - 1, im + 1) if 0 <= j < len(pre) and abs(pf[j] - pf[im]) == 1]
    o["one_frame_spike"] = (best < prm.min_high_frames) or (len(nbv) == 0 or max(nbv) < .5 * pre[im])
    seen = False
    for k in range(ip):
        if s[k] >= .6 * pk: seen = True
        elif seen and s[k] <= .4 * pk: o["pre_peak_oscillation"] = True; break
    o["keep"] = (not o["rebound"]) and (not o["pre_peak_oscillation"]) and npost >= prm.min_post and (not o["one_frame_spike"]) \
        and o["frac_post_raw_high"] <= prm.rebound_raw_max
    return o


def _reason(r):
    if r.category not in ("abrupt_drop", "gradual_decline"): return ""
    if r.keep: return "유지(선별)"
    rs = []
    if r.rebound: rs.append("감소 후 재증가")
    if r.frac_post_raw_high > .15: rs.append("감소 후 원본값 반복 상승")
    if r.n_post < 3: rs.append("감소 후 관찰 부족")
    if r.one_frame_spike: rs.append("1프레임 반짝 신호")
    if r.pre_peak_oscillation: rs.append("최고점 전 오르내림")
    return ", ".join(rs)


def risefall(points: pd.DataFrame, prm: RiseFallParams, dt_min=20.0, groups=("inside", "outside")):
    """points: (통합) point 표. 반환 (트랙별 분류표, 데이터셋·그룹별 분율표)."""
    if "dataset" not in points: points = points.assign(dataset="dataset")
    if "track_uid" not in points: points = points.assign(track_uid=points.dataset + "::" + points.track_id.astype(str))
    L = points[points.group.isin(groups)]
    rows = [dict(dataset=t.dataset.iloc[0], track_uid=u, track_id=t.track_id.iloc[0], group=t.group.iloc[0], **classify_track(t, prm, dt_min))
            for u, t in L.groupby("track_uid")]
    R = pd.DataFrame(rows)
    if R.empty: return R, pd.DataFrame()
    R["candidate"] = R.category.isin(["abrupt_drop", "gradual_decline"]); R["filter_result"] = R.apply(_reason, axis=1)
    for c in ["peak", "drop", "decline"]: R[f"{c}_time_h"] = (R[f"{c}_frame"] - 1) * dt_min / 60
    out = []
    for ds in list(R.dataset.unique()) + ["전체"]:
        for g in groups:
            s = R[(R.group == g) & ((R.dataset == ds) | (ds == "전체"))]; n = len(s); k = int(s.keep.sum()); lo, hi = wilson_ci(k, n)
            out.append(dict(dataset=ds, group=g, n_tracks=n, rise=int(s.rise.sum()), candidates=int(s.candidate.sum()), kept=k,
                            kept_abrupt=int((s.keep & (s.category == "abrupt_drop")).sum()), kept_gradual=int((s.keep & (s.category == "gradual_decline")).sum()),
                            candidate_pct=s.candidate.mean() * 100 if n else np.nan, kept_pct=k / n * 100 if n else np.nan,
                            kept_pct_ci_low=lo * 100, kept_pct_ci_high=hi * 100))
    return R, pd.DataFrame(out)

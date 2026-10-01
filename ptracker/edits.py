"""특이 입자 경로 교정 (track_edits.csv).

교정한 트랙은 결과를 읽을 때(analysis.combine) 원래 점 대신 교정한 경로로 바뀝니다.
  · 검출을 고른 프레임: 그 검출의 측정값을 그대로 씀 (어느 트랙에 속했든)
  · 빈 곳을 찍은 프레임: 그 자리에서 원형 ROI로 측정한 값 (measure_free)을 파일에 저장해 씀
  · status: orig(트래커 위치, 확인 안 함) | edit(사용자가 고름) | ok(사용자가 맞다고 확인)
교정·확인한 점은 verified=True가 되어 신뢰도 계산에서 '확실한 연결'로 취급되고, 정답(ground_truth.csv)에도 추가됩니다.
"""
from __future__ import annotations
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd

EDIT_FILE = "track_edits.csv"
MEAS_COLS = ["I_A_raw", "bg_A", "I_A", "I_B_raw", "bg_B", "I_B", "I_B_max", "ratio_BA", "HT_RI_mean", "HT_RI_max", "area_px", "radius_equiv_px"]
EDIT_COLS = ["track_id", "frame", "roi_label", "x_fl", "y_fl", "status"] + MEAS_COLS + ["updated"]


def load_edits(folder) -> pd.DataFrame:
    f = Path(folder) / EDIT_FILE
    return pd.read_csv(f) if f.exists() else pd.DataFrame(columns=EDIT_COLS)


def save_track(folder, track_id: int, path: pd.DataFrame):
    """한 트랙의 교정 경로 저장 (그 트랙의 이전 교정은 바꿈). path: frame, roi_label, x_fl, y_fl, status (+ 측정 열)."""
    E = load_edits(folder); E = E[E.track_id != track_id]
    p = path.copy(); p["track_id"] = int(track_id); p["updated"] = datetime.now().isoformat(timespec="seconds")
    for c in EDIT_COLS:
        if c not in p: p[c] = np.nan
    E = pd.concat([E, p[EDIT_COLS]], ignore_index=True) if len(E) else p[EDIT_COLS]
    E.to_csv(Path(folder) / EDIT_FILE, index=False)


def apply_edits(points: pd.DataFrame, folder) -> pd.DataFrame:
    """결과 폴더의 교정을 point 표에 반영. edited·verified 열 추가."""
    E = load_edits(folder); P = points.copy(); P["edited"] = False; P["verified"] = False
    if not len(E): return P
    meas = P.drop_duplicates(["frame", "roi_label"]).set_index(["frame", "roi_label"])
    new = []
    for tid, e in E.groupby("track_id"):
        old = P[P.track_id == tid]; grp = old.group.iloc[0] if len(old) and "group" in old else "inside"
        for r in e.itertuples():
            key = (int(r.frame), int(r.roi_label)) if pd.notna(r.roi_label) else None
            if key is not None and key[1] >= 0 and key in meas.index: row = meas.loc[key].to_dict()
            else:
                row = {c: getattr(r, c) for c in MEAS_COLS}; row.update(x_fl=r.x_fl, y_fl=r.y_fl)
            row.update(frame=int(r.frame), roi_label=int(r.roi_label) if pd.notna(r.roi_label) else -1, track_id=int(tid), group=grp, edited=True,
                       verified=r.status in ("edit", "ok"))
            if "time_min" in P and len(P): row["time_min"] = (int(r.frame) - 1) * (P.time_min / (P.frame - 1).replace(0, np.nan)).median()
            new.append(row)
        P = P[P.track_id != tid]
    N = pd.DataFrame(new)
    for c in P.columns:
        if c not in N: N[c] = np.nan
    return pd.concat([P, N[P.columns]], ignore_index=True)


def measure_free(ctx, src, frame, x, y, radius, prm):
    """검출이 없는 곳을 찍었을 때: (x, y) 원형 ROI(반지름 radius)로 측정. 다른 입자 ROI는 배경 고리에서 빠짐 (측정 단계와 같은 함수)."""
    from .measurement import measure_frame
    z = frame - 1; lab = None; cache = Path(ctx.folder) / "_cache" / "LAB.npy"
    if cache.exists(): lab = np.array(np.load(cache, mmap_mode="r")[z])
    elif (Path(ctx.folder) / "particle_labels.tif").exists():
        import tifffile; lab = tifffile.imread(Path(ctx.folder) / "particle_labels.tif", key=z)
    a = src.frame("cy5", z); dy_b, dx_b = ctx.shift_B(frame); b = np.roll(src.frame("phrodo", z), (dy_b, dx_b), (0, 1)); ht = src.frame("ht", z)
    if lab is None: lab = np.zeros(a.shape, np.int32)
    lab = lab.astype(np.int32); new = int(lab.max()) + 1; yy, xx = np.ogrid[:lab.shape[0], :lab.shape[1]]
    disk = ((yy - y) ** 2 + (xx - x) ** 2 <= radius ** 2) & (lab == 0); lab[disk] = new
    reg = ctx.reg; dyh = reg.dy_ht.values[z] if reg is not None else 0.0; dxh = reg.dx_ht.values[z] if reg is not None else 0.0
    rec = measure_frame(lab, a, b, ht, dyh, dxh, ht.shape[0] / a.shape[0], prm, labels=[new])
    if new not in rec: return {c: np.nan for c in MEAS_COLS}
    IA_raw, IB_raw, IB_max, bgA, bgB, htm, htx = rec[new]; ia, ib = IA_raw - bgA, IB_raw - bgB
    return dict(I_A_raw=IA_raw, bg_A=bgA, I_A=ia, I_B_raw=IB_raw, bg_B=bgB, I_B=ib, I_B_max=IB_max, ratio_BA=ib / ia if ia > prm.min_ia else np.nan,
                HT_RI_mean=htm, HT_RI_max=htx, area_px=int(disk.sum()), radius_equiv_px=float(radius))

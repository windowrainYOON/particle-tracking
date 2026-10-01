"""정답(ground truth) 궤적 저장과 트래킹 평가.

정답 파일: <데이터셋 결과 폴더>/ground_truth.csv
  gt_id, frame, x_fl, y_fl, roi_label(검출 번호, 직접 찍은 위치면 -1), status, source_track, updated
  status: auto(아직 확인 안 함) | ok(이 위치가 맞음) | merged(다른 입자와 합쳐짐/가림) | absent(보이지 않음) | unsure(판단 불가)

평가 (status=auto는 쓰지 않음)
  · 연속 프레임 정답 쌍(ok→ok)마다 트래커가
      correct = 같은 트랙으로 이었음 / switch = 다른 입자로 이었음 / break = 이 입자를 이어가지 못함
  · 누락 구간(ok → merged/absent/unsure … → ok)을 건너 같은 트랙으로 다시 이었는지 (재식별)
  · 정답의 merged 표시와 트래커의 merged 표시 비교 (merged 열이 있을 때)
"""
from __future__ import annotations
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd
from scipy.spatial import cKDTree

GT_FILE = "ground_truth.csv"
GT_COLS = ["gt_id", "frame", "x_fl", "y_fl", "roi_label", "status", "source_track", "updated"]
STATUS_LABELS = {"auto": "미확인", "ok": "맞음", "merged": "합쳐짐/가림", "absent": "안 보임", "unsure": "모름"}


def load_gt(folder) -> pd.DataFrame:
    f = Path(folder) / GT_FILE
    if not f.exists(): return pd.DataFrame(columns=GT_COLS)
    d = pd.read_csv(f)
    for c in GT_COLS:
        if c not in d: d[c] = np.nan
    return d[GT_COLS]


def save_gt(folder, gt: pd.DataFrame):
    gt.sort_values(["gt_id", "frame"]).to_csv(Path(folder) / GT_FILE, index=False)


def now(): return datetime.now().isoformat(timespec="seconds")


def new_from_track(gt: pd.DataFrame, t: pd.DataFrame, source: str) -> tuple[pd.DataFrame, int]:
    """트래커 트랙 하나로 정답 궤적을 시작합니다 (모든 프레임 status=auto). 반환 (gt, gt_id)."""
    gid = int(gt.gt_id.max()) + 1 if len(gt) else 1
    rows = pd.DataFrame(dict(gt_id=gid, frame=t.frame.values.astype(int), x_fl=t.x_fl.values, y_fl=t.y_fl.values,
                             roi_label=t.roi_label.values.astype(int), status="auto", source_track=source, updated=now()))
    return pd.concat([gt, rows], ignore_index=True) if len(gt) else rows, gid


def set_point(gt, gid, frame, x=None, y=None, label=None, status="ok"):
    m = (gt.gt_id == gid) & (gt.frame == frame)
    if m.any():
        i = gt.index[m][0]
        if x is not None: gt.loc[i, ["x_fl", "y_fl", "roi_label"]] = [x, y, -1 if label is None else int(label)]
        gt.loc[i, ["status", "updated"]] = [status, now()]
    else:
        if x is None:          # 위치 없이 상태만 (예: 안 보임) → 가장 가까운 프레임 위치를 참고값으로
            g = gt[gt.gt_id == gid]; k = (g.frame - frame).abs().idxmin() if len(g) else None
            x, y = (gt.loc[k, "x_fl"], gt.loc[k, "y_fl"]) if k is not None else (np.nan, np.nan); label = -1
        src = gt.loc[gt.gt_id == gid, "source_track"].iloc[0] if (gt.gt_id == gid).any() else ""
        gt = pd.concat([gt, pd.DataFrame([dict(gt_id=gid, frame=int(frame), x_fl=x, y_fl=y, roi_label=-1 if label is None else int(label),
                                               status=status, source_track=src, updated=now())])], ignore_index=True)
    return gt


def reseed_after(gt, gid, frame, t: pd.DataFrame):
    """frame에서 다른 입자로 고친 뒤: 그 뒤의 '미확인(auto)' 점을 새로 고른 입자의 트래커 트랙 t(그 프레임 이후)로 바꿉니다.
    이미 확인한 점은 그대로 둡니다."""
    keep = ~((gt.gt_id == gid) & (gt.frame > frame) & (gt.status == "auto"))
    done = set(gt[(gt.gt_id == gid) & (gt.frame > frame)].frame) - set(gt[~keep].frame)
    t = t[(t.frame > frame) & ~t.frame.isin(done)]; src = gt.loc[gt.gt_id == gid, "source_track"].iloc[0]
    rows = pd.DataFrame(dict(gt_id=gid, frame=t.frame.values.astype(int), x_fl=t.x_fl.values, y_fl=t.y_fl.values, roi_label=t.roi_label.values.astype(int),
                             status="auto", source_track=src, updated=now()))
    return pd.concat([gt[keep], rows], ignore_index=True) if len(rows) else gt[keep].reset_index(drop=True)


def truncate(gt, gid, frame):
    """정답 궤적을 이 프레임 다음부터 지웁니다 (여기서 끝)."""
    return gt[~((gt.gt_id == gid) & (gt.frame > frame))].reset_index(drop=True)


def _assign_tracks(points, gt, snap_px=3.0):
    """정답 점 → 트래커 트랙 번호. roi_label이 있으면 (frame, label)로, 없거나 못 찾으면 가장 가까운 검출(snap_px 이내)."""
    key = points.set_index(["frame", "roi_label"]).track_id
    key = key[~key.index.duplicated()]
    tid = np.full(len(gt), -1)
    for i, r in enumerate(gt.itertuples()):
        k = (int(r.frame), int(r.roi_label)) if pd.notna(r.roi_label) else None
        if k is not None and k[1] >= 0 and k in key.index: tid[i] = key[k]; continue
        d = points[points.frame == r.frame]
        if len(d) and np.isfinite(r.x_fl):
            dist, j = cKDTree(d[["x_fl", "y_fl"]].values).query([r.x_fl, r.y_fl])
            if dist <= snap_px: tid[i] = d.track_id.values[j]
    return tid


def evaluate(points: pd.DataFrame, gt: pd.DataFrame):
    """points: track_id, frame, roi_label, x_fl, y_fl (+ merged). 반환 (요약 dict, 궤적별 표)."""
    g = gt[gt.status != "auto"].sort_values(["gt_id", "frame"]).copy()
    if not len(g): return dict(n_trajectories=0), pd.DataFrame()
    g["tid"] = _assign_tracks(points, g)
    has = {(int(a), int(b)) for a, b in zip(points.track_id, points.frame)}
    rows = []; tot = dict(correct=0, switch=0, brk=0, gap_total=0, gap_same=0)
    for gid, s in g.groupby("gt_id"):
        s = s.reset_index(drop=True); c = dict(correct=0, switch=0, brk=0, gap_total=0, gap_same=0)
        ok = s[(s.status == "ok") & (s.tid >= 0)]
        for a, b in zip(ok.index[:-1], ok.index[1:]):
            fa, fb = s.frame[a], s.frame[b]
            if fb - fa == 1 and b - a == 1:
                if s.tid[a] == s.tid[b]: c["correct"] += 1
                elif (int(s.tid[a]), int(fb)) in has: c["switch"] += 1
                else: c["brk"] += 1
            else:          # 사이에 merged/absent/unsure가 있는 누락 구간
                c["gap_total"] += 1; c["gap_same"] += int(s.tid[a] == s.tid[b])
        okt = ok.tid.values; dom = pd.Series(okt).mode().iloc[0] if len(okt) else -1
        rows.append(dict(gt_id=gid, source_track=s.source_track.iloc[0], frames=len(s), ok=int((s.status == "ok").sum()),
                         merged=int((s.status == "merged").sum()), absent=int((s.status == "absent").sum()),
                         main_track=int(dom), main_track_frac=float(np.mean(okt == dom)) if len(okt) else np.nan, n_tracks=len(set(okt)), **c))
        for k in tot: tot[k] += c[k]
    T = pd.DataFrame(rows); n = tot["correct"] + tot["switch"] + tot["brk"]
    summ = dict(n_trajectories=len(T), n_links=n, link_accuracy=tot["correct"] / n if n else np.nan,
                switch_rate=tot["switch"] / n if n else np.nan, break_rate=tot["brk"] / n if n else np.nan,
                gap_total=tot["gap_total"], gap_reidentified=tot["gap_same"] / tot["gap_total"] if tot["gap_total"] else np.nan,
                mean_main_track_frac=float(T.main_track_frac.mean()))
    if "merged" in points:          # 트래커의 merged 표시가 정답과 맞는지
        mk = points.set_index(["frame", "roi_label"]).merged; mk = mk[~mk.index.duplicated()]
        lab = g[g.status.isin(["ok", "merged"]) & (g.roi_label >= 0)]
        pred = np.array([bool(mk.get((int(f), int(l)), False)) for f, l in zip(lab.frame, lab.roi_label)]); true = (lab.status == "merged").values
        if true.any() or pred.any():
            summ.update(merged_true=int(true.sum()), merged_flagged=int(pred.sum()),
                        merged_precision=float((pred & true).sum() / pred.sum()) if pred.sum() else np.nan,
                        merged_recall=float((pred & true).sum() / true.sum()) if true.sum() else np.nan)
    return summ, T


def summary_text(s: dict) -> str:
    if not s.get("n_trajectories"): return "확인된 정답이 아직 없습니다 (status가 '미확인'이 아닌 점이 필요)"
    f = lambda v: "-" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v * 100:.1f}%"
    t = (f"정답 궤적 {s['n_trajectories']}개, 확인된 연속 연결 {s['n_links']}개\n"
         f"  올바른 연결 {f(s['link_accuracy'])} · 다른 입자로 바뀜 {f(s['switch_rate'])} · 끊김 {f(s['break_rate'])}\n"
         f"  가림/누락 구간 {s['gap_total']}개 중 같은 트랙으로 다시 이어짐 {f(s['gap_reidentified'])}\n"
         f"  궤적당 주 트랙이 덮은 비율 평균 {f(s['mean_main_track_frac'])}")
    if "merged_true" in s:
        t += f"\n  합쳐짐 표시: 정답 {s['merged_true']}개, 트래커 표시 {s['merged_flagged']}개 → 정밀도 {f(s['merged_precision'])}, 재현율 {f(s['merged_recall'])}"
    return t


# ------------------------------------------------------------------ 연결 검증 (누락 연결이 맞는지 표본 확인)
CHECK_FILE = "link_checks.csv"
CHECK_COLS = ["check_id", "track_id", "gap", "frame_a", "label_a", "x_a", "y_a", "frame_b", "label_b", "x_b", "y_b", "answer", "updated"]


def sample_links(points: pd.DataFrame, n_per=12, seed=0) -> pd.DataFrame:
    """트래커 결과에서 연결 표본을 뽑습니다: 누락 프레임 수(gap−1)별로 n_per개, 대조로 연속 연결 n_per개. 순서는 섞어 둡니다."""
    t = points.sort_values(["track_id", "frame"]); g = t.groupby("track_id"); nx = g.shift(-1)
    L = pd.DataFrame(dict(track_id=t.track_id.values, gap=(nx.frame - t.frame).values, frame_a=t.frame.values, label_a=t.roi_label.values,
                          x_a=t.x_fl.values, y_a=t.y_fl.values, frame_b=nx.frame.values, label_b=nx.roi_label.values, x_b=nx.x_fl.values, y_b=nx.y_fl.values)).dropna()
    rng = np.random.default_rng(seed); parts = []
    for gap, d in L.groupby(L.gap.clip(upper=4)):
        parts.append(d.iloc[rng.permutation(len(d))[:n_per]])
    S = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    for c in ["gap", "frame_a", "label_a", "frame_b", "label_b", "track_id"]: S[c] = S[c].astype(int)
    S.insert(0, "check_id", np.arange(1, len(S) + 1)); S["answer"] = ""; S["updated"] = ""
    return S[CHECK_COLS]


def load_checks(folder):
    f = Path(folder) / CHECK_FILE
    return pd.read_csv(f, keep_default_na=False) if f.exists() else pd.DataFrame(columns=CHECK_COLS)


def save_checks(folder, C): C.to_csv(Path(folder) / CHECK_FILE, index=False)


def checks_summary(C: pd.DataFrame) -> str:
    if not len(C): return "표본이 없습니다"
    lines = []
    for gap, d in C.groupby("gap"):
        a = d[d.answer.isin(["same", "diff"])]; n = len(a); k = int((a.answer == "same").sum())
        nm = "연속 연결(대조)" if gap == 1 else f"{int(gap) - 1}프레임 누락 연결"
        lines.append(f"  {nm}: 확인 {n}/{len(d)} → 같은 입자 {k}/{n}" + (f" ({k / n * 100:.0f}%)" if n else "") + f", 모름 {int((d.answer == 'unsure').sum())}")
    return "연결 검증 결과\n" + "\n".join(lines)

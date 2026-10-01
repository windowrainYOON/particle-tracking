"""정답 궤적 평가 로직 점검:  python -m pytest tests/test_gt.py"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ptracker.gt import new_from_track, set_point, truncate, evaluate, summary_text


def _points():
    # 트랙 1: 입자 A (label 1), 트랙 2: 입자 B (label 2), 5프레임
    rows = []
    for f in range(1, 6):
        rows += [dict(track_id=1, frame=f, roi_label=1, x_fl=10.0 + f, y_fl=10.0), dict(track_id=2, frame=f, roi_label=2, x_fl=40.0 + f, y_fl=40.0)]
    return pd.DataFrame(rows)


def test_perfect_and_switch():
    P = _points(); gt, gid = new_from_track(pd.DataFrame(), P[P.track_id == 1], "t1")
    s, _ = evaluate(P, gt); assert s["n_trajectories"] == 0           # 미확인(auto)은 평가하지 않음
    for f in range(1, 6): gt = set_point(gt, gid, f, status="ok")
    s, T = evaluate(P, gt); assert s["link_accuracy"] == 1.0 and s["n_links"] == 4
    # 트래커가 frame 3에서 입자 B로 바뀌었다면: 정답은 A인데 트랙 1이 frame 3에 B(label 2)를 가짐
    P2 = P.copy(); P2.loc[(P2.frame >= 3) & (P2.roi_label == 1), "track_id"] = 9; P2.loc[(P2.frame >= 3) & (P2.roi_label == 2), "track_id"] = 1
    s, _ = evaluate(P2, gt); assert s["switch_rate"] == 0.25 and s["link_accuracy"] == 0.75
    # 끊김: frame 3부터 트랙 번호가 달라지고 트랙 1은 거기서 끝
    P3 = P.copy(); P3.loc[(P3.frame >= 3) & (P3.roi_label == 1), "track_id"] = 7
    s, _ = evaluate(P3, gt); assert s["break_rate"] == 0.25
    assert "올바른 연결" in summary_text(s)


def test_gap_and_merged_and_truncate():
    P = _points(); gt, gid = new_from_track(pd.DataFrame(), P[P.track_id == 1], "t1")
    for f in (1, 2, 4, 5): gt = set_point(gt, gid, f, status="ok")
    gt = set_point(gt, gid, 3, status="merged")
    P["merged"] = (P.frame == 3) & (P.roi_label == 1)
    s, _ = evaluate(P, gt); assert s["gap_total"] == 1 and s["gap_reidentified"] == 1.0
    assert s["merged_precision"] == 1.0 and s["merged_recall"] == 1.0
    gt = truncate(gt, gid, 2); assert gt.frame.max() == 2
    gt = set_point(gt, gid, 3, 12.0, 10.0, None, status="ok"); assert gt[gt.frame == 3].roi_label.iloc[0] == -1
    s, _ = evaluate(P, gt); assert s["link_accuracy"] == 1.0           # 직접 찍은 점도 가까운 검출에 연결

"""추천·학습·학습 모델 트래킹 점검 (합성 데이터):  python -m pytest tests/test_learn.py"""
import sys, tempfile, json
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.make_synthetic import make
from ptracker.config import Config
from ptracker.io_utils import auto_group_files
from ptracker.pipeline import DatasetRunner
from ptracker.gt import new_from_track, set_point, save_gt, recommend_tracks, load_gt


def test_recommend_learn_and_track(tmp=None):
    from ptracker.learn import train, report_text
    tmp = Path(tmp or tempfile.mkdtemp()); make(tmp / "data")
    cfg = Config(); cfg.measurement.min_track = 4; cfg.output.make_movies = False; cfg.output.make_figures = False
    spec = auto_group_files(list((tmp / "data").glob("*.tif")), out_root=tmp / "out")[0]
    DatasetRunner(spec, cfg, log=lambda *a: None).run(["detect", "register", "cells", "track", "measure"])
    P = pd.read_csv(Path(spec.out_dir) / "tracks_points_A_B_HT.csv")
    R = recommend_tracks(P, None, n=10, min_len=4, spacing=5); assert len(R) > 0 and "reason" in R
    gt = pd.DataFrame()
    for tid in P.groupby("track_id").size().sort_values(ascending=False).index[:12]:      # 트래커 트랙으로 정답 흉내
        t = P[P.track_id == tid]; gt, gid = new_from_track(gt, t, f"x::{tid}")
        for f in t.frame: gt = set_point(gt, gid, int(f), status="ok")
    save_gt(spec.out_dir, gt)
    R2 = recommend_tracks(P, load_gt(spec.out_dir), n=10, min_len=4, spacing=5)
    assert not set(R2.track_id) & set(P.groupby("track_id").size().sort_values(ascending=False).index[:12])   # 정답 있는 트랙 제외
    model = tmp / "model.json"; rep = train([spec.out_dir], cfg.tracking, model, log=lambda *a: None)
    assert model.exists() and rep["weights_px"]["d"] == 1.0 and rep["scale"] > 0 and "교차검증" in report_text(rep)
    cfg.tracking.learned_model = str(model)
    DatasetRunner(spec, cfg, log=lambda *a: None).run(["track", "measure"])
    assert pd.read_csv(Path(spec.out_dir) / "tracks_points_A_B_HT.csv").track_id.nunique() > 10


if __name__ == "__main__":
    test_recommend_learn_and_track(sys.argv[1] if len(sys.argv) > 1 else None); print("OK")

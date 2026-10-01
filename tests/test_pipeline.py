"""합성 데이터로 전체 파이프라인 스모크 테스트:  python -m pytest tests  또는  python tests/test_pipeline.py"""
import sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.make_synthetic import make
from ptracker.config import Config
from ptracker.io_utils import auto_group_files
from ptracker.pipeline import DatasetRunner, combine_and_plot, run_risefall, make_crop


def test_end_to_end(tmp=None):
    tmp = Path(tmp or tempfile.mkdtemp()); make(tmp / "data")
    cfg = Config(); cfg.measurement.min_track = 4; cfg.risefall.min_points = 4; cfg.output.movie_fps = 2
    specs = auto_group_files(list((tmp / "data").glob("*.tif")), out_root=tmp / "out"); assert len(specs) == 1
    DatasetRunner(specs[0], cfg).run()
    out = Path(specs[0].out_dir)
    for f in ["tracks_points_A_B_HT.csv", "tracks_summary_A_B_HT.csv", "ratio_inside_vs_outside.png", "registration.csv"]:
        assert (out / f).exists(), f
    DatasetRunner(specs[0], cfg).run(["measure", "figures"])        # 캐시에서 재실행
    pts = combine_and_plot({specs[0].name: str(out)}, tmp / "combined", cfg)
    R, F = run_risefall(pts, cfg, tmp / "rf")
    uid = pts[pts.group.isin(["inside", "outside"])].track_uid.iloc[0]
    mv = make_crop(uid, pts, R, {specs[0].name: str(out)}, cfg, tmp / "crops"); assert Path(mv).exists()
    print("OK", tmp)


if __name__ == "__main__":
    test_end_to_end(sys.argv[1] if len(sys.argv) > 1 else None)


def test_tracking_options(tmp=None):
    """주변 배치 비교·여러 프레임 누락 연결·외형 특징을 모두 켜고 트래킹이 끝까지 도는지."""
    tmp = Path(tmp or tempfile.mkdtemp()); make(tmp / "data")
    cfg = Config(); cfg.measurement.min_track = 4; cfg.output.make_movies = False; cfg.output.make_figures = False
    cfg.tracking.constellation = True; cfg.tracking.gap_max = 3; cfg.tracking.appearance = True
    spec = auto_group_files(list((tmp / "data").glob("*.tif")), out_root=tmp / "out")[0]
    DatasetRunner(spec, cfg, log=lambda *a: None).run(["detect", "register", "cells", "track", "measure"])
    import pandas as pd
    pts = pd.read_csv(Path(spec.out_dir) / "tracks_points_A_B_HT.csv"); assert pts.track_id.nunique() > 10 and "merged" in pts

"""신뢰도·경로 교정 점검:  python -m pytest tests/test_reliability_edits.py"""
import os, sys, tempfile, json, time
from pathlib import Path
import numpy as np, pandas as pd
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ptracker import reliability


def test_window_reliability():
    perr = reliability._rates(reliability.PRIOR)
    t = pd.DataFrame(dict(frame=[1, 2, 3, 5], link_margin=[20.0, 1.0, 3.0, np.nan], merged=[False, False, True, False]))
    w = reliability.window_reliability(t, 1, 5, perr)
    exp = (1 - perr["≥10"]) * (1 - perr["<2"]) * (1 - 0.5) * (1 - 0.3)        # 3→5는 누락 연결, frame 3은 합쳐짐
    assert abs(w["reliability"] - exp) < 1e-9 and w["risky_links"] == 1 and w["gap_links"] == 1 and w["merged_frames"] == 1
    t["verified"] = [True, True, False, False]                                 # 확인한 두 점 사이 연결은 확실
    # 1→2는 둘 다 확인 → 확실, 2→3은 마진 1.0(<2), frame 3 합쳐짐
    assert abs(reliability.window_reliability(t, 1, 3, perr)["reliability"] - (1 - perr["<2"]) * 0.7) < 1e-9
    p, why = reliability.calibrate([]); assert "기본" in why and p["<2"] > p["≥10"]


def test_pipeline_edits_and_editor(tmp=None):
    from tests.make_synthetic import make
    from ptracker.config import Config
    from ptracker.io_utils import auto_group_files
    from ptracker.pipeline import DatasetRunner
    from ptracker.edits import save_track, apply_edits, measure_free
    from ptracker.analysis import combine, risefall
    tmp = Path(tmp or tempfile.mkdtemp()); make(tmp / "data")
    cfg = Config(); cfg.measurement.min_track = 4; cfg.risefall.min_points = 4; cfg.output.make_movies = False; cfg.output.make_figures = False
    spec = auto_group_files(list((tmp / "data").glob("*.tif")), out_root=tmp / "out")[0]
    DatasetRunner(spec, cfg, log=lambda *a: None).run(["detect", "register", "cells", "track", "measure"])
    P = pd.read_csv(Path(spec.out_dir) / "tracks_points_A_B_HT.csv"); assert "link_margin" in P and P.link_margin.notna().mean() > .5
    R, F = risefall(combine({"syn": spec.out_dir}), cfg.risefall); assert "reliability" in R and "kept_reliable" in F
    # 교정: 가장 긴 트랙의 두 번째 프레임을 다른 검출로, 세 번째는 빈 곳 측정
    tid = P.groupby("track_id").size().idxmax(); t = P[P.track_id == tid].sort_values("frame")
    f2 = int(t.frame.iloc[1]); other = P[(P.frame == f2) & (P.track_id != tid)].iloc[0]
    from ptracker.gui.track_views import DatasetContext
    from ptracker.preview import PreviewSource
    ctx = DatasetContext(spec.out_dir); src = PreviewSource(ctx.spec)
    m = measure_free(ctx, src, int(t.frame.iloc[2]), float(t.x_fl.iloc[2]) + 3, float(t.y_fl.iloc[2]), 3.0, cfg.measurement)
    assert np.isfinite(m["I_A_raw"]) and m["area_px"] > 0
    path = pd.DataFrame([dict(frame=int(t.frame.iloc[0]), roi_label=int(t.roi_label.iloc[0]), x_fl=t.x_fl.iloc[0], y_fl=t.y_fl.iloc[0], status="ok"),
                         dict(frame=f2, roi_label=int(other.roi_label), x_fl=other.x_fl, y_fl=other.y_fl, status="edit"),
                         dict(frame=int(t.frame.iloc[2]), roi_label=-1, x_fl=float(t.x_fl.iloc[2]) + 3, y_fl=float(t.y_fl.iloc[2]), status="edit", **m)])
    save_track(spec.out_dir, int(tid), path)
    A = apply_edits(P, spec.out_dir); e = A[A.track_id == tid].sort_values("frame")
    assert len(e) == 3 and e.edited.all() and e.verified.tolist() == [True, True, True]
    assert int(e.roi_label.iloc[1]) == int(other.roi_label) and abs(e.I_A.iloc[1] - other.I_A) < 1e-9
    C = combine({"syn": spec.out_dir}); assert (C[C.track_id == tid].frame.nunique() == 3)
    # 편집기 위젯이 화면 없이 동작
    from PySide6.QtWidgets import QApplication
    from ptracker.gui.track_views import TrackEditor
    app = QApplication.instance() or QApplication([]); ed = TrackEditor(); ed.resize(1000, 800); ed.show()
    uid = f"syn::{tid}"; ed.set_track(ctx, C, uid, None, cfg); app.processEvents()
    f = ed._f; d = C[(C.frame == f) & (C.track_uid != uid)].iloc[0]; ed._on_click(float(d.x_fl), float(d.y_fl), d); app.processEvents()
    assert ed._path[f]["status"] == "edit" and "교정 후" in ed.gfig.axes[0].get_title()
    got = []; ed.saved.connect(lambda u, n: got.append(n)); ed._save(); app.processEvents()
    assert got and len(got[0]) > 0 and (Path(spec.out_dir) / "ground_truth.csv").exists()


if __name__ == "__main__":
    test_window_reliability(); test_pipeline_edits_and_editor(sys.argv[1] if len(sys.argv) > 1 else None); print("OK")

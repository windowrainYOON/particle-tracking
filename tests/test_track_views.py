"""트랙 탐색·크롭 위젯이 화면 없이(offscreen) 동작하는지 확인:  python -m pytest tests/test_track_views.py"""
import os, sys, tempfile, time
from pathlib import Path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.make_synthetic import make
from ptracker.config import Config
from ptracker.io_utils import auto_group_files
from ptracker.pipeline import DatasetRunner


def _wait(app, cond, timeout=60):
    t = time.time()
    while not cond() and time.time() - t < timeout: app.processEvents(); time.sleep(.05)
    return cond()


def test_track_explorer(tmp=None):
    from PySide6.QtWidgets import QApplication
    from ptracker.gui.track_views import TrackExplorer
    tmp = Path(tmp or tempfile.mkdtemp()); make(tmp / "data")
    cfg = Config(); cfg.measurement.min_track = 4; cfg.output.make_movies = False; cfg.output.make_figures = False
    spec = auto_group_files(list((tmp / "data").glob("*.tif")), out_root=tmp / "out")[0]
    DatasetRunner(spec, cfg, log=lambda *a: None).run()
    app = QApplication.instance() or QApplication([])
    ex = TrackExplorer(lambda: cfg); ex.resize(1200, 900); ex.show(); ex.set_folder(spec.out_dir); app.processEvents()
    ex.minlen.setValue(4); assert ex.model.rowCount() > 0
    uid = ex.model.df().track_uid.iloc[0]; ex.map.trackClicked.emit(uid); app.processEvents()
    assert ex.table.selectionModel().selectedRows()[0].row() == 0
    ex.bottom.setCurrentIndex(1)
    assert _wait(app, lambda: ex.crop._data is not None), ex.crop.lbl.text()
    assert ex.crop._data["A"].shape[1:] == (2 * cfg.output.crop_half,) * 2
    ex.crop.grid.setChecked(True); app.processEvents(); ex.shutdown()


if __name__ == "__main__":
    test_track_explorer(sys.argv[1] if len(sys.argv) > 1 else None); print("OK")

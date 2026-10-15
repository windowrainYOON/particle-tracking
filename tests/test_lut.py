"""LUT 프리셋: 채널 매칭, 영상에 적용, 출력 폴더에 저장, 측정은 그대로."""
import sys, json, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from tests.make_synthetic import make
from ptracker.config import Config
from ptracker.io_utils import auto_group_files
from ptracker.pipeline import DatasetRunner
from ptracker import lut


def _preset(path, channels):
    path.write_text(json.dumps(dict(channels={n: dict(vmin=lo, vmax=hi, color=c, gamma=1.0, visible=True) for n, lo, hi, c in channels},
                                    order=[c[0] for c in channels])))
    return path


def test_match_names(tmp_path):
    p = _preset(tmp_path / "p.json", [("CH0 ex561/em600", 0, 100, [1, 0, 0]), ("CH1 ex640/em700", 0, 50, [0, 1, 0]), ("RI", 1.33, 1.40, [1, 1, 1])])
    pr = lut.load_preset(p)
    assert lut.match_channel(pr, "/x/a.TCF::CH1", "reference") == "CH1 ex640/em700"
    assert lut.match_channel(pr, "/x/a.TCF::CH0", "measure") == "CH0 ex561/em600"
    assert lut.match_channel(pr, "/x/a.TCF::RI", "ri") == "RI"
    assert lut.match_channel(pr, "/x/a.TCF::CH2", "measure") is None
    pr2 = lut.load_preset(_preset(tmp_path / "q.json", [("AF647-T1", 0, 9, [1, 0, 1]), ("Cy5", 0, 9, [1, 0, 1]), ("pHrodo", 0, 9, [1, 0, 0])]))
    assert lut.match_channel(pr2, "/x/Cy5_1.tif", "reference") == "Cy5"
    assert lut.match_channel(pr2, "/x/pHrodo_1.tif", "measure") == "pHrodo"
    L = lut.role_luts(p, {"ri": "/x/a.TCF::RI"}, {"ri": np.full((3, 4, 4), 13400, np.uint16)}, log=lambda *a: None)
    assert abs(L["ri"].vmin - 13300) < 1e-6 and abs(L["ri"].vmax - 14000) < 1e-6      # 굴절률 → × 10⁴
    rgb = L["ri"].rgb(np.array([[13300, 14000]]))
    assert rgb.shape == (1, 2, 3) and rgb[0, 0].max() == 0 and np.allclose(rgb[0, 1], 1)


def test_movies_with_preset(tmp=None):
    tmp = Path(tmp or tempfile.mkdtemp()); make(tmp / "data")
    cfg = Config(); cfg.measurement.min_track = 4; cfg.output.movie_fps = 2; cfg.output.make_figures = False
    spec = auto_group_files(list((tmp / "data").glob("*.tif")), out_root=tmp / "out")[0]
    DatasetRunner(spec, cfg).run(); out = Path(spec.out_dir)
    before = pd.read_csv(out / "tracks_points_A_B_HT.csv")
    cfg.output.lut_preset = str(_preset(tmp / "my_lut.json", [("Cy5", 0, 300, [1, 0, 1]), ("pHrodo", 0, 200, [1, .3, 0]), ("HT", 1.33, 1.38, [1, 1, 1])]))
    DatasetRunner(spec, cfg).run(["measure", "movies"])
    assert (out / "my_lut.json").read_text() == Path(cfg.output.lut_preset).read_text()
    roles = json.loads((out / "lut_roles.json").read_text())["roles"]
    assert roles["reference"]["name"] == "Cy5" and roles["measure"]["name"] == "pHrodo" and roles["ri"]["name"] == "HT"
    pd.testing.assert_frame_equal(before, pd.read_csv(out / "tracks_points_A_B_HT.csv"))     # 측정은 raw 그대로
    assert list(out.glob("movie_reference_inside.*"))


if __name__ == "__main__":
    test_movies_with_preset(sys.argv[1] if len(sys.argv) > 1 else None)

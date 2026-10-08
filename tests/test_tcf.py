"""raw TCF(HDF5) 입력: 채널 목록, 역할 할당, tif와 같은 분석 결과, 옛 설정 호환:  python -m pytest tests/test_tcf.py"""
import sys, tempfile, json
from pathlib import Path
import numpy as np, pandas as pd, tifffile, h5py
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.make_synthetic import make
from ptracker.config import Config
from ptracker.io_utils import (auto_group_files, tcf_channels, tcf_specs, load_dataset, DatasetSpec, source_shape, read_frame,
                               split_source, source_label)
from ptracker.pipeline import DatasetRunner


def write_tcf(path, A, B, H, dt=120.0):
    """Tomocube TCF와 같은 구조: Data/2DFLMIP/CHn, Data/3DFL/CHn, Data/2DMIP, Data/3D. 형광은 프레임마다 uint8/uint16 섞임."""
    with h5py.File(path, "w") as f:
        D = f.create_group("Data")
        for g in ("2DFLMIP", "3DFL"):
            G = D.create_group(g)
            for ch, st, em in (("CH0", B, 0.515), ("CH1", A, 0.698)):
                C = G.create_group(ch); C.attrs["Emission"] = [em]; C.attrs["Excitation"] = [em - 0.05]
                for t, im in enumerate(st):
                    a = im.astype(np.uint8) if im.max() < 256 else im.astype(np.uint16)
                    d = C.create_dataset(f"{t:06d}", data=a if g == "2DFLMIP" else a[None]); d.attrs["Time"] = [t * dt + 6]
        M = D.create_group("2DMIP"); V = D.create_group("3D")
        for t, im in enumerate(H):
            M.create_dataset(f"{t:06d}", data=im).attrs["Time"] = [t * dt]
            V.create_dataset(f"{t:06d}", data=np.stack([im - 5, im, im - 9])).attrs["Time"] = [t * dt]


def _data(tmp):
    make(tmp / "data"); d = tmp / "data"
    A, B, H = (tifffile.imread(d / n) for n in ("Cy5_syn.tif", "pHrodo_syn.tif", "HT_syn.tif"))
    write_tcf(tmp / "sample.TCF", A, B, H); return A, B, H


def test_tcf_read_and_assign(tmp=None):
    tmp = Path(tmp or tempfile.mkdtemp()); A, B, H = _data(tmp)
    ch = tcf_channels(tmp / "sample.TCF")
    assert [c["name"] for c in ch] == ["CH0", "CH1", "RI"] and ch[0]["frames"] == len(A) and abs(ch[2]["interval_min"] - 2) < 1e-9
    specs, skipped = tcf_specs([tmp / "sample.TCF"], dict(measure="CH0", reference="CH1", ri="RI"), tmp / "out")
    assert len(specs) == 1 and not skipped and specs[0].validate() == []
    s = specs[0]; assert split_source(s.measure) == (str(tmp / "sample.TCF"), "CH0") and source_label(s.ri) == "sample.TCF › RI"
    A2, B2, H2 = load_dataset(s)
    assert np.array_equal(A2, A) and np.array_equal(B2, B) and np.array_equal(H2, H)
    assert source_shape(s.ri, s.reference) == H.shape and np.array_equal(read_frame(s.measure, 3, s.reference), B[3])
    # 할당을 바꾸면 그대로 따라감, 없는 채널은 건너뜀
    sw, _ = tcf_specs([tmp / "sample.TCF"], dict(measure="CH1", reference="CH0", ri="RI"))
    assert np.array_equal(load_dataset(sw[0])[0], B)
    none, skipped = tcf_specs([tmp / "sample.TCF"], dict(measure="CH2", reference="CH1", ri="RI")); assert not none and "CH2" in skipped[0]
    # 3D만 있는 RI는 z 최대값
    with h5py.File(tmp / "sample.TCF", "a") as f: del f["Data/2DMIP"]
    from ptracker import io_utils; io_utils._tcf_index.cache_clear()
    assert np.array_equal(load_dataset(s)[2], H)


def test_tcf_matches_tif_pipeline(tmp=None):
    tmp = Path(tmp or tempfile.mkdtemp()); _data(tmp)
    cfg = Config(); cfg.measurement.min_track = 4; cfg.output.make_movies = False; cfg.output.make_figures = False
    tif = auto_group_files(list((tmp / "data").glob("*.tif")), out_root=tmp / "out_tif")[0]
    tcf = tcf_specs([tmp / "sample.TCF"], dict(measure="CH0", reference="CH1", ri="RI"), tmp / "out_tcf")[0][0]
    for s in (tif, tcf): DatasetRunner(s, cfg, log=lambda *a: None).run(["detect", "register", "cells", "track", "measure"])
    a, b = (pd.read_csv(Path(s.out_dir) / "tracks_points_A_B_HT.csv") for s in (tif, tcf))
    pd.testing.assert_frame_equal(a, b)
    assert DatasetSpec.load(Path(tcf.out_dir) / "dataset.json") == tcf


def test_old_keys_compatible(tmp=None):
    tmp = Path(tmp or tempfile.mkdtemp())
    (tmp / "dataset.json").write_text(json.dumps(dict(name="x", cy5="a.tif", phrodo="b.tif", ht="c.tif", out_dir="o")), encoding="utf-8")
    s = DatasetSpec.load(tmp / "dataset.json"); assert (s.reference, s.measure, s.ri) == ("a.tif", "b.tif", "c.tif")
    cfg = Config.from_dict({"channel": {"cy5_keyword": "X5", "phrodo_keyword": "pH", "ht_keyword": "H"}})
    assert (cfg.channel.reference_keyword, cfg.channel.measure_keyword, cfg.channel.ri_keyword) == ("X5", "pH", "H")


def test_assign_dialog(tmp=None):
    import os; os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from ptracker.gui.channel_dialog import ChannelAssignDialog, default_assign
    app = QApplication.instance() or QApplication([])
    tmp = Path(tmp or tempfile.mkdtemp()); _data(tmp)
    assert default_assign(["CH0", "CH1", "RI"]) == dict(measure="CH0", reference="CH1", ri="RI")
    assert default_assign(["CH0", "CH1", "CH2", "RI"], dict(measure="CH2", reference="CH0", ri="RI")) == dict(measure="CH2", reference="CH0", ri="RI")
    dlg = ChannelAssignDialog([tmp / "sample.TCF"], None, 20.0)
    assert dlg.assignment() == dict(measure="CH0", reference="CH1", ri="RI") and dlg.ok.isEnabled()
    assert abs(dlg.frame_interval() - 2.0) < 1e-9                       # 파라미터(20분)와 다르면 TCF 간격 제안
    dlg.combo["reference"].setCurrentText("CH0"); assert not dlg.ok.isEnabled()   # 같은 채널은 확인 불가


if __name__ == "__main__":
    test_tcf_read_and_assign(); test_tcf_matches_tif_pipeline(); test_old_keys_compatible(); test_assign_dialog(); print("OK")

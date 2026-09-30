"""파라미터 미리보기가 모든 섹션에서 동작하고, 전체 실행과 같은 결과를 내는지 확인:  python -m pytest tests/test_preview.py"""
import sys, tempfile, pickle
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.make_synthetic import make
from ptracker.config import Config
from ptracker.io_utils import auto_group_files
from ptracker.pipeline import DatasetRunner
from ptracker.preview import PreviewSource, run_preview, check_files, DataUnavailable


def test_preview_sections(tmp=None):
    tmp = Path(tmp or tempfile.mkdtemp()); make(tmp / "data")
    spec = auto_group_files(list((tmp / "data").glob("*.tif")), out_root=tmp / "out")[0]; cfg = Config()
    src = PreviewSource(spec)
    msg = run_preview(src, "mitosis", 0, None, 128, cfg)                     # 캐시 없으면 안내 메시지
    assert msg["kind"] == "message"
    DatasetRunner(spec, cfg, log=lambda *a: None).run(["detect", "register", "cells"])
    src = PreviewSource(spec)
    for sec in ["detection", "registration", "cells", "footprint", "mitosis", "measurement", "tracking", "channel"]:
        r = run_preview(src, sec, 2, None, 128, cfg)
        assert r["kind"] in ("image", "plot", "message") and "overview" in r, sec
    # 전체 실행과 같은 결과: 프레임 3 입자 수, 세포 threshold
    P = pickle.load(open(Path(spec.out_dir) / "_cache" / "P.pkl", "rb"))
    rd = run_preview(src, "detection", 2, None, 128, cfg)
    assert f"입자 {int((P.frame == 3).sum())}개" in rd["info"]
    thr = pickle.load(open(Path(spec.out_dir) / "_cache" / "cell_thr.pkl", "rb"))
    r = run_preview(src, "cells", 2, None, 128, cfg)
    assert f"{thr[0]:.1f} / 세포 중심 {thr[1]:.1f}" in r["info"]
    # 파라미터를 바꾸면 결과가 달라짐
    cfg2 = Config(); cfg2.detection.k_mad = 40
    assert run_preview(src, "detection", 2, None, 128, cfg2)["info"] != rd["info"]


def test_missing_file():
    from ptracker.io_utils import DatasetSpec
    try:
        check_files(DatasetSpec("x", "/nonexistent/a.tif", "/nonexistent/b.tif", "/nonexistent/c.tif", "/tmp/x")); assert False
    except DataUnavailable:
        pass


if __name__ == "__main__":
    test_preview_sections(sys.argv[1] if len(sys.argv) > 1 else None); test_missing_file(); print("OK")

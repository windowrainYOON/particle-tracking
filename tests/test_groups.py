"""그룹 비교: 합성 데이터 결과 두 개를 그룹으로 묶어 입자·세포 단위 계산과 저장이 되는지."""
import sys, tempfile, shutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from tests.make_synthetic import make
from ptracker.config import Config
from ptracker.io_utils import auto_group_files
from ptracker.pipeline import DatasetRunner, combine_and_plot
from ptracker import groups as G


def test_group_compare(tmp=None):
    tmp = Path(tmp or tempfile.mkdtemp()); make(tmp / "data")
    cfg = Config(); cfg.measurement.min_track = 4; cfg.output.make_movies = False
    spec = auto_group_files(list((tmp / "data").glob("*.tif")), out_root=tmp / "out")[0]
    DatasetRunner(spec, cfg, log=lambda *a: None).run(["detect", "register", "cells", "track", "measure", "figures"])
    out = Path(spec.out_dir); C = pd.read_csv(out / "cell_intensity.csv")
    assert C.cell_id.nunique() >= 1 and C.cell_I_A.notna().all() and (out / "cell_intensity_timecourse.png").exists()
    shutil.copytree(out, tmp / "B" / "img2")
    combine_and_plot({"img1": str(out)}, tmp / "comb", cfg); assert (tmp / "comb" / G.COMB_CELLS).exists()
    gs = [G.Group("A", [str(tmp / "comb")]), G.Group("B", [str(tmp / "B")])]
    P, Cc = G.load_groups(gs); assert set(P.group) == {"A", "B"} and set(Cc.group) == {"A", "B"} and "track_group" in P
    for kw in [dict(location="both", min_points=3), dict(location="split", min_points=3, bin_h=.5, agg="image"),
               dict(metric="cp_sum_ratio", min_points=2, norm="fold", min_n=1), dict(metric="cell_ratio_BA", min_points=2, spread="sem", min_n=1)]:
        s = G.CompareSettings(**kw); U = G.unit_values(P, Cc, s); S = G.summarize(U, s); M = G.unit_means(U, s)
        assert len(U) and set(U.group) == {"A", "B"} and S.center.notna().any(), kw
        T = G.stats_table(M); assert len(T) or M.groupby("series").size().min() < 2   # 검정은 그룹마다 2개 이상일 때
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(); G.plot_timecourse(ax, U, S, s, G.series_colors(gs), "unit")
    files = G.export(tmp / "exp" / "cmp.pdf", fig, U, S, M, T, gs, s); assert all(Path(f).exists() for f in files)
    gs2, s2, _ = G.load_setup(files[-1]); assert [g.name for g in gs2] == ["A", "B"] and s2.metric == "cell_ratio_BA"


if __name__ == "__main__":
    test_group_compare(sys.argv[1] if len(sys.argv) > 1 else None)

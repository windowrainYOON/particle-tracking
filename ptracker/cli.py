"""명령줄 실행 (GUI 없이 배치 처리할 때).

예)
  python -m ptracker.cli run --cy5 Cy5_1.tif --phrodo pHrodo_1.tif --ht HT_1.tif --out results/D3
  python -m ptracker.cli run-folder /data/folder --out results      (파일명 키워드로 자동 묶음)
  python -m ptracker.cli combine results/D1 results/D2 --out results/combined
  python -m ptracker.cli risefall results/combined/combined_tracks_points_A_B_HT.csv --out results/risefall
옵션 --config my.yaml 로 파라미터 지정, --stages track measure figures 로 일부 단계만 실행.
"""
import argparse, glob
from pathlib import Path
import pandas as pd
from .config import Config
from .io_utils import DatasetSpec, auto_group_files
from .pipeline import DatasetRunner, STAGES, combine_and_plot, run_risefall


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ptracker")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--cy5", required=True); r.add_argument("--phrodo", required=True); r.add_argument("--ht", required=True)
    r.add_argument("--out", required=True); r.add_argument("--name", default=None)
    rf = sub.add_parser("run-folder"); rf.add_argument("folder"); rf.add_argument("--out", default=None)
    for p_ in (r, rf):
        p_.add_argument("--config"); p_.add_argument("--stages", nargs="*", choices=STAGES)
    cb = sub.add_parser("combine"); cb.add_argument("dirs", nargs="+"); cb.add_argument("--out", required=True); cb.add_argument("--config")
    rs = sub.add_parser("risefall"); rs.add_argument("points_csv"); rs.add_argument("--out", required=True); rs.add_argument("--config")
    a = ap.parse_args(argv)
    cfg = Config.load(a.config) if getattr(a, "config", None) else Config()
    if a.cmd == "run":
        spec = DatasetSpec(a.name or Path(a.out).name, a.cy5, a.phrodo, a.ht, a.out); DatasetRunner(spec, cfg).run(a.stages)
    elif a.cmd == "run-folder":
        files = glob.glob(str(Path(a.folder) / "*.tif")) + glob.glob(str(Path(a.folder) / "*.tiff"))
        ch = cfg.channel; specs = auto_group_files(files, ch.cy5_keyword, ch.phrodo_keyword, ch.ht_keyword, a.out)
        print("데이터셋:", [s.name for s in specs])
        for s in specs: DatasetRunner(s, cfg).run(a.stages)
    elif a.cmd == "combine":
        combine_and_plot({Path(d).name: d for d in a.dirs}, a.out, cfg)
    elif a.cmd == "risefall":
        run_risefall(pd.read_csv(a.points_csv), cfg, a.out)


if __name__ == "__main__":
    main()

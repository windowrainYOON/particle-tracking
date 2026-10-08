"""명령줄 실행 (GUI 없이 배치 처리할 때).

예)
  python -m ptracker.cli run --measure pHrodo_1.tif --reference Cy5_1.tif --ri HT_1.tif --out results/D3
  python -m ptracker.cli run --measure a.TCF::CH0 --reference a.TCF::CH1 --ri a.TCF::RI --out results/a   (raw TCF 채널)
  python -m ptracker.cli run-folder /data/folder --out results      (tif: 파일명 키워드로 자동 묶음, TCF: --assign 채널 할당)
  python -m ptracker.cli run-folder /data/tcf --assign measure=CH0 reference=CH1 ri=RI --out results
  python -m ptracker.cli combine results/D1 results/D2 --out results/combined
  python -m ptracker.cli risefall results/combined/combined_tracks_points_A_B_HT.csv --out results/risefall
옵션 --config my.yaml 로 파라미터 지정, --stages track measure figures 로 일부 단계만 실행.
"""
import argparse, glob
from pathlib import Path
import pandas as pd
from .config import Config
from .io_utils import DatasetSpec, auto_group_files, tcf_specs
from .pipeline import DatasetRunner, STAGES, combine_and_plot, run_risefall


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ptracker")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--measure", "--phrodo", required=True); r.add_argument("--reference", "--cy5", required=True); r.add_argument("--ri", "--ht", required=True)
    r.add_argument("--out", required=True); r.add_argument("--name", default=None)
    rf = sub.add_parser("run-folder"); rf.add_argument("folder"); rf.add_argument("--out", default=None)
    rf.add_argument("--assign", nargs=3, default=["measure=CH0", "reference=CH1", "ri=RI"], help="TCF 채널 할당 (역할=채널)")
    for p_ in (r, rf):
        p_.add_argument("--config"); p_.add_argument("--stages", nargs="*", choices=STAGES)
    cb = sub.add_parser("combine"); cb.add_argument("dirs", nargs="+"); cb.add_argument("--out", required=True); cb.add_argument("--config")
    rs = sub.add_parser("risefall"); rs.add_argument("points_csv"); rs.add_argument("--out", required=True); rs.add_argument("--config")
    a = ap.parse_args(argv)
    cfg = Config.load(a.config) if getattr(a, "config", None) else Config()
    if a.cmd == "run":
        spec = DatasetSpec(a.name or Path(a.out).name, reference=a.reference, measure=a.measure, ri=a.ri, out_dir=a.out); DatasetRunner(spec, cfg).run(a.stages)
    elif a.cmd == "run-folder":
        files = glob.glob(str(Path(a.folder) / "*.tif")) + glob.glob(str(Path(a.folder) / "*.tiff"))
        ch = cfg.channel; specs = auto_group_files(files, ch.reference_keyword, ch.measure_keyword, ch.ri_keyword, a.out)
        tcfs = [f for f in Path(a.folder).iterdir() if f.suffix.lower() == ".tcf"]
        if tcfs:
            more, skipped = tcf_specs(tcfs, dict(x.split("=", 1) for x in a.assign), a.out); specs += more
            for m in skipped: print("건너뜀:", m)
        print("데이터셋:", [s.name for s in specs])
        for s in specs: DatasetRunner(s, cfg).run(a.stages)
    elif a.cmd == "combine":
        combine_and_plot({Path(d).name: d for d in a.dirs}, a.out, cfg)
    elif a.cmd == "risefall":
        run_risefall(pd.read_csv(a.points_csv), cfg, a.out)


if __name__ == "__main__":
    main()

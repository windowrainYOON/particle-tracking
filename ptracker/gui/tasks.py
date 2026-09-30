"""GUI에서 백그라운드로 실행하는 작업들. 모두 worker 인자를 받아 log/progress를 전달합니다."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from ..pipeline import DatasetRunner, STAGES, STAGE_WEIGHT, combine_and_plot, run_risefall, make_crop
from .. import analysis


def _logger(worker):
    """print처럼 여러 인자를 받는 log 함수. Qt 신호 log(str)는 인자를 하나만 받습니다."""
    return lambda *a: worker.log.emit(" ".join(str(x) for x in a))


def run_datasets(specs, cfg, stages, worker):
    stages = [s for s in STAGES if s in stages]; tot = sum(STAGE_WEIGHT[s] for s in stages) or 1; n = len(specs)
    for i, spec in enumerate(specs):
        done = {"w": 0}

        def prog(stage, frac, msg, i=i):
            before = sum(STAGE_WEIGHT[s] for s in stages[:stages.index(stage)]) if stage in stages else 0
            overall = (i + (before + STAGE_WEIGHT.get(stage, 1) * frac) / tot) / n
            worker.progress.emit(int(overall * 100), f"[{spec.name}] {msg}")
        errs = spec.validate()
        if errs: worker.log.emit("; ".join(errs)); continue
        DatasetRunner(spec, cfg, log=_logger(worker), progress=prog, cancel=worker.cancel_event).run(stages)
    worker.progress.emit(100, "완료")


def combine(dirs: dict, out_dir, cfg, worker):
    worker.progress.emit(10, "통합 중")
    combine_and_plot(dirs, out_dir, cfg, log=_logger(worker))
    worker.progress.emit(100, "통합 완료")


def load_points(source):
    """source: 통합 CSV 경로 또는 단일 데이터셋 결과 폴더. 반환 (points, {dataset: folder})."""
    p = Path(source)
    if p.is_dir() and (p / "tracks_points_A_B_HT.csv").exists():
        dirs = {p.name: str(p)}; return analysis.combine(dirs), dirs
    if p.is_dir() and (p / "combined_tracks_points_A_B_HT.csv").exists(): p = p / "combined_tracks_points_A_B_HT.csv"
    pts = pd.read_csv(p); m = p.parent / "datasets_map.json"
    dirs = json.loads(m.read_text(encoding="utf-8")) if m.exists() else {}
    return pts, dirs


def risefall(source, cfg, out_dir, result: dict, worker):
    worker.progress.emit(5, "데이터 읽는 중"); pts, dirs = load_points(source)
    worker.progress.emit(30, "분류 중"); R, F = run_risefall(pts, cfg, out_dir, log=_logger(worker))
    result.update(points=pts, dirs=dirs, R=R, F=F, out_dir=str(out_dir)); worker.progress.emit(100, "분류 완료")


def crops(uids, pts, R, dirs, cfg, out_dir, worker):
    for i, u in enumerate(uids):
        if worker.cancel_event.is_set(): break
        worker.progress.emit(int(i / max(len(uids), 1) * 100), f"크롭 {i+1}/{len(uids)}: {u}")
        try:
            mv = make_crop(u, pts, R, dirs, cfg, out_dir); worker.log.emit(f"크롭 저장: {mv}")
        except Exception as e:   # noqa: BLE001
            worker.log.emit(f"크롭 실패 {u}: {e}")
    worker.progress.emit(100, "크롭 완료")

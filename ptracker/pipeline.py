"""분석 파이프라인 오케스트레이션.

한 데이터셋 = Cy5 + pHrodo + HT 3채널. 단계(stage)마다 결과를 `<출력>/_cache/`에 저장해
파라미터를 바꾼 뒤 특정 단계부터 다시 실행할 수 있습니다.

    detect → register → cells → track → measure → figures → movies
"""
from __future__ import annotations
import json, time, pickle, threading
from pathlib import Path
import numpy as np, pandas as pd, tifffile
from .config import Config
from .io_utils import DatasetSpec, load_dataset
from . import detection, registration, cells, tracking, measurement, analysis, plots, movies

STAGES = ["detect", "register", "cells", "track", "measure", "figures", "movies"]
STAGE_LABELS = {"detect": "입자 검출", "register": "정합", "cells": "세포·영역·분열", "track": "트래킹",
                "measure": "측정·분류", "figures": "그래프", "movies": "영상"}
STAGE_WEIGHT = {"detect": 5, "register": 12, "cells": 12, "track": 20, "measure": 6, "figures": 3, "movies": 6}


class Cancelled(Exception):
    pass


class DatasetRunner:
    def __init__(self, spec: DatasetSpec, cfg: Config, log=print, progress=None, cancel: threading.Event | None = None):
        self.spec, self.cfg = spec, cfg
        self.out = Path(spec.out_dir); self.cache = self.out / "_cache"
        self._log, self._progress, self._cancel = log, progress, cancel or threading.Event()
        self.state: dict = {}

    # ---------------- helpers
    def log(self, *a):
        self._log(f"[{self.spec.name}] " + " ".join(str(x) for x in a))

    def _check(self):
        if self._cancel.is_set(): raise Cancelled()

    def _prog(self, stage, frac, msg=""):
        self._check()
        if self._progress: self._progress(stage, frac, msg)

    def _save(self, key, obj):
        self.cache.mkdir(parents=True, exist_ok=True)
        if isinstance(obj, np.ndarray): np.save(self.cache / f"{key}.npy", obj)
        else:
            with open(self.cache / f"{key}.pkl", "wb") as f: pickle.dump(obj, f)

    def _load(self, key):
        if (self.cache / f"{key}.npy").exists(): return np.load(self.cache / f"{key}.npy")
        if (self.cache / f"{key}.pkl").exists():
            with open(self.cache / f"{key}.pkl", "rb") as f: return pickle.load(f)
        raise FileNotFoundError(f"캐시 없음: {key} — 앞 단계를 먼저 실행하세요")

    def _get(self, key):
        if key not in self.state: self.state[key] = self._load(key)
        return self.state[key]

    def cached_stages(self):
        need = {"detect": "P", "register": "reg", "cells": "CS", "track": "tracks_raw", "measure": "tracks"}
        return [s for s, k in need.items() if (self.cache / f"{k}.pkl").exists() or (self.cache / f"{k}.npy").exists()]

    # ---------------- run
    def run(self, stages=None):
        stages = [s for s in STAGES if s in (stages or STAGES)]
        self.out.mkdir(parents=True, exist_ok=True); self.spec.save(); self.cfg.save(self.out / "config_used.yaml")
        t0 = time.time(); self.log("데이터 로드"); A, B, HT = load_dataset(self.spec)
        self.state.update(A=A, B=B, HT=HT, S=HT.shape[1] / A.shape[1])
        self.log(f"Cy5 {A.shape}, pHrodo {B.shape}, HT {HT.shape}, 배율 {self.state['S']:.4f}")
        for st in stages:
            self._check(); self.log(f"▶ {STAGE_LABELS[st]} 시작")
            getattr(self, f"stage_{st}")()
            self.log(f"✓ {STAGE_LABELS[st]} 완료 ({time.time()-t0:.0f}s)")
        if not self.cfg.output.keep_cache and "measure" in stages:
            for f in self.cache.glob("*"): f.unlink()
        self.log("완료:", self.out)

    # ---------------- stages
    def stage_detect(self):
        c = self.cfg; A = self.state["A"]
        LAB, P, summ = detection.detect_stack(A, c.detection, lambda f, m: self._prog("detect", f, m))
        P.round(3).to_csv(self.out / "particles_all_slices.csv", index=False); summ.round(3).to_csv(self.out / "summary_per_slice.csv", index=False)
        tifffile.imwrite(self.out / "particle_labels.tif", LAB, imagej=LAB.dtype == np.uint16)
        self._save("LAB", LAB); self._save("P", P); self.state.update(LAB=LAB, P=P)
        self.log(f"입자 {len(P)}개 (프레임당 중앙값 {int(summ.n_particles.median())})")

    def stage_register(self):
        s = self.state
        reg = registration.register(s["A"], s["B"], s["HT"], self.cfg.registration, lambda f, m: self._prog("register", f, m))
        reg.round(4).to_csv(self.out / "registration.csv", index=False); self._save("reg", reg); s["reg"] = reg
        self.log(f"형광→HT 상관 중앙값 {reg.corr_FL_HT.median():.2f}, Cy5↔pHrodo {reg.corr_A_B.median():.2f}")

    def stage_cells(self):
        c = self.cfg; s = self.state; HT = s["HT"]
        cim = cells.cell_body_images(HT, c.cells)
        CID, low, core = cells.segment_cells(cim, c.cells, lambda f, m: self._prog("cells", f * .5, m))
        FLOW = cells.cell_flow(cim, c.cells, lambda f, m: self._prog("cells", .5 + f * .4, m)); del cim
        FP, fps = cells.footprint(HT, CID, c.footprint); CS = cells.mitosis_table(HT, CID, c.mitosis)
        fps.round(4).to_csv(self.out / "footprint_per_frame.csv", index=False); CS.round(3).to_csv(self.out / "cell_stats_mitosis.csv", index=False)
        for k, v in dict(CID=CID, FLOW=FLOW, FP=FP, CS=CS, cell_thr=(low, core)).items(): self._save(k, v); s[k] = v
        mit = CS[CS.mitotic]
        self.log(f"세포 threshold {low:.1f}/{core:.1f}, 분열 판정 세포 {sorted(mit.cell_id.unique().tolist())}")

    def stage_track(self):
        c = self.cfg; s = self.state; A = s["A"]
        df = tracking.prepare_detections(self._get("P"), self._get("reg"), s["S"], self._get("CID"), self._get("FP"), self._get("CS"), self._get("FLOW"), s["HT"])
        df = tracking.pattern_predictions(df, A, self._get("reg"), s["S"], c.detection, c.tracking, lambda f, m: self._prog("track", f * .5, m))
        if c.tracking.appearance or tracking.load_learned(c.tracking): df = tracking.add_ht_features(df, self._get("LAB"), s["HT"], self._get("reg"), s["S"])
        LK, AMB, PR = tracking.link_frames(df, c.tracking, lambda f, m: self._prog("track", .5 + f * .45, m))
        W = None; st = tracking.link_frames.last_appearance
        if c.tracking.appearance and st:
            W = {k: v["weight_px"] for k, v in st.items() if k in tracking.APPEARANCE}
            (self.out / "tracking_appearance_weights.json").write_text(json.dumps(st, indent=1, ensure_ascii=False), encoding="utf-8")
            self.log("외형 가중치 (px 환산):", {k: round(v, 3) for k, v in W.items()})
        TR, ng = tracking.build_tracks(df, LK, c.tracking, PR, app_w=W)
        self.log(f"연결 {len(LK)}, 트랙 {TR.track_id.nunique()}, gap closing {ng}, 모호한 연결 {np.mean(AMB) * 100 if len(AMB) else 0:.1f}%")
        tr = tracking.add_motion_columns(TR, c.channel.frame_interval_min)
        self._save("tracks_raw", tr); s["tracks_raw"] = tr

    def stage_measure(self):
        c = self.cfg; s = self.state
        reg = self._get("reg"); Bal = registration.align_b(s["B"], reg); s["Bal"] = Bal
        tr = measurement.measure_rois(self._get("tracks_raw"), s["A"], Bal, s["HT"], self._get("LAB"), reg, s["S"], c.measurement,
                                      lambda f, m: self._prog("measure", f * .8, m))
        tr = measurement.flag_merged(tr, c.measurement)
        self.log(f"합쳐짐/가림 시점 {int(tr.merged.sum())}개 ({tr.merged.mean() * 100:.1f}%)" + (" — 비율에서 제외" if c.measurement.exclude_merged else ""))
        TS = measurement.track_summary(tr, c.measurement, self._get("CS"), c.channel.frame_interval_min, c.mitosis.enabled)
        tr = tr.drop(columns=[x for x in ["group"] if x in tr]).merge(TS[["track_id", "group"]], on="track_id")
        TS.round(5).to_csv(self.out / "tracks_summary_A_B_HT.csv", index=False)
        measurement.points_table(tr).round(4).to_csv(self.out / "tracks_points_A_B_HT.csv", index=False)
        measurement.per_cell(tr, self._get("CID"), c.channel.frame_interval_min).round(5).to_csv(self.out / "particles_per_cell.csv", index=False)
        self._save("tracks", tr); s.update(tracks=tr, TS=TS)
        self.log("트랙 분류:", TS.group.value_counts().to_dict())

    def stage_figures(self):
        if not self.cfg.output.make_figures: return
        s = self.state; tr = self._get("tracks"); o = str(self.out)
        long = tr[tr.group.isin(["inside", "outside"])]
        plots.ratio_figures(long, o); self._prog("figures", .3)
        plots.qc_detection_cells(s["A"], self._get("P"), s["HT"], self._get("CID"), f"{o}/qc_detection_and_cells.png"); self._prog("figures", .5)
        plots.tracks_overlay(s["HT"], self._get("CID"), long, f"{o}/tracks_overlay.png")
        plots.regions_figure(s["HT"], self._get("CID"), self._get("FP"), self._get("CS"), f"{o}/regions_cytoplasm_spread_membrane.png")
        plots.mitosis_figure(self._get("CS"), f"{o}/mitosis_detection.png"); self._prog("figures", .9)

    def stage_movies(self):
        c = self.cfg
        if not c.output.make_movies: return
        s = self.state; tr = self._get("tracks"); reg = self._get("reg"); CID = self._get("CID")
        Bal = s.get("Bal") if "Bal" in s else registration.align_b(s["B"], reg)
        d = tr[tr.group.isin(["inside", "outside", "excluded_mitotic"])]; dt = c.channel.frame_interval_min
        jobs = [("HT_regions", s["HT"], False, d, dict(cellmask_fp=self._get("FP"), CS=self._get("CS")))]
        for g in ["inside", "outside"]:
            dg = d[d.group == g]; col = plots._track_colors(dg.track_id.unique())
            jobs += [(f"Cy5_{g}", s["A"], True, dg, dict(colors=col)), (f"pHrodo_{g}", Bal, True, dg, dict(colors=col)), (f"HT_{g}", s["HT"], False, dg, dict(colors=col))]
        for i, (nm, st, fl, pts, kw) in enumerate(jobs):
            movies.channel_movie(st, pts, str(self.out / f"movie_{nm}"), fl, s["S"], CID, reg, dt, nm, c.output.movie_fps,
                                 progress=lambda f, m, i=i: self._prog("movies", (i + f) / len(jobs), m), **kw)


# ======================================================================= project-level
def combine_and_plot(dataset_dirs: dict, out_dir, cfg: Config, log=print):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    pts = analysis.combine(dataset_dirs)
    if pts.empty: raise ValueError("통합할 결과가 없습니다 (tracks_points_A_B_HT.csv 없음)")
    pts.to_csv(out / "combined_tracks_points_A_B_HT.csv", index=False)
    (out / "datasets_map.json").write_text(json.dumps({k: str(v) for k, v in dataset_dirs.items()}, ensure_ascii=False, indent=1), encoding="utf-8")
    pts.groupby(["dataset", "group"]).track_uid.nunique().unstack(fill_value=0).to_csv(out / "track_counts_by_group.csv")
    long = pts[pts.group.isin(["inside", "outside"])]
    plots.ratio_figures(long, str(out), "combined_", idcol="track_uid"); plots.per_dataset_medians(long, str(out / "combined_per_dataset_medians.png"))
    T = long.groupby(["dataset", "group", "track_uid"]).ratio_BA.mean().reset_index()
    summ = T.groupby(["dataset", "group"]).ratio_BA.median().unstack(); summ.loc["전체"] = T.groupby("group").ratio_BA.median()
    summ.round(5).to_csv(out / "combined_track_mean_ratio_median.csv"); log("통합 완료:", out)
    return pts


def run_risefall(pts: pd.DataFrame, cfg: Config, out_dir, log=print):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    R, F = analysis.risefall(pts, cfg.risefall, cfg.channel.frame_interval_min)
    if R.empty: raise ValueError("분류할 장기 트랙이 없습니다")
    R.round(5).to_csv(out / "risefall_classification_all_tracks.csv", index=False); F.round(3).to_csv(out / "risefall_fraction_summary.csv", index=False)
    plots.risefall_fraction(F, str(out / "risefall_fraction.png"))
    c = R[R.candidate].sort_values(["category", "dataset", "peak_frame"])
    plots.risefall_multiples(c[(c.group == "inside") & c.keep], pts, str(out / "risefall_traces_inside_kept.png"), f"선별된 세포 안 입자 {int(((c.group=='inside')&c.keep).sum())}개")
    plots.risefall_multiples(c[(c.group == "inside") & ~c.keep], pts, str(out / "risefall_traces_inside_excluded.png"), "제외된 세포 안 후보와 제외 이유")
    plots.risefall_multiples(c[c.group == "outside"], pts, str(out / "risefall_traces_outside_all.png"), "세포 밖 후보 (빨간 제목 = 제외)")
    plots.risefall_aligned(R, pts, str(out / "risefall_aligned_average.png"))
    log("특이 입자:", F[F.dataset == "전체"][["group", "n_tracks", "kept", "kept_pct"]].to_dict("records"))
    return R, F


def make_crop(track_uid: str, pts: pd.DataFrame, R: pd.DataFrame | None, dataset_dirs: dict, cfg: Config, out_dir, trace_png=True):
    """선택한 트랙의 크롭 영상 + 곡선 그래프. dataset_dirs: {dataset: 결과 폴더(dataset.json 포함)}"""
    import matplotlib.pyplot as plt
    ds, _ = track_uid.split("::", 1) if "::" in track_uid else (pts.dataset.iloc[0], track_uid)
    spec = DatasetSpec.load(Path(dataset_dirs[ds]) / "dataset.json"); A, B, HT = load_dataset(spec)
    reg = pd.read_csv(Path(dataset_dirs[ds]) / "registration.csv"); Bal = registration.align_b(B, reg)
    t = pts[pts.track_uid == track_uid]; r = None
    if R is not None and track_uid in set(R.track_uid): r = R[R.track_uid == track_uid].iloc[0]
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True); safe = track_uid.replace("::", "_").replace("/", "_")
    cat = plots.CNAME.get(getattr(r, "category", ""), "") if r is not None else ""
    info = f"{ds} | {track_uid.split('::')[-1]} | {plots.GNAME.get(t.group.iloc[0], t.group.iloc[0])} {cat}"
    mv = movies.crop_movie(A, Bal, HT, t, HT.shape[1] / A.shape[1], str(out / f"crop_{safe}"), info, r, cfg.output.crop_half, cfg.output.movie_fps)
    if trace_png:
        fig, ax = plt.subplots(figsize=(9, 4)); plots.track_trace(ax, t, r, show_channels=True); ax.set_title(info); fig.savefig(out / f"trace_{safe}.png", dpi=110, bbox_inches="tight"); plt.close(fig)
    return mv

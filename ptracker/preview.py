"""파라미터 미리보기 계산 (Qt 없음).

데이터셋에서 필요한 프레임만 읽어 한 단계를 계산하고, 선택한 영역만 잘라 돌려줍니다.
threshold가 영상 전체 분포로 정해지는 단계(입자 검출, 세포 영역, 퍼진 세포막)는 전체 프레임으로 계산한 뒤 자르므로
실제 실행 결과와 같습니다. 알고리즘은 전체 실행과 같은 함수(detect_frame, segment_frame, footprint_frame,
register_frame, measure_frame, classify_mitosis)를 씁니다.

결과 dict
  kind     : "image" | "plot" | "message"
  image    : 2D(회색) 또는 RGB 배열 (kind=image)
  overlays : [{"type": "contour"|"fill"|"points"|"text", ...}]
  series   : 분열 그래프 데이터 (kind=plot)
  info     : 요약 문자열
  overview : 전체 프레임 축소 영상 (영역 선택용), crop_box: 축소 영상 좌표의 선택 영역
  params   : 이 섹션의 파라미터 dict (이전/현재 비교 표시용)
"""
from __future__ import annotations
import os, pickle
from collections import OrderedDict
from dataclasses import asdict
from pathlib import Path
import numpy as np, pandas as pd, tifffile
from skimage.segmentation import find_boundaries
from skimage.transform import resize
from .config import Config
from .io_utils import DatasetSpec
from . import detection, registration, cells, measurement

SF_DATALESS = 0x40000000          # macOS: iCloud로만 옮겨진(로컬에 내용이 없는) 파일
PREVIEW_SECTIONS = {"detection": "입자 검출", "registration": "정합", "cells": "세포 영역", "footprint": "퍼진 세포막",
                    "mitosis": "분열 세포", "measurement": "측정"}
OVERVIEW_PX = 256


class DataUnavailable(Exception):
    pass


def is_dataless(path) -> bool:
    try: return bool(getattr(os.stat(path), "st_flags", 0) & SF_DATALESS)
    except OSError: return False


def check_files(spec: DatasetSpec):
    for nm, p in (("Cy5", spec.cy5), ("pHrodo", spec.phrodo), ("HT", spec.ht)):
        if not Path(p).is_file(): raise DataUnavailable(f"{nm} 파일 없음: {p}")
        if is_dataless(p):
            raise DataUnavailable(f"{nm} 파일이 iCloud에만 있고 이 Mac에 내려받아져 있지 않습니다:\n{p}\n"
                                  "Finder에서 파일을 우클릭 → [지금 다운로드]한 뒤 다시 시도하세요.")


def _shape(path):
    with tifffile.TiffFile(path) as t: s = [d for d in t.series[0].shape if d != 1]
    return (1, *s) if len(s) == 2 else tuple(s)


def _norm(im, p=(0.5, 99.8)):
    lo, hi = np.percentile(im, p); return np.clip((im.astype(np.float32) - lo) / max(hi - lo, 1e-9), 0, 1)


class PreviewSource:
    """한 데이터셋의 프레임 단위 읽기 + 중간 결과 메모. 한 스레드(미리보기 작업 스레드)에서만 사용합니다."""

    def __init__(self, spec: DatasetSpec, max_frames=12):
        check_files(spec); self.spec = spec
        self.T, self.NF, self.NW = _shape(spec.cy5); _, self.NH, self.NHW = _shape(spec.ht)
        self.S = self.NH / self.NF; self.cache_dir = Path(spec.out_dir) / "_cache"
        self._frames: OrderedDict = OrderedDict(); self._max = max_frames; self._memo: OrderedDict = OrderedDict()

    def frame(self, ch, z):
        """ch ∈ {cy5, phrodo, ht}, z는 0부터."""
        key = (ch, z)
        if key in self._frames: self._frames.move_to_end(key); return self._frames[key]
        path = getattr(self.spec, ch)
        if is_dataless(path): raise DataUnavailable(f"{path} 가 iCloud에만 있습니다. 내려받은 뒤 다시 시도하세요.")
        a = np.squeeze(tifffile.imread(path, key=z)) if self.T > 1 else np.squeeze(tifffile.imread(path))
        if a.ndim != 2: a = np.squeeze(tifffile.imread(path))[z]           # 페이지 = 프레임이 아닌 파일
        self._frames[key] = a
        while len(self._frames) > self._max: self._frames.popitem(last=False)
        return a

    def memo(self, key, fn, keep=6):
        if key in self._memo: self._memo.move_to_end(key); return self._memo[key]
        v = fn(); self._memo[key] = v
        while len(self._memo) > keep: self._memo.popitem(last=False)
        return v

    def cached(self, name):
        p = self.cache_dir / f"{name}.pkl"
        if not p.exists(): return None
        with open(p, "rb") as f: return pickle.load(f)

    # ---------------------------------------------------------------- 공통
    def overview(self, z):
        return self.memo(("ov", z), lambda: _norm(resize(self.frame("cy5", z).astype(np.float32), (OVERVIEW_PX, round(OVERVIEW_PX * self.NW / self.NF)),
                                                          anti_aliasing=True), (1, 99.8)))

    def fl_box(self, cy, cx, size):
        h = size // 2; y0 = int(np.clip(cy - h, 0, max(self.NF - size, 0))); x0 = int(np.clip(cx - h, 0, max(self.NW - size, 0)))
        return y0, min(y0 + size, self.NF), x0, min(x0 + size, self.NW)

    def ht_box(self, box, z):
        reg = self.cached("reg"); dy = dx = 0.0
        if reg is not None and len(reg) > z: dy, dx = reg.dy_ht.values[z], reg.dx_ht.values[z]
        y0, y1, x0, x1 = box; f = lambda v, d, n: int(np.clip(round((v + 0.5) * self.S - 0.5 + d), 0, n))
        return f(y0, dy, self.NH), f(y1, dy, self.NH), f(x0, dx, self.NHW), f(x1, dx, self.NHW)


def _params(cfg: Config, *sections):
    return {f"{s}.{k}": v for s in sections for k, v in asdict(getattr(cfg, s)).items()}


def _crop(a, box): y0, y1, x0, x1 = box; return a[y0:y1, x0:x1]


# ================================================================ 섹션별 미리보기
def _detect(src: PreviewSource, z, cfg: Config):
    key = ("det", z, tuple(asdict(cfg.detection).values()))
    return src.memo(key, lambda: detection.detect_frame(src.frame("cy5", z), cfg.detection))


def preview_detection(src, z, box, cfg):
    lab, thr = _detect(src, z, cfg); raw = src.frame("cy5", z); lc = _crop(lab, box); n_all = int(lab.max())
    ids = np.unique(lc[lc > 0]); area = np.bincount(lab.ravel())[1:]
    ys, xs = [], []
    for k in ids:
        yy, xx = np.nonzero(lc == k); ys.append(yy.mean()); xs.append(xx.mean())
    info = (f"frame {z+1} 전체: 입자 {n_all}개, threshold {thr:.2f}, 면적 중앙값 {np.median(area) if len(area) else 0:.0f} px\n"
            f"선택 영역: 입자 {len(ids)}개")
    return dict(kind="image", image=_norm(_crop(raw, box), (1, 99.8)),
                overlays=[dict(type="contour", mask=find_boundaries(lc, mode="inner"), color="red"),
                          dict(type="points", y=ys, x=xs, color="yellow", marker="+")], info=info, params=_params(cfg, "detection"))


def _cell_thr(src, cfg):
    p = cfg.cells; key = ("cthr", p.open_radius, p.smooth_sigma)

    def calc():
        sample = cells.cell_body_images(np.stack([src.frame("ht", z) for z in range(0, src.T, 4)]), p)
        return cells.cell_thresholds(sample)
    return src.memo(key, calc)


def _cells(src, z, cfg):
    p = cfg.cells; key = ("cells", z, tuple(asdict(p).values()))

    def calc():
        low, core = _cell_thr(src, cfg); img = cells.cell_body_images(src.frame("ht", z)[None], p)[0]
        return cells.segment_frame(img, low, core, p), low, core
    return src.memo(key, calc)


def preview_cells(src, z, box, cfg):
    lab, low, core = _cells(src, z, cfg); hb = src.ht_box(box, z); lc = _crop(lab, hb)
    areas = np.bincount(lab.ravel())[1:]; areas = areas[areas > 0]
    info = (f"threshold: 세포 영역 {low:.1f} / 세포 중심 {core:.1f} (4프레임 간격 표본, 전체 실행과 같음)\n"
            f"frame {z+1} 전체: 세포 {len(areas)}개, 세포질 비율 {(lab > 0).mean() * 100:.1f}%\n"
            "※ 'ID 연결 겹침 비율'과 'optical flow'는 여러 프레임에 걸친 값이라 여기서는 보이지 않습니다")
    return dict(kind="image", image=_norm(_crop(src.frame("ht", z), hb)),
                overlays=[dict(type="labels", labels=lc), dict(type="contour", mask=find_boundaries(lc, mode="inner"), color="red")],
                info=info, params=_params(cfg, "cells"))


def preview_footprint(src, z, box, cfg):
    lab, _, _ = _cells(src, z, cfg); ht = src.frame("ht", z)
    FP, st = src.memo(("fp", z, tuple(asdict(cfg.cells).values()), tuple(asdict(cfg.footprint).values())),
                      lambda: cells.footprint_frame(ht, lab, cfg.footprint))
    hb = src.ht_box(box, z); lc = _crop(lab, hb); sm = _crop(FP & (lab == 0), hb)
    info = (f"배경(배지) 최빈값 {st['bg_mode']:.1f}, σ {st['bg_sigma']:.2f} → threshold {st['threshold']:.1f}\n"
            f"frame {z+1} 전체: 세포질 {st['cytoplasm_frac'] * 100:.1f}%, 퍼진 세포막 {st['spread_membrane_frac'] * 100:.1f}%\n"
            "빨강 = 세포질(세포 영역 파라미터), 파랑 = 퍼진 세포막")
    return dict(kind="image", image=_norm(_crop(ht, hb)),
                overlays=[dict(type="fill", mask=sm, color=(0.1, 0.55, 1.0), alpha=0.45),
                          dict(type="contour", mask=find_boundaries(lc > 0, mode="inner"), color="red")],
                info=info, params=_params(cfg, "cells", "footprint"))


def preview_registration(src, z, box, cfg):
    a, b, ht = src.frame("cy5", z), src.frame("phrodo", z), src.frame("ht", z)
    c, dy, dx, cb, by, bx, c0 = src.memo(("reg", z, tuple(asdict(cfg.registration).values())),
                                         lambda: registration.register_frame(a, b, ht, cfg.registration))
    p = cfg.registration
    y0, y1, x0, x1 = box; f = lambda v, d, n: int(np.clip(round((v + 0.5) * src.S - 0.5 + d), 0, n))
    hb = (f(y0, dy, src.NH), f(y1, dy, src.NH), f(x0, dx, src.NHW), f(x1, dx, src.NHW))
    H = _norm(_crop(ht, hb)); F = _norm(resize(_crop(a, box).astype(np.float32), H.shape, anti_aliasing=True), (1, 99.8))
    rgb = np.dstack([F, H * 0.85, F])        # 자홍 = Cy5(정합 후), 초록 = HT
    warn = []
    if c < p.fl_ht_min_corr: warn.append(f"형광→HT 상관 {c:.2f} < {p.fl_ht_min_corr} → 전체 실행에서는 다른 프레임 이동량 중앙값으로 대체")
    if cb < p.ab_min_corr: warn.append(f"Cy5↔pHrodo 상관 {cb:.2f} < {p.ab_min_corr} → 전체 실행에서는 중앙값으로 대체")
    info = (f"형광→HT 이동 ({dy:+.2f}, {dx:+.2f}) HT px, 상관 {c:.2f}\n"
            f"Cy5↔pHrodo 이동 ({int(by):+d}, {int(bx):+d}) px, 상관 {cb:.2f} (이동 전 {c0:.2f})\n"
            "자홍 = Cy5(이동 적용), 초록 = HT — 입자가 흰색으로 겹치면 정합이 맞는 것" + ("\n⚠ " + "\n⚠ ".join(warn) if warn else ""))
    return dict(kind="image", image=rgb, overlays=[], info=info, params=_params(cfg, "registration"))


def preview_mitosis(src, z, box, cfg):
    CS = src.cached("CS")
    if CS is None:
        return dict(kind="message", info="분열 판정에는 모든 프레임의 세포 분할 결과가 필요합니다.\n"
                                         "[데이터·실행]에서 '세포·영역·분열' 단계를 먼저 실행하세요.", params=_params(cfg, "mitosis"))
    C = cells.classify_mitosis(CS, cfg.mitosis); mc = sorted(C[C.mitotic].cell_id.unique().tolist())
    series = [dict(cell=int(c), frame=s.frame.values, ri=s.meanRI.values / 1e4, area=s.area.values, mit=s.mitotic.values)
              for c, s in C.groupby("cell_id") if len(s) >= 5]
    info = (f"분열 판정 세포: {mc if mc else '없음'} (판정 {int(C.mitotic.sum())} 세포·프레임)\n"
            "세포 분할은 마지막으로 실행한 결과(캐시)를 사용합니다. 빨간 점 = 분열 판정")
    return dict(kind="plot", series=series, info=info, params=_params(cfg, "mitosis"))


def preview_measurement(src, z, box, cfg):
    lab, _ = _detect(src, z, cfg); reg = src.cached("reg")
    if reg is not None and len(reg) > z: r = reg.iloc[z]; dyh, dxh, dyb, dxb = r.dy_ht, r.dx_ht, int(r.dy_B), int(r.dx_B); src_txt = "정합: 캐시"
    else:
        c, dyh, dxh, cb, dyb, dxb, _ = src.memo(("reg", z, tuple(asdict(cfg.registration).values())),
                                                lambda: registration.register_frame(src.frame("cy5", z), src.frame("phrodo", z), src.frame("ht", z), cfg.registration))
        dyb, dxb = int(dyb), int(dxb); src_txt = "정합: 이 프레임만 계산"
    a = src.frame("cy5", z); b = np.roll(src.frame("phrodo", z), (dyb, dxb), (0, 1)); lc = _crop(lab, box)
    ids = [int(k) for k in np.unique(lc[lc > 0])]
    rec, masks = measurement.measure_frame(lab, a, b, src.frame("ht", z), dyh, dxh, src.S, cfg.measurement, labels=ids, return_masks=True)
    y0, _, x0, _ = box; H, W = lc.shape; roi = np.zeros((H, W), bool); ring = np.zeros((H, W), bool); ys, xs, txt = [], [], []
    vals = []
    for k, (IA, IB, _, bgA, bgB, _, _) in rec.items():
        my, mx, m, rg = masks[k]
        for dst, src_m in ((roi, m), (ring, rg)):
            yy, xx = np.nonzero(src_m); yy = yy + my - y0; xx = xx + mx - x0; ok = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
            dst[yy[ok], xx[ok]] = True
        ia, ib = IA - bgA, IB - bgB; ratio = ib / ia if ia > cfg.measurement.min_ia else np.nan; vals.append((ia, ib, ratio))
        yy, xx = np.nonzero(m); ys.append(yy.mean() + my - y0); xs.append(xx.mean() + mx - x0)
        txt.append("—" if np.isnan(ratio) else f"{ratio * 100:.1f}")
    v = np.array(vals) if vals else np.zeros((0, 3))
    info = (f"선택 영역 ROI {len(ids)}개 ({src_txt}) · 파란 고리 = 배경 측정 영역, 숫자 = I(pHrodo)/I(Cy5) × 100\n"
            f"중앙값: I(Cy5) {np.nanmedian(v[:, 0]) if len(v) else np.nan:.1f}, I(pHrodo) {np.nanmedian(v[:, 1]) if len(v) else np.nan:.2f}, "
            f"비율 {np.nanmedian(v[:, 2]) if len(v) and np.isfinite(v[:, 2]).any() else np.nan:.4f} · I(Cy5) ≤ {cfg.measurement.min_ia:g} 로 비율 제외 {int(np.isnan(v[:, 2]).sum()) if len(v) else 0}개\n"
            "※ '장기 트랙 최소 프레임 수'와 '세포 안 판정 비율'은 트래킹 결과가 있어야 하므로 여기서는 보이지 않습니다")
    ov = [dict(type="fill", mask=ring, color=(0.2, 0.6, 1.0), alpha=0.35), dict(type="contour", mask=find_boundaries(roi, mode="inner"), color="yellow")]
    if len(ids) <= 60: ov.append(dict(type="text", y=[y + cfg.measurement.ring_out + 4 for y in ys], x=xs, text=txt, color="yellow"))
    return dict(kind="image", image=_norm(_crop(a, box), (1, 99.8)), overlays=ov, info=info, params=_params(cfg, "detection", "measurement"))


PREVIEWS = dict(detection=preview_detection, registration=preview_registration, cells=preview_cells,
                footprint=preview_footprint, mitosis=preview_mitosis, measurement=preview_measurement)


def run_preview(src: PreviewSource, section, z, center, size, cfg: Config) -> dict:
    """center = (cy, cx) 형광 좌표. 없으면 영상 중앙."""
    z = int(np.clip(z, 0, src.T - 1)); cy, cx = center if center else (src.NF // 2, src.NW // 2)
    box = src.fl_box(cy, cx, size)
    if section in PREVIEWS: out = PREVIEWS[section](src, z, box, cfg)
    elif section == "tracking": out = dict(kind="message", info="트래킹 미리보기는 다음 단계(3단계)에서 추가될 예정입니다.")
    elif section == "risefall": out = dict(kind="message", info="특이 입자 기준은 [5. 특이 입자] 탭에서 분류를 실행하면 곡선으로 확인할 수 있습니다.")
    else: out = dict(kind="message", info="이 섹션은 영상으로 확인할 항목이 없습니다.")
    k = OVERVIEW_PX / src.NF
    out.update(overview=src.overview(z), crop_box=(box[0] * k, box[1] * k, box[2] * k, box[3] * k), section=section, frame=z,
               key=(src.spec.cy5, z, box))
    return out

"""트랙을 영상 위에서 보는 위젯들 (결과 보기·특이 입자 탭에서 공용).

TrackMapView : 전체 프레임 영상 + 트랙 경로. 경로를 클릭하면 그 트랙을 선택(trackClicked).
CropViewer   : 선택한 트랙을 따라가는 Cy5 / pHrodo / HT 크롭 (재생·슬라이더, 또는 프레임 모아보기) + 비율 곡선.
TrackExplorer: [결과 보기] 탭의 트랙 탐색 (트랙 표 + 경로 + 밝기 그래프/크롭).

밝기는 크롭 스택 전체 기준으로 맞춰(프레임마다 따로 맞추지 않음) 시간에 따른 밝기 변화가 그대로 보입니다
(movies.crop_movie와 같은 규칙).
"""
from __future__ import annotations
import json, traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QSpinBox, QCheckBox, QPushButton, QSlider,
                               QSplitter, QTableView, QAbstractItemView, QTabWidget)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle
from ..io_utils import DatasetSpec
from ..preview import PreviewSource, DataUnavailable
from ..plots import track_trace, GNAME
from .table_model import DataFrameModel

GROUP_COLORS = {"inside": "#ff5a5a", "outside": "#4aa3ff", "excluded_mitotic": "#ffa500", "short": "#9a9a9a"}
GROUP_LABELS = {**GNAME, "excluded_mitotic": "분열 제외", "short": "짧은 트랙"}


class DatasetContext:
    """결과 폴더(dataset.json·registration.csv)에서 원본 영상 위치와 정합 정보를 읽습니다."""

    def __init__(self, folder):
        self.folder = Path(folder); self.spec = DatasetSpec.load(self.folder / "dataset.json")
        f = self.folder / "registration.csv"; self.reg = pd.read_csv(f) if f.exists() else None

    def shift_B(self, frame):
        if self.reg is None or len(self.reg) < frame: return 0, 0
        r = self.reg.iloc[frame - 1]; return int(r.dy_B), int(r.dx_B)


def _crop(im, x, y, h):
    """(x, y) 중심 2h×2h 크롭 (밖은 0) — movies.crop_movie와 같은 방식."""
    xi, yi = int(round(x)), int(round(y)); p = np.pad(im, h); return p[yi:yi + 2 * h, xi:xi + 2 * h]


class _Bridge(QObject):
    done = Signal(int, object)


# ====================================================================== 전체 영상 + 경로
class TrackMapView(QWidget):
    trackClicked = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._ctx = None; self._src = None; self._err = ""; self._pts = pd.DataFrame(); self._markers = {}; self._sel = None
        self._bg = {}; self._tree = None; self._tree_uids = None; self._view_key = None; self._tree_cols = None
        v = QVBoxLayout(self); v.setContentsMargins(0, 0, 0, 0); row = QHBoxLayout()
        self.ch = QComboBox(); self.ch.addItems(["Cy5", "HT"]); self.ch.currentTextChanged.connect(lambda _: self.redraw(keep_view=False))
        self.fr = QSpinBox(); self.fr.setRange(1, 1); self.fr.setPrefix("frame "); self.fr.valueChanged.connect(lambda _: self.redraw())
        self.paths = QCheckBox("경로"); self.paths.setChecked(True); self.paths.toggled.connect(lambda _: self.redraw())
        self.lbl = QLabel(""); self.lbl.setStyleSheet("color: gray")
        for w in (QLabel("배경"), self.ch, self.fr, self.paths): row.addWidget(w)
        row.addWidget(self.lbl, 1); v.addLayout(row)
        self.fig = Figure(figsize=(6, 6)); self.fig.subplots_adjust(0, 0, 1, 1); self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasQTAgg(self.fig); self.tb = NavigationToolbar2QT(self.canvas, self)
        self.canvas.mpl_connect("button_press_event", self._on_click)
        v.addWidget(self.tb); v.addWidget(self.canvas, 1)

    def set_context(self, ctx: DatasetContext | None):
        if ctx is not None and self._ctx is not None and ctx.folder == self._ctx.folder: return
        self._ctx = ctx; self._bg.clear(); self._src = None; self._err = ""; self._view_key = None
        if ctx is None: self._err = "원본 데이터셋 위치 정보가 없습니다 (dataset.json)"; return
        try:
            self._src = PreviewSource(ctx.spec, max_frames=2)
            self.fr.blockSignals(True); self.fr.setRange(1, self._src.T); self.fr.blockSignals(False)
        except DataUnavailable as e: self._err = str(e)

    def set_tracks(self, pts: pd.DataFrame, markers: dict | None = None):
        """pts: track_uid, frame, x_fl, y_fl, x_ht, y_ht, group 열. markers: {track_uid: frame} 별표 위치."""
        self._pts = pts; self._markers = markers or {}; self._tree = None; self.redraw()

    def select(self, uid, frame=None):
        self._sel = uid
        if frame is not None and frame != self.fr.value():
            self.fr.blockSignals(True); self.fr.setValue(int(frame)); self.fr.blockSignals(False)
        self.redraw()

    def frame(self): return self.fr.value()

    def _cols(self): return ("x_fl", "y_fl") if self.ch.currentText() == "Cy5" else ("x_ht", "y_ht")

    def _background(self):
        ch = "cy5" if self.ch.currentText() == "Cy5" else "ht"; z = self.fr.value() - 1; key = (ch, z)
        if key not in self._bg:
            im = self._src.frame(ch, z).astype(np.float32); lo, hi = np.percentile(im[::4, ::4], (0.5, 99.8) if ch == "ht" else (1, 99.7))
            self._bg.clear(); self._bg[key] = np.clip((im - lo) / max(hi - lo, 1e-9), 0, 1)
        return self._bg[key]

    def redraw(self, keep_view=True):
        key = (self._ctx.folder if self._ctx else None, self.ch.currentText())
        lim = (self.ax.get_xlim(), self.ax.get_ylim()) if keep_view and self._view_key == key else None
        self.ax.clear(); self.ax.axis("off")
        if self._src is None:
            self.ax.text(.5, .5, self._err or "트랙을 선택하세요", ha="center", va="center", transform=self.ax.transAxes, fontsize=9, wrap=True)
            self.canvas.draw_idle(); return
        try: self.ax.imshow(self._background(), cmap="gray", interpolation="nearest")
        except DataUnavailable as e:
            self.ax.text(.5, .5, str(e), ha="center", va="center", transform=self.ax.transAxes, fontsize=9, wrap=True); self.canvas.draw_idle(); return
        xc, yc = self._cols(); P = self._pts; z = self.fr.value(); n = 0
        if len(P):
            n = P.track_uid.nunique()
            if self.paths.isChecked():
                for g, d in P.groupby("group"):
                    segs = [s[[xc, yc]].values for _, s in d.sort_values("frame").groupby("track_uid") if len(s) > 1]
                    self.ax.add_collection(LineCollection(segs, colors=GROUP_COLORS.get(g, "w"), linewidths=.7, alpha=.75))
            now = P[P.frame == z]
            for g, d in now.groupby("group"): self.ax.plot(d[xc], d[yc], "o", ms=3, color=GROUP_COLORS.get(g, "w"), mec="none")
            if self._markers:
                m = P.merge(pd.Series(self._markers, name="mf").rename_axis("track_uid").reset_index(), on="track_uid")
                m = m[m.frame == m.mf]; self.ax.plot(m[xc], m[yc], "*", ms=9, color="yellow", mec="k", mew=.5, ls="none")
            if self._sel is not None:
                s = P[P.track_uid == self._sel].sort_values("frame")
                if len(s):
                    self.ax.plot(s[xc], s[yc], "-", color="yellow", lw=2.2); sz = s[s.frame == z]
                    self.ax.plot(s[xc].iloc[0], s[yc].iloc[0], "o", ms=6, mfc="none", mec="lime", mew=1.5)
                    if len(sz): self.ax.add_patch(Circle((sz[xc].iloc[0], sz[yc].iloc[0]), 12, fill=False, ec="yellow", lw=1.5))
        if lim: self.ax.set_xlim(lim[0]); self.ax.set_ylim(lim[1])
        self._view_key = key
        leg = " · ".join(f"{GROUP_LABELS.get(g, g)}" for g in P.group.unique()) if len(P) else ""
        self.lbl.setText(f"트랙 {n}개 ({leg}) · 노란 선 = 선택 트랙, 초록 원 = 시작점, ★ = 최고점 · 경로를 클릭해 선택" if n else "")
        self.canvas.draw_idle()

    def _on_click(self, ev):
        if ev.inaxes is not self.ax or ev.xdata is None or self.tb.mode or not len(self._pts): return
        xc, yc = self._cols()
        if self._tree is None or self._tree_uids is None or self._tree_cols != (xc, yc):
            self._tree = cKDTree(self._pts[[xc, yc]].values); self._tree_uids = self._pts.track_uid.values; self._tree_cols = (xc, yc)
        d, i = self._tree.query([ev.xdata, ev.ydata])
        px = self.ax.transData.transform([[ev.xdata, ev.ydata], self._pts[[xc, yc]].values[i]])
        if np.hypot(*(px[0] - px[1])) <= 12: self.trackClicked.emit(str(self._tree_uids[i]))


# ====================================================================== 크롭
class CropViewer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._req = None; self._data = None; self._seq = 0; self._dirty = False
        self._pool = ThreadPoolExecutor(max_workers=1); self._bridge = _Bridge(); self._bridge.done.connect(self._on_done)
        v = QVBoxLayout(self); v.setContentsMargins(0, 0, 0, 0); row = QHBoxLayout()
        self.play = QPushButton("▶"); self.play.setFixedWidth(36); self.play.clicked.connect(self._toggle_play)
        self.slider = QSlider(Qt.Horizontal); self.slider.valueChanged.connect(lambda _: self._show_frame())
        self.grid = QCheckBox("프레임 모아보기"); self.grid.toggled.connect(lambda _: self._build_figure())
        self.lbl = QLabel("트랙을 선택하세요"); self.lbl.setStyleSheet("color: gray")
        for w in (self.play, self.slider, self.grid): row.addWidget(w)
        row.setStretch(1, 1); v.addLayout(row); v.addWidget(self.lbl)
        self.fig = Figure(figsize=(8, 6)); self.canvas = FigureCanvasQTAgg(self.fig); v.addWidget(self.canvas, 1)
        self._timer = QTimer(self); self._timer.timeout.connect(self._step)

    def set_track(self, ctx: DatasetContext | None, t: pd.DataFrame, r=None, half=32, dt_min=20.0, title="", fps=3):
        """t: 한 트랙의 point 표 (frame, x_fl, y_fl, x_ht, y_ht, ratio_BA, I_A, I_B, time_min, radius_equiv_px)."""
        self._timer.stop(); self.play.setText("▶")
        self._req = dict(ctx=ctx, t=t.sort_values("frame"), r=r, half=int(half), dt=dt_min, title=title); self._timer.setInterval(int(1000 / max(fps, 1)))
        if ctx is None: self._data = None; self.lbl.setText("원본 데이터셋 위치 정보가 없어 크롭을 만들 수 없습니다"); self.fig.clear(); self.canvas.draw_idle(); return
        if self.isVisible(): self._load()
        else: self._dirty = True

    def showEvent(self, e):
        super().showEvent(e)
        if self._dirty: self._dirty = False; self._load()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self._data is not None and not self.grid.isChecked():
            wide = self.canvas.width() > 2.2 * max(self.canvas.height(), 1)
            if wide != getattr(self, "_wide", None): self._wide = wide; self._build_figure()

    def shutdown(self):
        self._seq += 1; self._timer.stop(); self._pool.shutdown(wait=False, cancel_futures=True)

    def _load(self):
        self._seq += 1; seq = self._seq; self.lbl.setText("크롭 만드는 중…"); self._pool.submit(self._work, seq, self._req)

    def _work(self, seq, q):
        if seq != self._seq: return
        try:
            ctx, t, h = q["ctx"], q["t"], q["half"]; src = PreviewSource(ctx.spec, max_frames=3)
            f0, f1 = max(1, int(t.frame.min()) - 3), min(src.T, int(t.frame.max()) + 3); fr = np.arange(f0, f1 + 1)
            pos = {c: np.interp(fr, t.frame.values, t[c].values) for c in ["x_fl", "y_fl", "x_ht", "y_ht"]}
            hh = max(int(round(h * src.S)), 4); A, B, H = [], [], []
            for i, f in enumerate(fr):
                if seq != self._seq: return
                x, y = pos["x_fl"][i], pos["y_fl"][i]; dy, dx = ctx.shift_B(f)
                A.append(_crop(src.frame("cy5", f - 1), x, y, h)); B.append(_crop(src.frame("phrodo", f - 1), x - dx, y - dy, h))
                H.append(_crop(src.frame("ht", f - 1), pos["x_ht"][i], pos["y_ht"][i], hh))
            A, B, H = np.stack(A).astype(np.float32), np.stack(B).astype(np.float32), np.stack(H).astype(np.float32)
            ht_mid = src.frame("ht", src.T // 2); hlo, hhi = np.percentile(ht_mid[::4, ::4], [.5, 99.8])
            data = dict(fr=fr, A=A, B=B, H=H, h=h, hh=hh, S=src.S, va=(0, np.percentile(A, 99.8)), vb=(0, max(np.percentile(B, 99.8), 3)),
                        vh=(hlo, hhi), det=set(t.frame.astype(int)), rad=dict(zip(t.frame.astype(int), t.radius_equiv_px)) if "radius_equiv_px" in t else {})
            self._bridge.done.emit(seq, data)
        except DataUnavailable as e: self._bridge.done.emit(seq, str(e))
        except Exception: self._bridge.done.emit(seq, "크롭 오류:\n" + traceback.format_exc(limit=2))   # noqa: BLE001

    def _on_done(self, seq, d):
        if seq != self._seq: return
        if isinstance(d, str): self._data = None; self.lbl.setText(d); self.fig.clear(); self.canvas.draw_idle(); return
        self._data = d; self.slider.blockSignals(True); self.slider.setRange(0, len(d["fr"]) - 1)
        r, fr = self._req["r"], d["fr"]; pk = getattr(r, "peak_frame", None) if r is not None else None
        self.slider.setValue(int(np.searchsorted(fr, pk)) if pk is not None and np.isfinite(pk) else 0); self.slider.blockSignals(False)
        self._build_figure()

    def _build_figure(self):
        d = self._data; self.fig.clear()
        if d is None: self.canvas.draw_idle(); return
        q = self._req; fr = d["fr"]
        if self.grid.isChecked():
            n = min(len(fr), 10); idx = np.unique(np.linspace(0, len(fr) - 1, n).round().astype(int))
            axs = self.fig.subplots(3, len(idx), squeeze=False)
            for j, i in enumerate(idx):
                for k, (st, vr, nm, rad) in enumerate([(d["A"], d["va"], "Cy5", d["h"]), (d["B"], d["vb"], "pHrodo", d["h"]), (d["H"], d["vh"], "HT", d["hh"])]):
                    a = axs[k, j]; a.imshow(st[i], cmap="gray", vmin=vr[0], vmax=vr[1]); a.set_xticks([]); a.set_yticks([])
                    f = fr[i]; det = f in d["det"]
                    a.add_patch(Circle((rad - .5, rad - .5), (d["rad"].get(f, 4) + 2) * (d["S"] if k == 2 else 1), fill=False, ec="lime" if det else "yellow", lw=.8, ls="-" if det else "--"))
                    if k == 0: a.set_title(f"f{f} · {(f - 1) * q['dt'] / 60:.1f}h", fontsize=7)
                    if j == 0: a.set_ylabel(nm, fontsize=8)
            self.fig.suptitle(q["title"], fontsize=9); self.fig.subplots_adjust(.04, .02, .99, .86, .05, .12)
            self.lbl.setText(f"frame {fr[0]}–{fr[-1]} 중 {len(idx)}장 · 초록 원 = 검출, 노란 점선 = 미검출(위치 보간)")
            self.canvas.draw_idle(); return
        wide = self.canvas.width() > 2.2 * max(self.canvas.height(), 1)       # 넓고 낮은 영역: 크롭 3장과 곡선을 한 줄로
        if wide:
            gs = self.fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 2.6]); self._axs = [self.fig.add_subplot(gs[0, i]) for i in range(3)]; tr = gs[0, 3]
        else:
            gs = self.fig.add_gridspec(2, 3, height_ratios=[1.3, 1]); self._axs = [self.fig.add_subplot(gs[0, i]) for i in range(3)]; tr = gs[1, :]
        self._ims = []
        for a, (st, vr, nm) in zip(self._axs, [(d["A"], d["va"], "Cy5"), (d["B"], d["vb"], "pHrodo (정합)"), (d["H"], d["vh"], "HT")]):
            self._ims.append(a.imshow(st[0], cmap="gray", vmin=vr[0], vmax=vr[1])); a.set_title(nm, fontsize=9); a.axis("off")
        self._circ = [a.add_patch(Circle((0, 0), 1, fill=False, lw=1.3)) for a in self._axs]
        ax = self.fig.add_subplot(tr); track_trace(ax, q["t"], q["r"], show_channels=True)
        self._vl = ax.axvline(0, color="k", lw=1.5); self.fig.suptitle(q["title"], fontsize=9); self.fig.tight_layout()
        self._show_frame()

    def _show_frame(self):
        d = self._data
        if d is None or self.grid.isChecked() or not hasattr(self, "_ims"): return
        i = self.slider.value(); f = int(d["fr"][i]); det = f in d["det"]; dt = self._req["dt"]
        for k, (im, st) in enumerate(zip(self._ims, [d["A"], d["B"], d["H"]])):
            im.set_data(st[i]); rad = d["hh"] if k == 2 else d["h"]; s = d["S"] if k == 2 else 1
            c = self._circ[k]; c.set_center((rad - .5, rad - .5)); c.set_radius((d["rad"].get(f, 4) + 2) * s)
            c.set_edgecolor("lime" if det else "yellow"); c.set_linestyle("-" if det else "--")
        self._vl.set_xdata([(f - 1) * dt / 60] * 2)
        self.lbl.setText(f"frame {f} · t = {(f - 1) * dt / 60:.2f} h" + ("" if det else " · 미검출 (위치 보간)"))
        self.canvas.draw_idle()

    def _toggle_play(self):
        if self._timer.isActive(): self._timer.stop(); self.play.setText("▶")
        elif self._data is not None: self.grid.setChecked(False); self._timer.start(); self.play.setText("⏸")

    def _step(self):
        if self._data is None: self._timer.stop(); return
        self.slider.setValue((self.slider.value() + 1) % (self.slider.maximum() + 1))


# ====================================================================== 밝기 그래프
def draw_track_graph(fig, t, r=None, frame=None, dt_min=20.0, title=""):
    fig.clear(); gs = fig.add_gridspec(2, 1, height_ratios=[2, 1]); ax = fig.add_subplot(gs[0]); track_trace(ax, t, r, show_channels=True)
    ax.set_title(title, fontsize=9); a2 = fig.add_subplot(gs[1], sharex=ax); s = t.sort_values("frame")
    a2.plot(s.time_min / 60, s.HT_RI_mean, "o-", color="tab:green", ms=2.5, lw=1); a2.set_ylabel("ROI 평균 RI", fontsize=8)
    a2.set_xlabel("시간 (h)"); a2.grid(alpha=.3)
    if "inside_cell" in s:
        for a in (ax, a2):
            ins = s.inside_cell.astype(bool).values; th = s.time_min.values / 60
            for k in range(len(s) - 1):
                if ins[k]: a.axvspan(th[k], th[k + 1], color="tab:red", alpha=.06, lw=0)
    if frame is not None:
        for a in (ax, a2): a.axvline((frame - 1) * dt_min / 60, color="k", lw=1, ls=":")
    fig.tight_layout()


# ====================================================================== [결과 보기] 트랙 탐색
GROUP_FILTERS = {"장기 트랙 (세포 안+밖)": ["inside", "outside"], "세포 안": ["inside"], "세포 밖": ["outside"],
                 "분열 세포 제외됨": ["excluded_mitotic"], "짧은 트랙": ["short"], "전체": None}


def track_table(pts: pd.DataFrame) -> pd.DataFrame:
    g = pts.groupby("track_uid")
    T = g.agg(dataset=("dataset", "first"), track_id=("track_id", "first"), group=("group", "first"), n_points=("frame", "size"),
              start=("frame", "min"), end=("frame", "max"), ratio_mean=("ratio_BA", "mean"), ratio_max=("ratio_BA", "max"),
              I_A_mean=("I_A", "mean"), I_B_mean=("I_B", "mean"), frac_in_cell=("inside_cell", "mean")).reset_index()
    T["group"] = T.group.map(lambda x: GROUP_LABELS.get(x, x))
    # 주변 밀집도: 같은 프레임에서 10 HT px 안의 다른 검출 수의 트랙 평균 (정답 표시할 어려운 트랙 고르기용)
    crowd = np.zeros(len(pts))
    for f, idx in pts.groupby(["dataset", "frame"]).indices.items():
        xy = pts[["x_ht", "y_ht"]].values[idx]; crowd[idx] = [len(n) - 1 for n in cKDTree(xy).query_ball_point(xy, 10)]
    T = T.merge(pd.Series(crowd, index=pts.index).groupby(pts.track_uid).mean().rename("crowding").reset_index(), on="track_uid")
    if "merged" in pts: T = T.merge(pts.groupby("track_uid").merged.sum().rename("merged_frames").reset_index(), on="track_uid")
    return T


class TrackExplorer(QWidget):
    def __init__(self, get_config, parent=None):
        super().__init__(parent)
        self._get_config = get_config; self._folder = None; self._loaded = None; self._pts = None; self._dirs = {}; self._ctx = {}; self._T = None
        v = QVBoxLayout(self); row = QHBoxLayout()
        self.ds = QComboBox(); self.ds.currentTextChanged.connect(lambda _: self._apply_filter())
        self.grp = QComboBox(); self.grp.addItems(list(GROUP_FILTERS)); self.grp.currentTextChanged.connect(lambda _: self._apply_filter())
        self.minlen = QSpinBox(); self.minlen.setRange(1, 1000); self.minlen.setValue(10); self.minlen.setPrefix("최소 길이 "); self.minlen.valueChanged.connect(lambda _: self._apply_filter())
        self.info = QLabel(""); self.info.setStyleSheet("color: gray")
        for w in (QLabel("데이터셋"), self.ds, QLabel("그룹"), self.grp, self.minlen): row.addWidget(w)
        row.addWidget(self.info, 1); v.addLayout(row)
        self.table = QTableView(); self.model = DataFrameModel(); self.table.setModel(self.model); self.table.setSortingEnabled(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows); self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.selectionModel().selectionChanged.connect(lambda *_: self._on_select())
        self.map = TrackMapView(); self.map.trackClicked.connect(self.select_uid); self.map.fr.valueChanged.connect(lambda _: self._draw_graph())
        self.gfig = Figure(figsize=(7, 4)); self.gcanvas = FigureCanvasQTAgg(self.gfig); self.crop = CropViewer()
        self.annot = AnnotationPanel(); self.annot.get_selected = self._selected_track
        self.bottom = QTabWidget(); self.bottom.addTab(self.gcanvas, "밝기 그래프"); self.bottom.addTab(self.crop, "크롭 프레임"); self.bottom.addTab(self.annot, "정답 표시")
        self.lcheck = LinkCheckPanel(); self.bottom.addTab(self.lcheck, "연결 검증")
        top = QSplitter(); top.addWidget(self.table); top.addWidget(self.map); top.setSizes([520, 700])
        vs = QSplitter(Qt.Vertical); vs.addWidget(top); vs.addWidget(self.bottom); vs.setSizes([520, 360]); v.addWidget(vs, 1)

    def set_folder(self, folder):
        self._folder = folder or None
        if self.isVisible(): self._load()

    def showEvent(self, e):
        super().showEvent(e); self._load()

    def shutdown(self): self.crop.shutdown()

    def _load(self):
        if self._folder == self._loaded: return
        self._loaded = self._folder; self._pts = None; self._ctx = {}; self.model.set_df(pd.DataFrame()); self.map.set_context(None); self.map.set_tracks(pd.DataFrame())
        p = Path(self._folder) if self._folder else None
        if not p or not ((p / "tracks_points_A_B_HT.csv").exists() or (p / "combined_tracks_points_A_B_HT.csv").exists()):
            self.info.setText("이 폴더에는 트랙 결과(tracks_points_A_B_HT.csv)가 없습니다 — 데이터셋 결과 폴더나 통합 결과 폴더를 고르세요"); return
        from .tasks import load_points
        pts, dirs = load_points(p)
        if "track_uid" not in pts: pts = pts.assign(dataset=p.name, track_uid=p.name + "::" + pts.track_id.astype(str))
        self._pts, self._dirs = pts, dirs; self._T = track_table(pts)
        self.ds.blockSignals(True); self.ds.clear(); self.ds.addItems(list(pts.dataset.unique())); self.ds.blockSignals(False)
        self._apply_filter()

    def _context(self, ds):
        if ds not in self._ctx:
            d = self._dirs.get(ds)
            try: self._ctx[ds] = DatasetContext(d) if d else None
            except Exception: self._ctx[ds] = None   # noqa: BLE001
        return self._ctx[ds]

    def _apply_filter(self):
        if self._pts is None: return
        ds = self.ds.currentText(); gs = GROUP_FILTERS[self.grp.currentText()]
        T = self._T[(self._T.dataset == ds) & (self._T.n_points >= self.minlen.value())]
        if gs is not None: T = T[T.group.isin([GROUP_LABELS.get(g, g) for g in gs])]
        self.model.set_df(T.sort_values("n_points", ascending=False))
        P = self._pts[self._pts.track_uid.isin(set(T.track_uid))]
        self.annot.set_dataset(self._context(ds), self._pts_ds(ds)); self.lcheck.set_dataset(self._context(ds), self._pts_ds(ds))
        self.map.set_context(self._context(ds)); self.map.set_tracks(P[["track_uid", "frame", "x_fl", "y_fl", "x_ht", "y_ht", "group"]])
        self.info.setText(f"트랙 {len(T)}개 · 표나 영상에서 트랙을 고르면 아래에 밝기 그래프와 크롭이 표시됩니다")

    def _pts_ds(self, ds):
        if getattr(self, "_pts_ds_key", None) != ds: self._pts_ds_key = ds; self._pts_ds_val = self._pts[self._pts.dataset == ds]
        return self._pts_ds_val

    def _selected_track(self):
        uid = getattr(self, "_uid", None)
        if uid is None or self._pts is None: return None
        return uid, self._pts[self._pts.track_uid == uid]

    def select_uid(self, uid):
        df = self.model.df()
        if not len(df): return
        hit = np.nonzero(df.track_uid.values == uid)[0]
        if len(hit): self.table.selectRow(int(hit[0])); self.table.scrollTo(self.model.index(int(hit[0]), 0))

    def _on_select(self):
        rows = self.table.selectionModel().selectedRows()
        if not rows: return
        uid = self.model.df().track_uid.iloc[rows[0].row()]; self._uid = uid; t = self._pts[self._pts.track_uid == uid]
        self.map.select(uid, frame=int(t.frame.min()) if self.map.frame() not in set(t.frame) else None)
        self._draw_graph(); cfg = self._get_config(); ds = t.dataset.iloc[0]
        self.crop.set_track(self._context(ds), t, None, cfg.output.crop_half, cfg.channel.frame_interval_min, self._title(t), cfg.output.movie_fps)

    def _title(self, t):
        g = t.group.iloc[0]; return f"{t.dataset.iloc[0]} | track {t.track_id.iloc[0]} | {GROUP_LABELS.get(g, g)} | {len(t)} points"

    def _draw_graph(self):
        uid = getattr(self, "_uid", None)
        if uid is None or self._pts is None: return
        t = self._pts[self._pts.track_uid == uid]
        if not len(t): return
        draw_track_graph(self.gfig, t, None, self.map.frame(), self._get_config().channel.frame_interval_min, self._title(t) + " · 붉은 배경 = 세포 안")
        self.gcanvas.draw_idle()


# ====================================================================== 정답 표시 (ground truth)
class AnnotationPanel(QWidget):
    """트랙 하나를 시작점으로 프레임마다 '이 입자가 맞는지'를 확인해 정답 궤적을 만듭니다.

    · 현재 프레임 Cy5 크롭을 클릭 → 그 위치(가까운 검출에 붙음)를 정답으로 하고 다음 프레임으로
    · Enter = 표시된 위치가 맞음, M = 합쳐짐/가림, X = 안 보임, U = 모름, E = 여기서 끝, ←/→ = 이동
    · pHrodo는 보여주지 않습니다 (분석 결과값을 보고 판단하지 않도록).
    저장: <결과 폴더>/ground_truth.csv (바뀔 때마다 자동 저장)
    """
    HALF = 40

    def __init__(self, parent=None):
        super().__init__(parent)
        from PySide6.QtGui import QShortcut, QKeySequence
        self._ctx = None; self._src = None; self._pts = None; self._gt = None; self._gid = None; self._f = 1; self._T = 1; self._eval = None
        v = QVBoxLayout(self); row = QHBoxLayout()
        self.sel = QComboBox(); self.sel.currentIndexChanged.connect(self._on_pick)
        b_new = QPushButton("선택한 트랙으로 새 정답"); b_new.clicked.connect(self._new)
        b_del = QPushButton("이 정답 삭제"); b_del.clicked.connect(self._delete)
        b_ev = QPushButton("현재 결과 평가"); b_ev.clicked.connect(self._evaluate)
        for w in (QLabel("정답 궤적"), self.sel, b_new, b_del, b_ev): row.addWidget(w)
        row.setStretch(1, 1); v.addLayout(row)
        row = QHBoxLayout(); self.btns = {}
        for key, text, fn in [("prev", "◀ 이전 (←)", lambda: self._go(-1)), ("ok", "✔ 맞음 (Enter)", lambda: self._mark("ok")),
                              ("merged", "합쳐짐/가림 (M)", lambda: self._mark("merged")), ("absent", "안 보임 (X)", lambda: self._mark("absent")),
                              ("unsure", "모름 (U)", lambda: self._mark("unsure")), ("end", "여기서 끝 (E)", self._end), ("next", "다음 ▶ (→)", lambda: self._go(1))]:
            b = QPushButton(text); b.clicked.connect(fn); row.addWidget(b); self.btns[key] = b
        v.addLayout(row)
        self.lbl = QLabel("[트랙 탐색] 표에서 트랙을 고른 뒤 [선택한 트랙으로 새 정답]을 누르세요."); self.lbl.setWordWrap(True); v.addWidget(self.lbl)
        self.fig = Figure(figsize=(9, 3.6)); self.canvas = FigureCanvasQTAgg(self.fig); self.canvas.mpl_connect("button_press_event", self._on_click)
        v.addWidget(self.canvas, 1)
        for keyseq, fn in [("Return", lambda: self._mark("ok")), ("Enter", lambda: self._mark("ok")), ("M", lambda: self._mark("merged")),
                           ("X", lambda: self._mark("absent")), ("U", lambda: self._mark("unsure")), ("E", self._end),
                           ("Left", lambda: self._go(-1)), ("Right", lambda: self._go(1))]:
            sc = QShortcut(QKeySequence(keyseq), self); sc.setContext(Qt.WidgetWithChildrenShortcut); sc.activated.connect(fn)
        self.get_selected = lambda: None          # TrackExplorer가 (track_uid, 트랙 point 표)를 돌려주는 함수로 바꿈

    # ------------------------------------------------------------ 데이터
    def set_dataset(self, ctx: DatasetContext | None, pts: pd.DataFrame | None):
        """pts: 이 데이터셋의 모든 검출 point (frame, roi_label, x_fl, y_fl, track_uid ...)."""
        from ..gt import load_gt
        if ctx is not None and self._ctx is not None and ctx.folder == self._ctx.folder and pts is self._pts: return
        self._ctx, self._pts, self._gid = ctx, pts, None; self._src = None
        if ctx is None: self._gt = None; self.lbl.setText("원본 데이터셋 위치 정보가 없어 정답을 만들 수 없습니다"); self._refresh_list(); self._draw(); return
        try: self._src = PreviewSource(ctx.spec, max_frames=6); self._T = self._src.T
        except DataUnavailable as e: self.lbl.setText(str(e))
        self._gt = load_gt(ctx.folder); self._refresh_list(); self._draw()

    def _save(self):
        from ..gt import save_gt
        if self._ctx is not None and self._gt is not None: save_gt(self._ctx.folder, self._gt)

    def _refresh_list(self, keep=None):
        self.sel.blockSignals(True); self.sel.clear()
        if self._gt is not None:
            for gid, s in self._gt.groupby("gt_id"):
                done = int((s.status != "auto").sum())
                self.sel.addItem(f"#{int(gid)} · {s.source_track.iloc[0]} · {len(s)}프레임 (확인 {done})", int(gid))
        i = self.sel.findData(keep) if keep is not None else -1
        if i >= 0: self.sel.setCurrentIndex(i)
        self.sel.blockSignals(False)
        if self.sel.count() and self._gid is None: self._on_pick()

    def _on_pick(self, *_):
        gid = self.sel.currentData()
        if gid is None: return
        self._gid = int(gid); s = self._gt[self._gt.gt_id == self._gid]
        un = s[s.status == "auto"]; self._f = int(un.frame.min() if len(un) else s.frame.max()); self._draw()

    def _new(self):
        from ..gt import new_from_track
        sel = self.get_selected()
        if sel is None or self._gt is None: self.lbl.setText("먼저 [트랙 탐색] 표에서 트랙을 고르세요"); return
        uid, t = sel
        self._gt, gid = new_from_track(self._gt, t.sort_values("frame"), uid); self._gid = gid; self._f = int(t.frame.min())
        self._save(); self._refresh_list(keep=gid); self._draw()

    def _delete(self):
        if self._gid is None: return
        self._gt = self._gt[self._gt.gt_id != self._gid].reset_index(drop=True); self._gid = None; self._save(); self._refresh_list(); self._draw()

    # ------------------------------------------------------------ 표시
    def _row(self, f):
        s = self._gt[(self._gt.gt_id == self._gid) & (self._gt.frame == f)]; return s.iloc[0] if len(s) else None

    def _center(self, f):
        """이 프레임 정답 위치, 없으면 가장 가까운 프레임의 정답 위치 (확인된 점 우선)."""
        r = self._row(f)
        if r is not None and np.isfinite(r.x_fl): return r.x_fl, r.y_fl
        s = self._gt[(self._gt.gt_id == self._gid) & np.isfinite(self._gt.x_fl)]
        if not len(s): return None
        s = s.assign(d=(s.frame - f).abs() + (s.status == "auto") * 0.5); k = s.d.idxmin(); return s.x_fl[k], s.y_fl[k]

    def _draw(self):
        self.fig.clear(); en = self._gid is not None and self._src is not None
        for b in self.btns.values(): b.setEnabled(en)
        if not en: self.canvas.draw_idle(); return
        f = self._f; c = self._center(f); h = self.HALF
        if c is None: self.canvas.draw_idle(); return
        x0, y0 = c; S = self._src.S; hh = int(round(h * S)); axs = self.fig.subplots(1, 3); self._axs = axs; self._origin = (x0 - h, y0 - h)
        reg = self._ctx.reg; dy = dx = 0.0
        if reg is not None and len(reg) >= f: dy, dx = reg.dy_ht.values[f - 1], reg.dx_ht.values[f - 1]
        hx, hy = (x0 + .5) * S - .5 + dx, (y0 + .5) * S - .5 + dy
        panels = [(axs[0], "cy5", f - 1, x0, y0, h, f"이전 frame {f - 1}"), (axs[1], "cy5", f, x0, y0, h, f"frame {f} — 클릭해 위치 지정"),
                  (axs[2], "ht", f, hx, hy, hh, f"HT frame {f}")]
        for ax, ch, ff, cx, cy, hr, title in panels:
            ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=9)
            if ff < 1 or ff > self._T: ax.axis("off"); continue
            try: im = _crop(self._src.frame(ch, ff - 1), cx, cy, hr).astype(np.float32)
            except DataUnavailable as e: self.lbl.setText(str(e)); continue
            lo, hi = np.percentile(im, [1, 99.7]); ax.imshow(im, cmap="gray", vmin=lo, vmax=max(hi, lo + 1))
            sc = S if ch == "ht" else 1; ox, oy = cx - hr, cy - hr
            to = lambda X, Y: ((X + .5) * S - .5 + dx - ox, (Y + .5) * S - .5 + dy - oy) if ch == "ht" else (X - ox, Y - oy)
            if ch == "cy5" and self._pts is not None:          # 이 프레임의 모든 검출 (클릭하면 여기에 붙음)
                d = self._pts[(self._pts.frame == ff) & (np.abs(self._pts.x_fl - x0) < h + 5) & (np.abs(self._pts.y_fl - y0) < h + 5)]
                ax.plot(d.x_fl - ox, d.y_fl - oy, "o", ms=9, mfc="none", mec="cyan", mew=.7, ls="none")
            r = self._row(ff)
            if r is not None and np.isfinite(r.x_fl):
                px, py = to(r.x_fl, r.y_fl); col = {"ok": "lime", "auto": "yellow", "merged": "orange", "absent": "red", "unsure": "violet"}.get(r.status, "w")
                ax.plot(px, py, "+", ms=16, mew=2, color=col)
            ax.set_xlim(-.5, 2 * hr - .5); ax.set_ylim(2 * hr - .5, -.5)
        self.fig.tight_layout()
        s = self._gt[self._gt.gt_id == self._gid]; r = self._row(f); st = r.status if r is not None else "(없음)"
        from ..gt import STATUS_LABELS
        self.lbl.setText(f"정답 #{self._gid} · frame {f}/{self._T} · 이 프레임: {STATUS_LABELS.get(st, st)} · 확인 {int((s.status != 'auto').sum())}/{len(s)}  "
                         "— 노란 + = 트래커 위치(미확인), 초록 = 맞음, 주황 = 합쳐짐/가림, 하늘색 원 = 이 프레임의 검출 (클릭하면 그 검출로 지정)"
                         + (f"\n{self._eval}" if self._eval else ""))
        self.canvas.draw_idle()

    # ------------------------------------------------------------ 조작
    def _go(self, step):
        if self._gid is None: return
        self._f = int(np.clip(self._f + step, 1, self._T)); self._draw()

    def _mark(self, status):
        from ..gt import set_point
        if self._gid is None: return
        self._gt = set_point(self._gt, self._gid, self._f, status=status); self._save(); self._refresh_list(keep=self._gid); self._go(1)

    def _end(self):
        from ..gt import truncate
        if self._gid is None: return
        self._gt = truncate(self._gt, self._gid, self._f); self._save(); self._refresh_list(keep=self._gid); self._draw()

    def _on_click(self, ev):
        from ..gt import set_point
        if self._gid is None or not hasattr(self, "_axs") or ev.inaxes is not self._axs[1] or ev.xdata is None: return
        x, y = ev.xdata + self._origin[0], ev.ydata + self._origin[1]; lab = None
        d = self._pts[self._pts.frame == self._f] if self._pts is not None else pd.DataFrame()
        if len(d):
            dist, j = cKDTree(d[["x_fl", "y_fl"]].values).query([x, y])
            if dist <= 6: x, y, lab = d.x_fl.values[j], d.y_fl.values[j], int(d.roi_label.values[j])
        prev = self._row(self._f); changed = lab is not None and (prev is None or int(prev.roi_label) != lab)
        self._gt = set_point(self._gt, self._gid, self._f, x, y, lab, status="ok")
        if changed and "track_uid" in d:          # 다른 입자로 고쳤으면 이후 미확인 점을 그 입자의 트랙으로 다시 채움
            from ..gt import reseed_after
            uid = d.track_uid.values[j]; self._gt = reseed_after(self._gt, self._gid, self._f, self._pts[self._pts.track_uid == uid])
        self._save(); self._refresh_list(keep=self._gid); self._go(1)

    def _evaluate(self):
        from ..gt import evaluate, summary_text
        if self._ctx is None or self._gt is None: return
        p = self._ctx.folder / "tracks_points_A_B_HT.csv"
        if not p.exists(): self._eval = "평가할 트래킹 결과가 없습니다"; self._draw(); return
        pts = pd.read_csv(p); summ, T = evaluate(pts, self._gt)
        if len(T): T.to_csv(self._ctx.folder / "ground_truth_eval.csv", index=False)
        self._eval = "[평가] " + summary_text(summ); self._draw()


class LinkCheckPanel(QWidget):
    """트래커가 이은 연결(특히 누락 프레임을 건너뛴 연결)의 양 끝을 나란히 보여주고 '같은 입자인지'만 답합니다.
    Y = 같은 입자, N = 다른 입자, U = 모름, ←/→ = 이동. 결과: <결과 폴더>/link_checks.csv"""
    HALF = 32

    def __init__(self, parent=None):
        super().__init__(parent)
        from PySide6.QtGui import QShortcut, QKeySequence
        self._ctx = None; self._src = None; self._pts = None; self._C = None; self._i = 0
        v = QVBoxLayout(self); row = QHBoxLayout()
        b = QPushButton("연결 표본 만들기 (현재 결과)"); b.clicked.connect(self._make); row.addWidget(b)
        self.nper = QSpinBox(); self.nper.setRange(3, 100); self.nper.setValue(12); self.nper.setPrefix("종류별 "); self.nper.setSuffix("개"); row.addWidget(self.nper)
        for text, fn in [("◀ (←)", lambda: self._go(-1)), ("같은 입자 (Y)", lambda: self._answer("same")), ("다른 입자 (N)", lambda: self._answer("diff")),
                         ("모름 (U)", lambda: self._answer("unsure")), ("(→) ▶", lambda: self._go(1))]:
            bb = QPushButton(text); bb.clicked.connect(fn); row.addWidget(bb)
        row.addStretch(); v.addLayout(row)
        self.lbl = QLabel("[연결 표본 만들기]를 누르면 현재 트래킹 결과에서 연속 연결과 누락 프레임을 건너뛴 연결을 종류별로 뽑습니다.")
        self.lbl.setWordWrap(True); v.addWidget(self.lbl)
        self.fig = Figure(figsize=(9, 3.6)); self.canvas = FigureCanvasQTAgg(self.fig); v.addWidget(self.canvas, 1)
        for keyseq, fn in [("Y", lambda: self._answer("same")), ("N", lambda: self._answer("diff")), ("U", lambda: self._answer("unsure")),
                           ("Left", lambda: self._go(-1)), ("Right", lambda: self._go(1))]:
            sc = QShortcut(QKeySequence(keyseq), self); sc.setContext(Qt.WidgetWithChildrenShortcut); sc.activated.connect(fn)

    def set_dataset(self, ctx, pts):
        from ..gt import load_checks
        if ctx is not None and self._ctx is not None and ctx.folder == self._ctx.folder and pts is self._pts: return
        self._ctx, self._pts, self._src = ctx, pts, None
        if ctx is None: self._C = None; self._draw(); return
        try: self._src = PreviewSource(ctx.spec, max_frames=8)
        except DataUnavailable as e: self.lbl.setText(str(e))
        self._C = load_checks(ctx.folder); pend = np.nonzero(self._C.answer.values == "")[0] if len(self._C) else []
        self._i = int(pend[0]) if len(pend) else 0; self._draw()

    def _make(self):
        from ..gt import sample_links, save_checks, load_checks
        if self._ctx is None or self._pts is None: return
        old = load_checks(self._ctx.folder)
        if len(old) and (old.answer != "").any():
            from PySide6.QtWidgets import QMessageBox
            if QMessageBox.question(self, "연결 표본", "이미 답한 표본이 있습니다. 새 표본으로 바꾸면 기존 답은 link_checks_old.csv로 옮깁니다. 계속할까요?") != QMessageBox.Yes: return
            old.to_csv(self._ctx.folder / "link_checks_old.csv", index=False)
        self._C = sample_links(self._pts.rename(columns={}), self.nper.value()); save_checks(self._ctx.folder, self._C); self._i = 0; self._draw()

    def _go(self, s):
        if self._C is None or not len(self._C): return
        self._i = int(np.clip(self._i + s, 0, len(self._C) - 1)); self._draw()

    def _answer(self, a):
        from ..gt import save_checks, now
        if self._C is None or not len(self._C): return
        self._C.loc[self._i, ["answer", "updated"]] = [a, now()]; save_checks(self._ctx.folder, self._C); self._go(1)

    def _draw(self):
        from ..gt import checks_summary
        self.fig.clear()
        if self._C is None or not len(self._C) or self._src is None: self.canvas.draw_idle(); return
        r = self._C.iloc[self._i]; h = self.HALF; S = self._src.S; hh = int(round(h * S)); reg = self._ctx.reg
        axs = self.fig.subplots(1, 4)
        for k, (ax, ch, f, x, y) in enumerate([(axs[0], "cy5", r.frame_a, r.x_a, r.y_a), (axs[1], "cy5", r.frame_b, r.x_b, r.y_b),
                                                (axs[2], "ht", r.frame_a, r.x_a, r.y_a), (axs[3], "ht", r.frame_b, r.x_b, r.y_b)]):
            f = int(f); ax.set_xticks([]); ax.set_yticks([])
            if ch == "ht":
                dy = dx = 0.0
                if reg is not None and len(reg) >= f: dy, dx = reg.dy_ht.values[f - 1], reg.dx_ht.values[f - 1]
                x, y, hr = (x + .5) * S - .5 + dx, (y + .5) * S - .5 + dy, hh
            else: hr = h
            try: im = _crop(self._src.frame(ch, f - 1), x, y, hr).astype(np.float32)
            except DataUnavailable as e: self.lbl.setText(str(e)); return
            lo, hi = np.percentile(im, [1, 99.7]); ax.imshow(im, cmap="gray", vmin=lo, vmax=max(hi, lo + 1))
            ax.add_patch(Circle((hr - .5, hr - .5), 7 * (S if ch == "ht" else 1), fill=False, ec="lime", lw=1.2))
            ax.set_title(f"{'Cy5' if ch == 'cy5' else 'HT'} frame {f}" + (" (앞)" if k % 2 == 0 else " (뒤)"), fontsize=9)
        self.fig.tight_layout()
        ans = {"same": "같은 입자", "diff": "다른 입자", "unsure": "모름", "": "미답"}[r.answer]
        self.lbl.setText(f"표본 {self._i + 1}/{len(self._C)} · 프레임 {int(r.frame_a)} → {int(r.frame_b)} · 지금 답: {ans} — 초록 원 안의 두 입자가 같은 입자인가요? (Y/N/U)\n"
                         + checks_summary(self._C))
        self.canvas.draw_idle()

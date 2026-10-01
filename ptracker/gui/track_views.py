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
    T["group"] = T.group.map(lambda x: GROUP_LABELS.get(x, x)); return T


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
        self.bottom = QTabWidget(); self.bottom.addTab(self.gcanvas, "밝기 그래프"); self.bottom.addTab(self.crop, "크롭 프레임")
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
        self.map.set_context(self._context(ds)); self.map.set_tracks(P[["track_uid", "frame", "x_fl", "y_fl", "x_ht", "y_ht", "group"]])
        self.info.setText(f"트랙 {len(T)}개 · 표나 영상에서 트랙을 고르면 아래에 밝기 그래프와 크롭이 표시됩니다")

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

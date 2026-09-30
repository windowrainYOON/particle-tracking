"""파라미터 탭 오른쪽의 미리보기 패널.

파라미터를 바꾸면 잠시(0.4초) 기다렸다가 백그라운드 스레드에서 선택한 섹션의 미리보기를 계산하고,
직전 결과(이전)와 새 결과(현재)를 나란히 보여줍니다. 계산은 ptracker.preview (Qt 없음)에서 합니다.
"""
from __future__ import annotations
import time, traceback
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QSpinBox, QCheckBox, QPushButton,
                               QSizePolicy)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib import colormaps, colors as mcolors, patheffects
from ..config import Config
from ..preview import PreviewSource, DataUnavailable, run_preview, PREVIEW_SECTIONS

DEBOUNCE_MS = 400


class _Bridge(QObject):
    done = Signal(int, object)        # (작업 번호, 결과 dict 또는 예외 문자열)


class PreviewPanel(QWidget):
    def __init__(self, get_config, is_busy=lambda: False, parent=None):
        super().__init__(parent)
        self._get_config, self._is_busy = get_config, is_busy
        self._specs = []; self._sources = {}; self._centers = {}; self._section = ""
        self._seq = 0; self._cur = None; self._prev = None; self._dirty = False; self._t0 = 0.0
        self._pool = ThreadPoolExecutor(max_workers=1); self._bridge = _Bridge(); self._bridge.done.connect(self._on_done)
        self._timer = QTimer(self); self._timer.setSingleShot(True); self._timer.timeout.connect(self._submit)

        v = QVBoxLayout(self); v.setContentsMargins(4, 0, 0, 0)
        row = QHBoxLayout()
        self.ds = QComboBox(); self.ds.currentIndexChanged.connect(lambda _: self.request())
        self.fr = QSpinBox(); self.fr.setRange(1, 1); self.fr.setPrefix("frame "); self.fr.valueChanged.connect(lambda _: self.request())
        self.size = QComboBox(); self.size.addItems(["128", "256", "512"]); self.size.setCurrentText("256"); self.size.currentTextChanged.connect(lambda _: self.request())
        self.auto = QCheckBox("자동 갱신"); self.auto.setChecked(True); self.auto.setToolTip("파라미터를 바꾸면 자동으로 다시 계산")
        b = QPushButton("새로고침"); b.clicked.connect(lambda: self.request(manual=True))
        row.addWidget(QLabel("미리보기")); row.addWidget(self.ds, 1); row.addWidget(self.fr); row.addWidget(QLabel("영역(px)")); row.addWidget(self.size)
        row.addWidget(self.auto); row.addWidget(b); v.addLayout(row)

        row = QHBoxLayout()
        self.ov_fig = Figure(figsize=(2.2, 2.2)); self.ov_fig.subplots_adjust(0, 0, 1, 1); self.ov_ax = self.ov_fig.add_subplot(111)
        self.ov_canvas = FigureCanvasQTAgg(self.ov_fig); self.ov_canvas.setFixedSize(210, 210); self.ov_canvas.mpl_connect("button_press_event", self._on_click)
        self.ov_canvas.setToolTip("클릭한 위치를 중심으로 영역을 잘라 미리봅니다")
        row.addWidget(self.ov_canvas)
        rv = QVBoxLayout(); self.status = QLabel(""); self.status.setStyleSheet("color: gray")
        self.info = QLabel("[1. 데이터·실행]에서 데이터셋을 추가하면 여기에서 파라미터 효과를 미리 볼 수 있습니다.")
        self.info.setWordWrap(True); self.info.setAlignment(Qt.AlignTop | Qt.AlignLeft); self.info.setTextInteractionFlags(Qt.TextSelectableByMouse)
        rv.addWidget(self.status); rv.addWidget(self.info, 1); row.addLayout(rv, 1); v.addLayout(row)

        self.fig = Figure(figsize=(8, 4)); self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding); v.addWidget(self.canvas, 1)
        self._draw()

    # ------------------------------------------------------------ 외부에서 호출
    def set_specs(self, specs):
        cur = self._spec()
        self._specs = list(specs); self.ds.blockSignals(True); self.ds.clear()
        for s in self._specs: self.ds.addItem(s.name, s.cy5)
        if cur is not None:
            i = self.ds.findData(cur.cy5)
            if i >= 0: self.ds.setCurrentIndex(i)
        self.ds.blockSignals(False)
        self.request()

    def set_section(self, sec):
        if sec != self._section: self._section = sec; self._prev = None; self.request(manual=True)

    def request(self, manual=False):
        """파라미터·데이터셋·프레임이 바뀌었을 때 호출. 잠시 기다렸다가 한 번만 계산합니다."""
        if not manual and not self.auto.isChecked(): return
        if not self.isVisible(): self._dirty = True; return
        if not manual and self._is_busy():
            self.status.setText("분석 실행 중에는 자동 갱신을 멈춥니다 — [새로고침]으로 직접 갱신할 수 있습니다"); return
        self._timer.start(0 if manual else DEBOUNCE_MS)

    def reset_sources(self):
        """분석 실행 뒤 캐시가 바뀌었을 수 있으므로 다음 미리보기에서 데이터셋을 다시 엽니다."""
        self._pool.submit(self._sources.clear); self.request()

    def shutdown(self):
        self._seq += 1; self._pool.shutdown(wait=False, cancel_futures=True); self._sources.clear()

    def showEvent(self, e):
        super().showEvent(e)
        if self._dirty: self._dirty = False; self.request(manual=True)

    # ------------------------------------------------------------ 계산
    @staticmethod
    def _key(s): return (s.cy5, s.phrodo, s.ht, s.out_dir)

    def _spec(self):
        i = self.ds.currentIndex(); return self._specs[i] if 0 <= i < len(self._specs) else None

    def _submit(self):
        spec = self._spec()
        if spec is None: self._cur = self._prev = None; self._draw(); return
        try: cfg = self._get_config()
        except Exception as e: self.status.setText(f"파라미터 오류: {e}"); return   # noqa: BLE001
        self._seq += 1; seq = self._seq; self._t0 = time.time()
        sec = self._section; z = self.fr.value() - 1; size = int(self.size.currentText()); center = self._centers.get(spec.cy5)
        self.status.setText(f"계산 중… ({PREVIEW_SECTIONS.get(sec, sec)})")
        self._pool.submit(self._work, seq, spec, sec, z, center, size, cfg)

    def _work(self, seq, spec, sec, z, center, size, cfg: Config):
        if seq != self._seq: return                      # 더 새로운 요청이 있으면 건너뜀
        try:
            k = self._key(spec)
            if k not in self._sources:
                self._sources.clear()                    # 한 번에 한 데이터셋만 메모리에 유지
                self._sources[k] = PreviewSource(spec)
            src = self._sources[k]; r = run_preview(src, sec, z, center, size, cfg); r["T"] = src.T; r["fl_shape"] = (src.NF, src.NW)
            self._bridge.done.emit(seq, r)
        except DataUnavailable as e:
            self._bridge.done.emit(seq, str(e))
        except Exception:   # noqa: BLE001
            self._bridge.done.emit(seq, "미리보기 오류:\n" + traceback.format_exc(limit=3))

    def _on_done(self, seq, r):
        if seq != self._seq: return
        if isinstance(r, str): self.status.setText(""); self.info.setText(r); return
        self.status.setText(f"{time.time() - self._t0:.1f}초")
        if self.fr.maximum() != r["T"]:           # 데이터셋을 처음 열면 입자가 충분한 중간 프레임으로 이동
            self.fr.blockSignals(True); self.fr.setMaximum(r["T"]); self.fr.setValue(r["T"] // 2 + 1); self.fr.blockSignals(False)
            self._cur = None; self.request(manual=True); return
        same = self._cur is not None and self._cur.get("section") == r["section"] and self._cur.get("key") == r["key"]
        if same and self._cur.get("params") != r.get("params"): self._prev = self._cur
        elif not same: self._prev = None
        self._cur = r; self.info.setText(r["info"]); self._draw()

    def _on_click(self, ev):
        if ev.inaxes is not self.ov_ax or self._cur is None or ev.xdata is None: return
        spec = self._spec(); k = self._cur["overview"].shape[0] / self._cur["fl_shape"][0]
        self._centers[spec.cy5] = (ev.ydata / k, ev.xdata / k); self._prev = None; self.request(manual=True)

    # ------------------------------------------------------------ 그리기
    def _draw(self):
        self.ov_ax.clear(); self.ov_ax.axis("off")
        r = self._cur
        if r is not None and "overview" in r:
            self.ov_ax.imshow(r["overview"], cmap="gray"); y0, y1, x0, x1 = r["crop_box"]
            self.ov_ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="yellow", lw=1.2))
        self.ov_canvas.draw_idle()
        self.fig.clear(); axs = self.fig.subplots(1, 2)
        diff = _diff(self._prev, r)
        _draw_result(axs[0], self._prev, "이전" + (f"  ({diff[0]})" if diff[0] else ""), "파라미터를 바꾸면 직전 결과가 여기에 표시됩니다")
        _draw_result(axs[1], r, "현재" + (f"  ({diff[1]})" if diff[1] else ""), "미리볼 데이터셋이 없습니다" if not self._specs else "")
        self.fig.tight_layout(); self.canvas.draw_idle()


def _diff(prev, cur):
    if not prev or not cur: return "", ""
    p, c = prev.get("params", {}), cur.get("params", {}); ks = [k for k in c if p.get(k) != c.get(k)]
    fmt = lambda d: ", ".join(f"{k.split('.', 1)[1]}={d.get(k)}" for k in ks[:3]) + (" …" if len(ks) > 3 else "")
    return fmt(p), fmt(c)


def _rgba(mask, color, alpha):
    out = np.zeros((*mask.shape, 4), np.float32); out[mask] = (*mcolors.to_rgb(color), alpha); return out


def _draw_result(ax, r, title, empty_msg):
    ax.set_title(title, fontsize=9)
    if r is None or r.get("kind") == "message":
        ax.axis("off"); ax.text(.5, .5, (r or {}).get("info", empty_msg), ha="center", va="center", fontsize=9, wrap=True, transform=ax.transAxes)
        return
    if r["kind"] == "plot":
        for s in r["series"]:
            ln, = ax.plot(s["frame"], s["ri"], lw=1, alpha=.7, label=f"cell {s['cell']}")
            m = s["mit"].astype(bool)
            if m.any(): ax.plot(s["frame"][m], s["ri"][m], "o", color="red", ms=5)
        ax.set_xlabel("frame", fontsize=8); ax.set_ylabel("세포 평균 RI", fontsize=8); ax.tick_params(labelsize=7); ax.grid(alpha=.3)
        if len(r["series"]) <= 20: ax.legend(fontsize=6, ncol=2, loc="upper left")
        return
    im = r["image"]; ax.imshow(im, cmap=None if im.ndim == 3 else "gray", interpolation="nearest"); ax.axis("off")
    for o in r["overlays"]:
        t = o["type"]
        if t == "contour": ax.imshow(_rgba(o["mask"], o["color"], 1.0), interpolation="nearest")
        elif t == "fill": ax.imshow(_rgba(o["mask"], o["color"], o.get("alpha", .4)), interpolation="nearest")
        elif t == "labels":
            L = o["labels"]; cm = colormaps["tab20"]; rgba = cm((L % 20) / 19.0); rgba[..., 3] = np.where(L > 0, .25, 0)
            ax.imshow(rgba, interpolation="nearest")
        elif t == "points": ax.plot(o["x"], o["y"], o.get("marker", "+"), color=o["color"], ms=5, mew=.8, ls="none")
        elif t == "text":
            H, W = r["image"].shape[:2]
            for y, x, s in zip(o["y"], o["x"], o["text"]):
                if 0 <= y < H and 0 <= x < W:
                    ax.text(x, y, s, color=o["color"], fontsize=7, ha="center", va="center",
                            path_effects=[patheffects.withStroke(linewidth=2, foreground="black")])

"""raw TCF 파일의 채널을 역할(measure / reference / RI)에 할당하는 창."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QComboBox, QDialogButtonBox, QCheckBox,
                               QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from ..io_utils import ROLES, ROLE_LABELS, ROLE_HELP, RI_CHANNEL, tcf_channels, read_frame, make_source

DEFAULT_ASSIGN = {"measure": "CH0", "reference": "CH1", "ri": RI_CHANNEL}


def default_assign(names, prefer=None):
    """있는 채널 이름에 맞춰 기본 할당 (prefer → CH0/CH1/RI → 남은 형광 순)."""
    fl = [n for n in names if n != RI_CHANNEL]; out = {}
    for role in ROLES:
        for cand in ((prefer or {}).get(role), DEFAULT_ASSIGN[role]):
            if cand in names and cand not in out.values(): out[role] = cand; break
        else:
            pool = [RI_CHANNEL] if role == "ri" and RI_CHANNEL in names else [n for n in fl if n not in out.values()] or list(names)
            out[role] = pool[0] if pool else ""
    return out


class ChannelAssignDialog(QDialog):
    """files: TCF 경로들 (첫 파일의 채널을 보여주고, 같은 할당을 모든 파일에 적용)."""

    def __init__(self, files, prefer=None, frame_interval=None, parent=None):
        super().__init__(parent)
        self.files = [str(f) for f in files]; first = self.files[0]
        self.setWindowTitle("TCF 채널 할당"); self.resize(980, 640)
        self.channels = tcf_channels(first); names = [c["name"] for c in self.channels]
        v = QVBoxLayout(self)
        more = f" 외 {len(self.files) - 1}개 (같은 할당 적용)" if len(self.files) > 1 else ""
        v.addWidget(QLabel(f"<b>{Path(first).name}</b>{more}<br>파일에서 찾은 채널을 각 역할에 할당하세요. "
                           "아래 미리보기(첫 프레임)를 보고 어떤 형광인지 확인할 수 있습니다."))
        t = QTableWidget(len(self.channels), 5); t.setHorizontalHeaderLabels(["채널", "종류", "프레임 수", "크기 (px)", "프레임 간격"])
        t.verticalHeader().hide(); t.setEditTriggers(QAbstractItemView.NoEditTriggers); t.setSelectionMode(QAbstractItemView.NoSelection)
        for r, c in enumerate(self.channels):
            vals = [c["name"], c["label"], str(c["frames"]), f"{c['size'][1]} × {c['size'][0]}",
                    f"{c['interval_min']:.1f} 분" if c["interval_min"] else "-"]
            for k, val in enumerate(vals): t.setItem(r, k, QTableWidgetItem(val))
        t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents); t.horizontalHeader().setStretchLastSection(True)
        t.setMaximumHeight(36 + 30 * len(self.channels)); v.addWidget(t)

        self.fig = Figure(figsize=(2.6 * max(len(self.channels), 1), 2.6)); self.canvas = FigureCanvasQTAgg(self.fig); v.addWidget(self.canvas, 1)
        self._thumbs(first)

        g = QGridLayout(); self.combo = {}; init = default_assign(names, prefer)
        for i, role in enumerate(ROLES):
            cb = QComboBox(); cb.addItems(names); cb.setCurrentText(init[role]); cb.currentTextChanged.connect(self._check)
            g.addWidget(QLabel(f"<b>{ROLE_LABELS[role]}</b>"), i, 0); g.addWidget(cb, i, 1); g.addWidget(QLabel(ROLE_HELP[role]), i, 2)
            self.combo[role] = cb
        g.setColumnStretch(2, 1); v.addLayout(g)

        dts = [c["interval_min"] for c in self.channels if c["interval_min"]]
        self.tcf_interval = float(np.median(dts)) if dts else None
        self.use_interval = QCheckBox()
        if self.tcf_interval and frame_interval is not None and abs(self.tcf_interval - frame_interval) > 1e-6:
            self.use_interval.setText(f"프레임 간격을 TCF 촬영 간격 {self.tcf_interval:.1f}분으로 설정 (현재 파라미터 {frame_interval:g}분)")
            self.use_interval.setChecked(True); v.addWidget(self.use_interval)
        else: self.use_interval.hide()

        self.warn = QLabel(""); self.warn.setStyleSheet("color: #c0392b"); v.addWidget(self.warn)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel); bb.accepted.connect(self.accept); bb.rejected.connect(self.reject)
        self.ok = bb.button(QDialogButtonBox.Ok); v.addWidget(bb); self._check()

    def _thumbs(self, path):
        self.fig.clear(); n = len(self.channels)
        for i, c in enumerate(self.channels):
            ax = self.fig.add_subplot(1, n, i + 1); ax.set_xticks([]); ax.set_yticks([])
            try:
                im = read_frame(make_source(path, c["name"]), 0).astype(np.float32)
                lo, hi = np.percentile(im[::2, ::2], (0.5, 99.8)); ax.imshow(im, cmap="gray", vmin=lo, vmax=max(hi, lo + 1))
            except Exception as e:   # noqa: BLE001
                ax.text(.5, .5, f"읽기 실패\n{e}", ha="center", va="center", fontsize=7, transform=ax.transAxes)
            ax.set_title(c["name"], fontsize=10)
        self.fig.tight_layout(); self.canvas.draw_idle()

    def _check(self):
        a = self.assignment(); msg = ""
        if a["measure"] == a["reference"]: msg = "measure와 reference에 같은 채널을 할당했습니다. 서로 다른 형광 채널을 고르세요."
        elif a["ri"] != RI_CHANNEL: msg = "RI 역할에 형광 채널을 할당했습니다. 보통 RI 채널을 씁니다."
        elif RI_CHANNEL in (a["measure"], a["reference"]): msg = "형광 역할에 RI 채널을 할당했습니다."
        self.warn.setText(msg); self.ok.setEnabled(a["measure"] != a["reference"])

    def assignment(self) -> dict:
        return {r: cb.currentText() for r, cb in self.combo.items()}

    def frame_interval(self):
        """체크했으면 TCF 촬영 간격(분), 아니면 None."""
        return self.tcf_interval if not self.use_interval.isHidden() and self.use_interval.isChecked() else None

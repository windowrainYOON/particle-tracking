"""config.py의 dataclass 정의로부터 자동 생성되는 파라미터 편집기."""
from __future__ import annotations
from dataclasses import fields
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QFormLayout, QToolBox, QCheckBox, QSpinBox, QDoubleSpinBox,
                               QLineEdit, QComboBox, QScrollArea, QLabel)
from ..config import Config, sections


class ParamForm(QWidget):
    changed = Signal(str)             # 값이 바뀐 섹션 이름
    sectionChanged = Signal(str)      # 펼친 섹션 이름

    def __init__(self, cfg: Config | None = None, parent=None, only=None):
        super().__init__(parent)
        self._widgets = {}            # (section, name) -> widget
        self._sections = []
        lay = QVBoxLayout(self); self.box = QToolBox(); lay.addWidget(self.box)
        cfg = cfg or Config()
        for sec, label, obj in sections(cfg):
            if only and sec not in only: continue
            page = QWidget(); form = QFormLayout(page)
            for f in fields(obj):
                w = self._make_widget(f, getattr(obj, f.name))
                lab = QLabel(f.metadata.get("label", f.name)); tip = f.metadata.get("help", "")
                if tip: lab.setToolTip(tip); w.setToolTip(tip)
                form.addRow(lab, w); self._widgets[(sec, f.name)] = w; self._connect(w, sec)
            sc = QScrollArea(); sc.setWidgetResizable(True); sc.setWidget(page)
            self.box.addItem(sc, label); self._sections.append(sec)
        self.box.currentChanged.connect(lambda i: self.sectionChanged.emit(self.current_section()))

    def _connect(self, w, sec):
        emit = lambda *_: self.changed.emit(sec)
        if isinstance(w, QCheckBox): w.toggled.connect(emit)
        elif isinstance(w, (QSpinBox, QDoubleSpinBox)): w.valueChanged.connect(emit)
        elif isinstance(w, QComboBox): w.currentTextChanged.connect(emit)
        else: w.editingFinished.connect(emit)

    def current_section(self) -> str:
        i = self.box.currentIndex(); return self._sections[i] if 0 <= i < len(self._sections) else ""

    @staticmethod
    def _make_widget(f, value):
        md = f.metadata
        if isinstance(value, bool):
            w = QCheckBox(); w.setChecked(value); return w
        if isinstance(value, int):
            w = QSpinBox(); w.setRange(int(md.get("min", -10**9)), int(md.get("max", 10**9))); w.setValue(value); return w
        if isinstance(value, float):
            w = QDoubleSpinBox(); w.setDecimals(int(md.get("decimals", 3))); w.setRange(float(md.get("min", -1e12)), float(md.get("max", 1e12)))
            w.setSingleStep(float(md.get("step", 0.1))); w.setValue(value); return w
        if "choices" in md:
            w = QComboBox(); w.addItems(md["choices"]); w.setCurrentText(str(value)); return w
        w = QLineEdit(str(value)); return w

    def set_config(self, cfg: Config):
        for (sec, name), w in self._widgets.items():
            v = getattr(getattr(cfg, sec), name)
            if isinstance(w, QCheckBox): w.setChecked(bool(v))
            elif isinstance(w, (QSpinBox, QDoubleSpinBox)): w.setValue(v)
            elif isinstance(w, QComboBox): w.setCurrentText(str(v))
            else: w.setText(str(v))

    def apply_to(self, cfg: Config) -> Config:
        for (sec, name), w in self._widgets.items():
            obj = getattr(cfg, sec)
            if isinstance(w, QCheckBox): v = w.isChecked()
            elif isinstance(w, (QSpinBox, QDoubleSpinBox)): v = w.value()
            elif isinstance(w, QComboBox): v = w.currentText()
            else: v = w.text()
            setattr(obj, name, type(getattr(obj, name))(v))
        return cfg

"""[그룹 비교] 탭: 여러 이미지 결과를 그룹으로 묶어 시간 추이·분포를 비교하고, 그래프와 값을 저장합니다.

왼쪽: 그룹 목록(그룹마다 결과 폴더·통합 결과·CSV를 import)과 비교 설정. 설정을 바꾸면 그래프가 바로 다시 그려집니다.
오른쪽: 그래프(확대·이동 도구 포함)와 표(시간별 요약 / 단위별 평균 / 통계 검정).
"""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import numpy as np, pandas as pd
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor, QBrush
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QPushButton, QLabel, QTreeWidget, QTreeWidgetItem,
                               QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit, QSplitter, QTabWidget, QTableView, QFileDialog,
                               QMessageBox, QColorDialog, QAbstractItemView, QApplication, QGroupBox, QCheckBox, QScrollArea)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

from .. import groups as G
from .table_model import DataFrameModel

ROLE_KIND = Qt.UserRole + 1          # "group" | "source"
PLOTS = {"time": "시간 추이 (한 그래프에 그룹별 선)", "dist": "분포 (단위별 평균, 상자그림 + 검정)", "both": "시간 추이 + 분포"}
INDIV = {"none": "없음", "image": "이미지별 선", "unit": "트랙·세포별 선"}


def _combo(d: dict, cur=None):
    c = QComboBox()
    for k, v in d.items(): c.addItem(v, k)
    if cur is not None: c.setCurrentIndex(max(c.findData(cur), 0))
    return c


class GroupCompare(QWidget):
    log = Signal(str)

    def __init__(self, dataset_dirs=None, parent=None):
        """dataset_dirs: () → [(이름, 결과 폴더)] — [데이터·실행] 표에서 선택한(없으면 전체) 데이터셋."""
        super().__init__(parent)
        self._dataset_dirs = dataset_dirs or (lambda: [])
        self._P = self._C = None; self._dirty_data = True; self._res = {}
        self._timer = QTimer(self); self._timer.setSingleShot(True); self._timer.timeout.connect(self.refresh)
        sp = QSplitter(); lay = QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0); lay.addWidget(sp)

        # ------------------------------------------------ 왼쪽: 그룹 + 설정
        left = QWidget(); lv = QVBoxLayout(left)
        lv.addWidget(QLabel("그룹 (체크한 그룹만 비교 · 이름은 더블클릭해서 바꿈)"))
        self.tree = QTreeWidget(); self.tree.setHeaderLabels(["그룹 / 결과", "내용"]); self.tree.setColumnWidth(0, 260)
        self.tree.setSelectionMode(QAbstractItemView.ExtendedSelection); self.tree.itemChanged.connect(self._item_changed)
        lv.addWidget(self.tree, 2)
        r1 = QHBoxLayout(); r2 = QHBoxLayout()
        for row, items in [(r1, [("+ 그룹", self.add_group), ("결과 폴더 추가…", self.add_folder), ("CSV 추가…", self.add_csv), ("데이터 표에서 추가", self.add_from_table)]),
                           (r2, [("색…", self.set_color), ("선택 삭제", self.remove_selected), ("구성 저장…", self.save_setup_dialog), ("구성 불러오기…", self.load_setup_dialog)])]:
            for text, fn in items:
                b = QPushButton(text); b.clicked.connect(fn); row.addWidget(b)
            lv.addLayout(row)
        tip = QLabel("결과 폴더: 데이터셋 결과 폴더, 통합 결과 폴더, 또는 여러 데이터셋 결과 폴더를 담은 상위 폴더")
        tip.setWordWrap(True); tip.setStyleSheet("color: gray"); lv.addWidget(tip)

        gb = QGroupBox("비교 설정 (바꾸면 그래프가 바로 갱신됩니다)"); f = QFormLayout(gb); s = G.CompareSettings()
        self.w_level = _combo(G.LEVELS); self.w_metric = QComboBox(); self.w_loc = _combo(G.LOCATIONS, s.location)
        self.w_minpts = QSpinBox(); self.w_minpts.setRange(1, 10000); self.w_minpts.setValue(s.min_points)
        self.w_bin = QDoubleSpinBox(); self.w_bin.setRange(0, 1000); self.w_bin.setDecimals(2); self.w_bin.setSingleStep(.5); self.w_bin.setSpecialValueText("묶지 않음 (촬영 시점)")
        self.w_t0 = QDoubleSpinBox(); self.w_t0.setRange(0, 10000); self.w_t0.setDecimals(2); self.w_t0.setSpecialValueText("처음부터")
        self.w_t1 = QDoubleSpinBox(); self.w_t1.setRange(0, 10000); self.w_t1.setDecimals(2); self.w_t1.setSpecialValueText("끝까지")
        self.w_norm = _combo(G.NORMS); self.w_normn = QSpinBox(); self.w_normn.setRange(1, 100); self.w_normn.setValue(s.norm_n)
        self.w_center = _combo(G.CENTERS); self.w_spread = _combo(G.SPREADS); self.w_agg = _combo(G.AGGS)
        self.w_minn = QSpinBox(); self.w_minn.setRange(1, 10000); self.w_minn.setValue(s.min_n)
        self.w_plot = _combo(PLOTS); self.w_indiv = _combo(INDIV)
        self.w_filter = QLineEdit(); self.w_filter.setPlaceholderText("예: P01 · cell 3, P02 · cell 1  (비우면 전부)")
        self.w_band = QCheckBox("범위 띠 표시"); self.w_band.setChecked(True)
        tips = {self.w_level: "입자 = 트랙 하나가 선 하나. 세포(세포 내 입자) = 세포마다 그 안 입자 값을 모음. 세포(세포 전체) = 세포 영역 전체의 평균 형광 (cell_intensity.csv 필요)",
                self.w_loc: "입자 단위에서 어느 트랙을 쓸지. '안·밖 따로'는 그룹마다 세포 안(실선)·밖(점선) 두 선",
                self.w_minpts: "트랙·세포마다 값이 있는 시점(시간 칸) 수가 이보다 적으면 제외",
                self.w_bin: "촬영 간격이 다른 그룹을 함께 볼 때 시간을 이 폭(h)으로 묶습니다",
                self.w_norm: "트랙·세포마다 처음 n개 시점 평균으로 나누거나(배수) 뺍니다(변화량)",
                self.w_agg: "이미지별: 이미지마다 중심값을 먼저 구하고 이미지들 사이에서 평균·범위 (이미지 = 반복 실험)",
                self.w_minn: "한 시점의 트랙·세포 수가 이보다 적으면 그 점은 그리지 않음",
                self.w_filter: "쉼표로 구분. 단위 이름과 똑같으면 그 단위만, 일부 글자면 그 글자가 들어 있는 단위 모두. [단위별 평균] 표에서 행을 더블클릭하면 추가됩니다"}
        for w_, t in tips.items(): w_.setToolTip(t)
        for w_ in (self.w_bin, self.w_t0, self.w_t1): w_.setMinimumWidth(170)
        for lab, w_ in [("단위", self.w_level), ("값", self.w_metric), ("입자 위치", self.w_loc), ("최소 시점 수", self.w_minpts), ("시간 칸 (h)", self.w_bin),
                        ("시작 (h)", self.w_t0), ("끝 (h)", self.w_t1), ("정규화", self.w_norm), ("처음 n점", self.w_normn), ("중심값", self.w_center),
                        ("범위", self.w_spread), ("집계", self.w_agg), ("시점별 최소 n", self.w_minn), ("그래프", self.w_plot), ("개별 선", self.w_indiv),
                        ("단위 선택", self.w_filter), ("", self.w_band)]:
            f.addRow(lab, w_)
        sc = QScrollArea(); sc.setWidgetResizable(True); sc.setWidget(gb); lv.addWidget(sc, 3)
        self.w_level.currentIndexChanged.connect(self._level_changed); self._level_changed()
        for w_ in (self.w_metric, self.w_loc, self.w_norm, self.w_center, self.w_spread, self.w_agg, self.w_plot, self.w_indiv):
            w_.currentIndexChanged.connect(self._schedule)
        for w_ in (self.w_minpts, self.w_bin, self.w_t0, self.w_t1, self.w_normn, self.w_minn): w_.valueChanged.connect(self._schedule)
        self.w_filter.editingFinished.connect(self._schedule); self.w_band.toggled.connect(self._schedule)
        sp.addWidget(left)

        # ------------------------------------------------ 오른쪽: 그래프 + 표
        right = QWidget(); rv = QVBoxLayout(right)
        self.fig = Figure(figsize=(9, 5.5)); self.canvas = FigureCanvasQTAgg(self.fig)
        rv.addWidget(NavigationToolbar2QT(self.canvas, right)); rv.addWidget(self.canvas, 3)
        row = QHBoxLayout()
        b = QPushButton("그래프 + 데이터 저장…"); b.setToolTip("그림(png/pdf/svg/tif)과 같은 이름으로 _summary/_per_unit/_per_image/_unit_means/_stats.csv, _settings.json 저장")
        b.clicked.connect(self.export_dialog); row.addWidget(b)
        b = QPushButton("다시 읽기"); b.setToolTip("결과 파일을 디스크에서 다시 읽습니다 (분석을 다시 실행한 뒤)"); b.clicked.connect(self.reload); row.addWidget(b)
        self.info = QLabel(""); self.info.setStyleSheet("color: gray"); row.addWidget(self.info, 1); rv.addLayout(row)
        self.tabs = QTabWidget(); self.models = {}
        for key, lab in [("S", "시간별 요약"), ("M", "단위별 평균"), ("T", "통계 검정")]:
            tv = QTableView(); m = DataFrameModel(); tv.setModel(m); tv.setSortingEnabled(True); self.models[key] = m; self.tabs.addTab(tv, lab)
            if key == "M": tv.doubleClicked.connect(self._unit_clicked)
        rv.addWidget(self.tabs, 2); sp.addWidget(right); sp.setSizes([430, 1000])

    # ================================================================ 그룹 목록
    def groups(self) -> list[G.Group]:
        out = []
        for i in range(self.tree.topLevelItemCount()):
            it = self.tree.topLevelItem(i)
            out.append(G.Group(name=it.text(0), sources=[it.child(j).data(0, Qt.UserRole) for j in range(it.childCount())],
                               color=it.data(0, Qt.UserRole) or "", enabled=it.checkState(0) == Qt.Checked))
        return out

    def set_groups(self, groups: list[G.Group]):
        self.tree.blockSignals(True); self.tree.clear()
        for g in groups:
            it = self._new_group_item(g.name, g.color, g.enabled)
            for s in g.sources: self._add_source_item(it, s)
        self.tree.blockSignals(False); self._paint_colors(); self._data_changed()

    def _new_group_item(self, name, color="", enabled=True):
        it = QTreeWidgetItem([name, ""]); it.setData(0, ROLE_KIND, "group"); it.setData(0, Qt.UserRole, color)
        it.setFlags(it.flags() | Qt.ItemIsEditable | Qt.ItemIsUserCheckable); it.setCheckState(0, Qt.Checked if enabled else Qt.Unchecked)
        self.tree.addTopLevelItem(it); it.setExpanded(True); return it

    def _add_source_item(self, gi, src):
        for j in range(gi.childCount()):
            if gi.child(j).data(0, Qt.UserRole) == str(src): return
        try: desc = G.load_source(src)[2]
        except Exception as e: desc = f"⚠ {e}"   # noqa: BLE001
        p = Path(src); ch = QTreeWidgetItem([p.name if p.is_dir() else p.name, desc]); ch.setData(0, Qt.UserRole, str(src)); ch.setData(0, ROLE_KIND, "source")
        ch.setToolTip(0, str(src)); ch.setToolTip(1, desc); gi.addChild(ch)

    def _paint_colors(self):
        cols = G.series_colors(self.groups())
        for i in range(self.tree.topLevelItemCount()):
            it = self.tree.topLevelItem(i); self.tree.blockSignals(True)
            it.setForeground(0, QBrush(QColor(cols.get(it.text(0), "#000")))); f = it.font(0); f.setBold(True); it.setFont(0, f)
            self.tree.blockSignals(False)

    def _target_group(self, create=True):
        it = self.tree.currentItem()
        if it is not None and it.data(0, ROLE_KIND) == "source": it = it.parent()
        if it is None and self.tree.topLevelItemCount(): it = self.tree.topLevelItem(self.tree.topLevelItemCount() - 1)
        if it is None and create: it = self._new_group_item("그룹 1"); self._paint_colors()
        return it

    def add_group(self):
        names = {self.tree.topLevelItem(i).text(0) for i in range(self.tree.topLevelItemCount())}; k = self.tree.topLevelItemCount() + 1
        while f"그룹 {k}" in names: k += 1
        it = self._new_group_item(f"그룹 {k}"); self.tree.setCurrentItem(it); self._paint_colors()

    def _add_sources(self, srcs):
        gi = self._target_group()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            for s in srcs: self._add_source_item(gi, s)
        finally: QApplication.restoreOverrideCursor()
        gi.setExpanded(True); self._data_changed()

    def add_folder(self):
        d = QFileDialog.getExistingDirectory(self, "결과 폴더 (데이터셋·통합 결과, 또는 그 상위 폴더)")
        if d: self._add_sources([d])

    def add_csv(self):
        fs, _ = QFileDialog.getOpenFileNames(self, "결과 CSV (tracks_points_A_B_HT.csv / combined_tracks_points_A_B_HT.csv)", "", "CSV (*.csv)")
        if fs: self._add_sources(fs)

    def add_from_table(self):
        dirs = [d for _, d in self._dataset_dirs() if (Path(d) / G.POINTS_FILE).exists()]
        if not dirs: QMessageBox.information(self, "알림", "[데이터·실행] 표에 결과가 있는 데이터셋이 없습니다 (먼저 측정 단계까지 실행하세요)."); return
        self._add_sources(dirs)

    def remove_selected(self):
        for it in self.tree.selectedItems():
            (it.parent() or self.tree.invisibleRootItem()).removeChild(it)
        self._paint_colors(); self._data_changed()

    def set_color(self):
        gi = self._target_group(create=False)
        if gi is None: return
        c = QColorDialog.getColor(QColor(G.series_colors(self.groups()).get(gi.text(0), "#000")), self, f"{gi.text(0)} 색")
        if c.isValid(): gi.setData(0, Qt.UserRole, c.name()); self._paint_colors(); self._schedule()

    def _item_changed(self, it, col):
        if it.data(0, ROLE_KIND) == "group": self._paint_colors(); self._data_changed()

    # ================================================================ 설정
    def _level_changed(self):
        lvl = self.w_level.currentData(); cur = self.w_metric.currentData(); self.w_metric.blockSignals(True); self.w_metric.clear()
        for k, (lab, l) in G.METRICS.items():
            if l == lvl: self.w_metric.addItem(lab, k)
        i = self.w_metric.findData(cur); self.w_metric.setCurrentIndex(max(i, 0)); self.w_metric.blockSignals(False)
        self.w_loc.setEnabled(lvl == "particle"); self._schedule()

    def settings(self) -> G.CompareSettings:
        return G.CompareSettings(metric=self.w_metric.currentData() or "ratio_BA", location=self.w_loc.currentData(), min_points=self.w_minpts.value(),
                                 bin_h=self.w_bin.value(), t_min_h=self.w_t0.value(), t_max_h=self.w_t1.value(), center=self.w_center.currentData(),
                                 spread=self.w_spread.currentData(), norm=self.w_norm.currentData(), norm_n=self.w_normn.value(), agg=self.w_agg.currentData(),
                                 min_n=self.w_minn.value(), unit_filter=self.w_filter.text())

    def view_settings(self) -> dict:
        return dict(plot=self.w_plot.currentData(), individual=self.w_indiv.currentData(), band=self.w_band.isChecked())

    def set_settings(self, s: G.CompareSettings, view: dict | None = None):
        ws = [self.w_level, self.w_metric, self.w_loc, self.w_minpts, self.w_bin, self.w_t0, self.w_t1, self.w_norm, self.w_normn, self.w_center,
              self.w_spread, self.w_agg, self.w_minn, self.w_plot, self.w_indiv, self.w_filter, self.w_band]
        for w_ in ws: w_.blockSignals(True)
        self.w_level.setCurrentIndex(max(self.w_level.findData(G.METRICS.get(s.metric, ("", "particle"))[1]), 0))
        for w_ in ws: w_.blockSignals(False)
        self._level_changed()
        for w_ in ws: w_.blockSignals(True)
        self.w_metric.setCurrentIndex(max(self.w_metric.findData(s.metric), 0))
        for w_, v in [(self.w_loc, s.location), (self.w_norm, s.norm), (self.w_center, s.center), (self.w_spread, s.spread), (self.w_agg, s.agg)]:
            w_.setCurrentIndex(max(w_.findData(v), 0))
        for w_, v in [(self.w_minpts, s.min_points), (self.w_bin, s.bin_h), (self.w_t0, s.t_min_h), (self.w_t1, s.t_max_h), (self.w_normn, s.norm_n), (self.w_minn, s.min_n)]:
            w_.setValue(v)
        self.w_filter.setText(s.unit_filter)
        if view:
            self.w_plot.setCurrentIndex(max(self.w_plot.findData(view.get("plot", "time")), 0))
            self.w_indiv.setCurrentIndex(max(self.w_indiv.findData(view.get("individual", "none")), 0)); self.w_band.setChecked(view.get("band", True))
        for w_ in ws: w_.blockSignals(False)
        self._schedule()

    def _unit_clicked(self, idx):
        M = self.models["M"].df()
        if M.empty or "unit" not in M: return
        u = str(M.unit.iloc[idx.row()]); cur = [t.strip() for t in self.w_filter.text().split(",") if t.strip()]
        if u not in cur: self.w_filter.setText(", ".join(cur + [u])); self._schedule()

    # ================================================================ 계산·그리기
    def _schedule(self, *_):
        self._timer.start(150)

    def _data_changed(self):
        self._dirty_data = True; self._schedule()

    def reload(self):
        G._cache.clear()
        for i in range(self.tree.topLevelItemCount()):
            gi = self.tree.topLevelItem(i)
            for j in range(gi.childCount()):
                ch = gi.child(j)
                try: ch.setText(1, G.load_source(ch.data(0, Qt.UserRole))[2])
                except Exception as e: ch.setText(1, f"⚠ {e}")   # noqa: BLE001
        self._data_changed()

    def refresh(self):
        gs = self.groups(); s = self.settings(); v = self.view_settings()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            if self._dirty_data:
                self._P, self._C = G.load_groups(gs, log=self.log.emit); self._dirty_data = False
            U = G.unit_values(self._P, self._C, s); S = G.summarize(U, s); M = G.unit_means(U, s)
            T = G.stats_table(M) if M.series.nunique() >= 2 else pd.DataFrame()
        except Exception as e:   # noqa: BLE001
            QApplication.restoreOverrideCursor(); self.info.setText(f"⚠ {e}"); self.log.emit(f"그룹 비교 오류: {e}"); return
        QApplication.restoreOverrideCursor()
        self._res = dict(U=U, S=S, M=M, T=T, groups=gs, settings=s)
        self.models["S"].set_df(S); self.models["M"].set_df(M); self.models["T"].set_df(T)
        self.fig.clear(); cols = G.series_colors(gs)
        if U.empty:
            ax = self.fig.add_subplot(111); ax.axis("off")
            msg = "그룹을 만들고 결과를 추가하세요" if not any(g.sources for g in gs if g.enabled) else "조건에 맞는 값이 없습니다"
            if G.METRICS[s.metric][1] == "cell_whole" and (self._C is None or self._C.empty):
                msg = "세포 전체 밝기(cell_intensity.csv)가 없습니다.\n[데이터·실행]에서 '측정·분류' 단계를 다시 실행하면 만들어집니다 (1.10 이상)."
            ax.text(.5, .5, msg, ha="center", va="center", fontsize=12)
        elif v["plot"] == "both":
            a1 = self.fig.add_subplot(1, 3, (1, 2)); a2 = self.fig.add_subplot(1, 3, 3)
            G.plot_timecourse(a1, U, S, s, cols, v["individual"], v["band"]); G.plot_distribution(a2, M, s, cols, T)
        elif v["plot"] == "dist": G.plot_distribution(self.fig.add_subplot(111), M, s, cols, T)
        else: G.plot_timecourse(self.fig.add_subplot(111), U, S, s, cols, v["individual"], v["band"])
        try: self.fig.tight_layout()
        except Exception: pass   # noqa: BLE001
        self.canvas.draw_idle()
        if not U.empty:
            unit = "세포" if G.METRICS[s.metric][1] != "particle" else "트랙"
            self.info.setText(" · ".join(f"{k}: 이미지 {q.image.nunique()}, {unit} {q.unit.nunique()}" for k, q in U.groupby("series", sort=False)))
        else: self.info.setText("")

    # ================================================================ 저장
    def export_dialog(self):
        if not self._res or self._res["U"].empty: QMessageBox.information(self, "알림", "저장할 그래프가 없습니다."); return
        f, _ = QFileDialog.getSaveFileName(self, "그래프 + 데이터 저장 (같은 이름으로 CSV도 저장)", "group_compare.png",
                                           "PNG (*.png);;PDF (*.pdf);;SVG (*.svg);;TIFF (*.tif)")
        if not f: return
        r = self._res
        out = G.export(f, self.fig, r["U"], r["S"], r["M"], r["T"] if len(r["T"]) else None, r["groups"], r["settings"], dict(view=self.view_settings()))
        self.log.emit("그룹 비교 저장: " + ", ".join(Path(o).name for o in out))
        QMessageBox.information(self, "저장 완료", "저장했습니다:\n" + "\n".join(out))

    def save_setup_dialog(self):
        f, _ = QFileDialog.getSaveFileName(self, "그룹 구성 저장", "groups.json", "JSON (*.json)")
        if f: G.save_setup(f, self.groups(), self.settings(), dict(view=self.view_settings()))

    def load_setup_dialog(self):
        f, _ = QFileDialog.getOpenFileName(self, "그룹 구성 불러오기 (groups.json 또는 저장한 _settings.json)", "", "JSON (*.json)")
        if not f: return
        try: gs, s, d = G.load_setup(f)
        except Exception as e: QMessageBox.warning(self, "오류", f"읽을 수 없습니다: {e}"); return   # noqa: BLE001
        self.set_groups(gs); self.set_settings(s, d.get("view"))

    # 프로젝트 파일용
    def to_dict(self):
        return dict(groups=[asdict(g) for g in self.groups()], settings=asdict(self.settings()), view=self.view_settings())

    def from_dict(self, d):
        if not d: return
        self.set_groups([G.Group(**g) for g in d.get("groups", [])]); self.set_settings(G.CompareSettings.from_dict(d.get("settings")), d.get("view"))

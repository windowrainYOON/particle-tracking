"""ParticleTracker 메인 창.

탭 구성
  1. 데이터·실행   : 채널 파일 입력(자동 인식), 단계 선택 실행, 진행/로그
  2. 파라미터      : config.py 기반 자동 생성 폼, YAML 저장/불러오기
  3. 결과 보기     : 결과 폴더의 그림·영상·CSV 미리보기
  4. 통합 분석     : 여러 데이터셋 결과 합치기
  5. 특이 입자     : I(pHrodo)/I(Cy5) 증가 후 감소 입자 색인, 곡선 보기, 크롭 영상
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd
from PySide6.QtCore import Qt, QUrl, QSettings
from PySide6.QtGui import QPixmap, QDesktopServices, QAction
from PySide6.QtWidgets import (QMainWindow, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton, QLabel,
                               QTableWidget, QTableWidgetItem, QFileDialog, QCheckBox, QGroupBox, QProgressBar, QPlainTextEdit,
                               QMessageBox, QListWidget, QListWidgetItem, QComboBox, QScrollArea, QTableView, QLineEdit,
                               QSplitter, QAbstractItemView, QHeaderView, QStackedWidget)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from .. import __version__, style
from ..config import Config
from ..io_utils import DatasetSpec, auto_group_files
from ..pipeline import STAGES, STAGE_LABELS, DatasetRunner
from ..plots import track_trace
from .param_form import ParamForm
from .preview_panel import PreviewPanel
from .track_views import TrackMapView, CropViewer, TrackExplorer, DatasetContext
from .table_model import DataFrameModel
from .worker import start_worker
from . import tasks

try:
    from PySide6.QtMultimedia import QMediaPlayer
    from PySide6.QtMultimediaWidgets import QVideoWidget
    HAS_VIDEO = True
except Exception:   # noqa: BLE001
    HAS_VIDEO = False

COLS = ["이름", "Cy5 (A)", "pHrodo (B)", "HT", "출력 폴더", "상태"]
IMG_EXT, MOV_EXT, TAB_EXT = {".png", ".jpg"}, {".mp4", ".gif"}, {".csv"}


def open_path(p):
    QDesktopServices.openUrl(QUrl.fromLocalFile(str(p)))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        style.setup()
        self.setWindowTitle(f"ParticleTracker {__version__} — Cy5 / pHrodo / HT 입자 분석")
        self.resize(1400, 900)
        self.settings = QSettings("ParticleTracker", "ParticleTracker")
        self.cfg = Config(); self._thread = None; self._worker = None; self.rf = {}
        self.tabs = QTabWidget(); self.setCentralWidget(self.tabs)
        self.tabs.addTab(self._tab_data(), "1. 데이터·실행")
        self.tabs.addTab(self._tab_params(), "2. 파라미터")
        self.tabs.addTab(self._tab_results(), "3. 결과 보기")
        self.tabs.addTab(self._tab_combine(), "4. 통합 분석")
        self.tabs.addTab(self._tab_risefall(), "5. 특이 입자")
        self._menu(); self.statusBar().showMessage("준비")
        last = self.settings.value("last_config")
        if last and Path(last).exists():
            try: self.cfg = Config.load(last); self.params.set_config(self.cfg); self.rf_params.set_config(self.cfg)
            except Exception: pass   # noqa: BLE001

    # ================================================================ menu
    def _menu(self):
        m = self.menuBar().addMenu("파일")
        for text, fn in [("프로젝트 저장…", self.save_project), ("프로젝트 열기…", self.load_project), ("파라미터 저장…", self.save_params), ("파라미터 불러오기…", self.load_params)]:
            a = QAction(text, self); a.triggered.connect(fn); m.addAction(a)
        h = self.menuBar().addMenu("도움말"); a = QAction("사용법", self); a.triggered.connect(self._help); h.addAction(a)

    def _help(self):
        QMessageBox.information(self, "사용법",
            "1) [데이터·실행] 탭에서 tif 파일을 추가하면 파일명 키워드(Cy5/pHrodo/HT)로 자동 묶입니다.\n"
            "2) [파라미터] 탭에서 값을 조정합니다. 마우스를 올리면 설명이 보입니다.\n"
            "3) 실행할 단계를 고르고 [실행]. 파라미터를 바꾼 뒤에는 바뀐 단계부터만 다시 실행하면 됩니다 (캐시 사용).\n"
            "4) [결과 보기]에서 그림·영상·표를 확인합니다.\n"
            "5) [통합 분석]으로 여러 데이터셋을 합치고, [특이 입자]에서 증가 후 감소 입자를 찾고 크롭 영상을 만듭니다.")

    # ================================================================ tab 1
    def _tab_data(self):
        w = QWidget(); v = QVBoxLayout(w)
        row = QHBoxLayout()
        for text, fn in [("tif 파일 추가 (자동 인식)", self.add_files), ("폴더에서 불러오기", self.add_folder), ("수동 추가", self.add_manual),
                         ("선택 삭제", self.remove_rows), ("출력 루트 폴더 지정", self.set_out_root)]:
            b = QPushButton(text); b.clicked.connect(fn); row.addWidget(b)
        row.addStretch(); v.addLayout(row)
        self.tbl = QTableWidget(0, len(COLS)); self.tbl.setHorizontalHeaderLabels(COLS)
        self.tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive); self.tbl.horizontalHeader().setStretchLastSection(True)
        self.tbl.setSelectionBehavior(QAbstractItemView.SelectRows); self.tbl.itemSelectionChanged.connect(self._update_cache_label)
        self.tbl.itemChanged.connect(lambda _: self._specs_changed())     # 경로를 직접 고친 경우
        v.addWidget(self.tbl, 2)
        gb = QGroupBox("실행할 단계 (앞 단계 결과는 캐시에서 불러옵니다)"); g = QHBoxLayout(gb); self.stage_cb = {}
        for s in STAGES:
            cb = QCheckBox(STAGE_LABELS[s]); cb.setChecked(True); self.stage_cb[s] = cb; g.addWidget(cb)
        b = QPushButton("모두"); b.clicked.connect(lambda: [c.setChecked(True) for c in self.stage_cb.values()]); g.addWidget(b)
        b = QPushButton("그래프·영상만"); b.clicked.connect(lambda: [c.setChecked(s in ("figures", "movies")) for s, c in self.stage_cb.items()]); g.addWidget(b)
        g.addStretch(); v.addWidget(gb)
        self.cache_lbl = QLabel("캐시: -"); v.addWidget(self.cache_lbl)
        row = QHBoxLayout()
        self.btn_run_all = QPushButton("▶ 전체 데이터셋 실행"); self.btn_run_all.clicked.connect(lambda: self.run_datasets(False))
        self.btn_run_sel = QPushButton("▶ 선택 데이터셋 실행"); self.btn_run_sel.clicked.connect(lambda: self.run_datasets(True))
        self.btn_stop = QPushButton("■ 중지"); self.btn_stop.clicked.connect(self.stop); self.btn_stop.setEnabled(False)
        for b in (self.btn_run_all, self.btn_run_sel, self.btn_stop): row.addWidget(b)
        self.prog = QProgressBar(); row.addWidget(self.prog, 1); v.addLayout(row)
        self.prog_lbl = QLabel(""); v.addWidget(self.prog_lbl)
        self.logbox = QPlainTextEdit(); self.logbox.setReadOnly(True); self.logbox.setMaximumBlockCount(20000); v.addWidget(self.logbox, 2)
        return w

    def _specs(self, selected_only=False):
        rows = sorted({i.row() for i in self.tbl.selectedIndexes()}) if selected_only else range(self.tbl.rowCount())
        return [DatasetSpec(*(self.tbl.item(r, c).text() for c in range(5))) for r in rows]

    def _add_spec(self, s: DatasetSpec):
        r = self.tbl.rowCount(); self.tbl.insertRow(r)
        for c, val in enumerate([s.name, s.cy5, s.phrodo, s.ht, s.out_dir, ""]):
            it = QTableWidgetItem(val); it.setToolTip(val)
            if c == 5: it.setFlags(it.flags() & ~Qt.ItemIsEditable)
            self.tbl.setItem(r, c, it)
        self._refresh_status(r)

    def _refresh_status(self, r):
        out = Path(self.tbl.item(r, 4).text()); st = "완료" if (out / "tracks_points_A_B_HT.csv").exists() else "미실행"
        self.tbl.item(r, 5).setText(st)

    def _out_root(self):
        return self.settings.value("out_root", "")

    def add_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "tif 파일 선택 (Cy5/pHrodo/HT 여러 세트 가능)", self.settings.value("last_dir", ""), "TIFF (*.tif *.tiff)")
        if files: self._group_and_add(files)

    def add_folder(self):
        d = QFileDialog.getExistingDirectory(self, "tif 폴더 선택", self.settings.value("last_dir", ""))
        if d: self._group_and_add([str(p) for p in Path(d).glob("*.tif*")])

    def _group_and_add(self, files):
        self.cfg = self.params.apply_to(self.cfg); ch = self.cfg.channel
        self.settings.setValue("last_dir", str(Path(files[0]).parent))
        specs = auto_group_files(files, ch.cy5_keyword, ch.phrodo_keyword, ch.ht_keyword, self._out_root() or None)
        if not specs:
            QMessageBox.warning(self, "인식 실패", "Cy5 / pHrodo / HT 세트를 찾지 못했습니다.\n파일명 키워드를 [파라미터 > 채널·시간]에서 확인하거나 [수동 추가]를 쓰세요."); return
        for s in specs: self._add_spec(s)
        self.log(f"데이터셋 {len(specs)}개 추가: {[s.name for s in specs]}"); self._refresh_combine_list(); self._specs_changed()

    def add_manual(self):
        paths = []
        for ch in ("Cy5 (A, ROI 검출)", "pHrodo (B)", "HT"):
            f, _ = QFileDialog.getOpenFileName(self, f"{ch} 파일", self.settings.value("last_dir", ""), "TIFF (*.tif *.tiff)")
            if not f: return
            paths.append(f)
        name = Path(paths[0]).stem; root = self._out_root() or str(Path(paths[0]).parent / "ParticleTracker_results")
        self._add_spec(DatasetSpec(name, *paths, str(Path(root) / name))); self._refresh_combine_list(); self._specs_changed()

    def remove_rows(self):
        for r in sorted({i.row() for i in self.tbl.selectedIndexes()}, reverse=True): self.tbl.removeRow(r)
        self._specs_changed()

    def set_out_root(self):
        d = QFileDialog.getExistingDirectory(self, "출력 루트 폴더", self._out_root())
        if not d: return
        self.settings.setValue("out_root", d)
        for r in range(self.tbl.rowCount()):
            self.tbl.item(r, 4).setText(str(Path(d) / self.tbl.item(r, 0).text())); self._refresh_status(r)
        self._specs_changed()

    def _update_cache_label(self):
        specs = self._specs(True)
        if not specs: self.cache_lbl.setText("캐시: -"); return
        s = specs[0]; cached = DatasetRunner(s, self.cfg).cached_stages()
        self.cache_lbl.setText(f"캐시 ({s.name}): " + (", ".join(STAGE_LABELS[c] for c in cached) if cached else "없음"))

    def run_datasets(self, selected_only):
        if self._busy(): return
        specs = self._specs(selected_only)
        if not specs: QMessageBox.information(self, "알림", "실행할 데이터셋이 없습니다."); return
        stages = [s for s, c in self.stage_cb.items() if c.isChecked()]
        if not stages: return
        self.cfg = self.params.apply_to(self.cfg)
        self._start(tasks.run_datasets, specs, self.cfg, stages, done=self._after_run)

    def _after_run(self, ok, msg):
        self.preview.reset_sources()        # 새로 만들어진 캐시(정합·분열)를 미리보기에 반영
        for r in range(self.tbl.rowCount()): self._refresh_status(r)
        self._refresh_combine_list(); self._refresh_result_dirs()

    # ================================================================ tab 2
    def _tab_params(self):
        w = QWidget(); v = QVBoxLayout(w); row = QHBoxLayout()
        for text, fn in [("불러오기 (YAML)", self.load_params), ("저장 (YAML)", self.save_params), ("기본값 복원", self.reset_params)]:
            b = QPushButton(text); b.clicked.connect(fn); row.addWidget(b)
        row.addStretch(); v.addLayout(row)
        v.addWidget(QLabel("값을 바꾼 뒤 실행하면 적용됩니다. 항목에 마우스를 올리면 설명이 표시됩니다. "
                           "바뀐 파라미터가 속한 단계부터 다시 실행하세요 (예: 트래킹 파라미터 → 트래킹·측정·그래프·영상)."))
        sp = QSplitter(); self.params = ParamForm(self.cfg); sp.addWidget(self.params)
        self.preview = PreviewPanel(lambda: self.params.apply_to(Config()), is_busy=lambda: self._thread is not None)
        sp.addWidget(self.preview); sp.setSizes([460, 940]); v.addWidget(sp, 1)
        self.params.changed.connect(lambda sec: self.preview.request())
        self.params.sectionChanged.connect(self.preview.set_section); self.preview.set_section(self.params.current_section())
        return w

    def _apply_learned(self, path):
        self.params._widgets[("tracking", "learned_model")].setText(path); self.cfg = self.params.apply_to(self.cfg)
        self.log(f"학습된 연결 모델을 지정했습니다: {path} — 트래킹 → 측정 단계를 다시 실행하면 반영됩니다")

    def _specs_changed(self):
        if hasattr(self, "preview"):
            try: self.preview.set_specs(self._specs())
            except AttributeError: pass     # 행을 채우는 중 (아직 빈 칸)

    def load_params(self):
        f, _ = QFileDialog.getOpenFileName(self, "파라미터 불러오기", "", "YAML (*.yaml *.yml)")
        if f: self.cfg = Config.load(f); self.params.set_config(self.cfg); self.rf_params.set_config(self.cfg); self.settings.setValue("last_config", f); self.log(f"파라미터 불러옴: {f}")

    def save_params(self):
        f, _ = QFileDialog.getSaveFileName(self, "파라미터 저장", "params.yaml", "YAML (*.yaml)")
        if f: self.cfg = self.params.apply_to(self.cfg); self.cfg.save(f); self.settings.setValue("last_config", f); self.log(f"파라미터 저장: {f}")

    def reset_params(self):
        self.cfg = Config(); self.params.set_config(self.cfg); self.rf_params.set_config(self.cfg)

    # ================================================================ project
    def save_project(self):
        f, _ = QFileDialog.getSaveFileName(self, "프로젝트 저장", "project.json", "JSON (*.json)")
        if not f: return
        self.cfg = self.params.apply_to(self.cfg)
        Path(f).write_text(json.dumps(dict(datasets=[s.__dict__ for s in self._specs()], config=self.cfg.to_dict()), ensure_ascii=False, indent=1), encoding="utf-8")

    def load_project(self):
        f, _ = QFileDialog.getOpenFileName(self, "프로젝트 열기", "", "JSON (*.json)")
        if not f: return
        d = json.loads(Path(f).read_text(encoding="utf-8")); self.tbl.setRowCount(0)
        for s in d.get("datasets", []): self._add_spec(DatasetSpec(**s))
        self.cfg = Config.from_dict(d.get("config", {})); self.params.set_config(self.cfg); self.rf_params.set_config(self.cfg); self._refresh_combine_list(); self._refresh_result_dirs()
        self._specs_changed()

    # ================================================================ tab 3
    def _tab_results(self):
        w = QWidget(); v = QVBoxLayout(w); row = QHBoxLayout()
        self.res_dir = QComboBox(); self.res_dir.setEditable(True); self.res_dir.currentTextChanged.connect(self._list_results)
        row.addWidget(QLabel("결과 폴더")); row.addWidget(self.res_dir, 1)
        for text, fn in [("폴더 추가…", self._browse_result_dir), ("새로고침", lambda: self._list_results(self.res_dir.currentText())),
                         ("폴더 열기", lambda: open_path(self.res_dir.currentText()))]:
            b = QPushButton(text); b.clicked.connect(fn); row.addWidget(b)
        self.res_filter = QComboBox(); self.res_filter.addItems(["전체", "그림", "영상", "표(CSV)"]); self.res_filter.currentTextChanged.connect(lambda _: self._list_results(self.res_dir.currentText()))
        row.addWidget(self.res_filter); v.addLayout(row)
        sp = QSplitter(); self.res_list = QListWidget(); self.res_list.currentItemChanged.connect(self._preview); sp.addWidget(self.res_list)
        right = QWidget(); rv = QVBoxLayout(right); self.prev_stack = QStackedWidget()
        self.img_lbl = QLabel("파일을 선택하세요"); self.img_lbl.setAlignment(Qt.AlignCenter); sc = QScrollArea(); sc.setWidgetResizable(True); sc.setWidget(self.img_lbl)
        self.prev_stack.addWidget(sc)
        self.csv_view = QTableView(); self.csv_model = DataFrameModel(); self.csv_view.setModel(self.csv_model); self.csv_view.setSortingEnabled(True); self.prev_stack.addWidget(self.csv_view)
        if HAS_VIDEO:
            self.video = QVideoWidget(); self.player = QMediaPlayer(self); self.player.setVideoOutput(self.video); self.prev_stack.addWidget(self.video)
        rv.addWidget(self.prev_stack, 1); row = QHBoxLayout()
        b = QPushButton("외부 앱으로 열기"); b.clicked.connect(self._open_current); row.addWidget(b)
        if HAS_VIDEO:
            for text, fn in [("▶ 재생", lambda: self.player.play()), ("⏸ 일시정지", lambda: self.player.pause())]:
                b = QPushButton(text); b.clicked.connect(fn); row.addWidget(b)
        row.addStretch(); rv.addLayout(row); sp.addWidget(right); sp.setSizes([350, 1000])
        self.res_tabs = QTabWidget(); self.res_tabs.addTab(sp, "파일 보기")
        self.explorer = TrackExplorer(lambda: self.params.apply_to(Config())); self.res_tabs.addTab(self.explorer, "트랙 탐색 (경로·밝기 그래프)")
        self.res_dir.currentTextChanged.connect(self.explorer.set_folder); v.addWidget(self.res_tabs, 1)
        self.explorer.annot.on_trained = self._apply_learned
        base_folders = self.explorer.annot.get_gt_folders
        self.explorer.annot.get_gt_folders = lambda: [self.tbl.item(r, 4).text() for r in range(self.tbl.rowCount())] + base_folders()
        return w

    def _refresh_result_dirs(self):
        cur = self.res_dir.currentText(); dirs = [self.tbl.item(r, 4).text() for r in range(self.tbl.rowCount())]
        extra = [self.comb_out.text(), self.rf_out.text()] if hasattr(self, "rf_out") else []
        self.res_dir.blockSignals(True); self.res_dir.clear()
        for d in dict.fromkeys(dirs + extra):
            if d and Path(d).exists(): self.res_dir.addItem(d)
        self.res_dir.blockSignals(False)
        if cur: self.res_dir.setCurrentText(cur)
        self._list_results(self.res_dir.currentText())

    def _browse_result_dir(self):
        d = QFileDialog.getExistingDirectory(self, "결과 폴더")
        if d: self.res_dir.addItem(d); self.res_dir.setCurrentText(d)

    def _list_results(self, d):
        self.res_list.clear(); p = Path(d) if d else None
        if not p or not p.exists(): return
        f = self.res_filter.currentText(); ext = {"그림": IMG_EXT, "영상": MOV_EXT, "표(CSV)": TAB_EXT}.get(f, IMG_EXT | MOV_EXT | TAB_EXT)
        for x in sorted(p.rglob("*")):
            if x.suffix.lower() in ext and "_cache" not in x.parts:
                it = QListWidgetItem(str(x.relative_to(p))); it.setData(Qt.UserRole, str(x)); self.res_list.addItem(it)

    def _preview(self, it, _prev=None):
        if it is None: return
        p = Path(it.data(Qt.UserRole)); s = p.suffix.lower()
        if HAS_VIDEO: self.player.stop()
        if s in IMG_EXT:
            pm = QPixmap(str(p)); self.img_lbl.setPixmap(pm.scaledToWidth(min(pm.width(), 1300), Qt.SmoothTransformation)); self.prev_stack.setCurrentIndex(0)
        elif s in TAB_EXT:
            try: self.csv_model.set_df(pd.read_csv(p, nrows=2000)); self.prev_stack.setCurrentIndex(1)
            except Exception as e: self.img_lbl.setText(str(e))   # noqa: BLE001
        elif s in MOV_EXT and HAS_VIDEO and s == ".mp4":
            self.player.setSource(QUrl.fromLocalFile(str(p))); self.prev_stack.setCurrentIndex(2); self.player.play()
        else:
            self.img_lbl.setText(f"{p.name}\n[외부 앱으로 열기]를 누르세요"); self.prev_stack.setCurrentIndex(0)

    def _open_current(self):
        it = self.res_list.currentItem()
        if it: open_path(it.data(Qt.UserRole))

    # ================================================================ tab 4
    def _tab_combine(self):
        w = QWidget(); v = QVBoxLayout(w)
        v.addWidget(QLabel("통합할 데이터셋 결과 폴더 (체크한 항목만). 이름이 데이터셋 이름으로 쓰입니다."))
        self.comb_list = QListWidget(); v.addWidget(self.comb_list, 1); row = QHBoxLayout()
        b = QPushButton("결과 폴더 추가…"); b.clicked.connect(self._add_comb_dir); row.addWidget(b)
        b = QPushButton("목록 새로고침"); b.clicked.connect(self._refresh_combine_list); row.addWidget(b); row.addStretch(); v.addLayout(row)
        row = QHBoxLayout(); row.addWidget(QLabel("출력 폴더")); self.comb_out = QLineEdit(); row.addWidget(self.comb_out, 1)
        b = QPushButton("…"); b.clicked.connect(lambda: self._pick_dir(self.comb_out)); row.addWidget(b); v.addLayout(row)
        b = QPushButton("▶ 통합 실행"); b.clicked.connect(self.run_combine); v.addWidget(b)
        return w

    def _pick_dir(self, le):
        d = QFileDialog.getExistingDirectory(self, "폴더 선택", le.text())
        if d: le.setText(d)

    def _add_comb_dir(self):
        d = QFileDialog.getExistingDirectory(self, "데이터셋 결과 폴더")
        if d: self._add_comb_item(d)

    def _add_comb_item(self, d):
        for i in range(self.comb_list.count()):
            if self.comb_list.item(i).data(Qt.UserRole) == d: return
        it = QListWidgetItem(f"{Path(d).name}   —   {d}"); it.setData(Qt.UserRole, d); it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
        it.setCheckState(Qt.Checked if (Path(d) / "tracks_points_A_B_HT.csv").exists() else Qt.Unchecked); self.comb_list.addItem(it)

    def _refresh_combine_list(self):
        for r in range(self.tbl.rowCount()): self._add_comb_item(self.tbl.item(r, 4).text())
        if not self.comb_out.text() and self.tbl.rowCount():
            self.comb_out.setText(str(Path(self.tbl.item(0, 4).text()).parent / "Combined"))
        if hasattr(self, "rf_src") and not self.rf_src.text() and self.comb_out.text(): self.rf_src.setText(self.comb_out.text())
        if hasattr(self, "rf_out") and not self.rf_out.text() and self.comb_out.text(): self.rf_out.setText(str(Path(self.comb_out.text()).parent / "RiseFall"))

    def run_combine(self):
        if self._busy(): return
        dirs = {Path(self.comb_list.item(i).data(Qt.UserRole)).name: self.comb_list.item(i).data(Qt.UserRole)
                for i in range(self.comb_list.count()) if self.comb_list.item(i).checkState() == Qt.Checked}
        if not dirs or not self.comb_out.text(): QMessageBox.information(self, "알림", "통합할 폴더와 출력 폴더를 지정하세요."); return
        self.cfg = self.params.apply_to(self.cfg)
        self._start(tasks.combine, dirs, self.comb_out.text(), self.cfg, done=lambda ok, m: (self._refresh_result_dirs(), self.res_dir.setCurrentText(self.comb_out.text())))

    # ================================================================ tab 5
    def _tab_risefall(self):
        w = QWidget(); v = QVBoxLayout(w); g = QGridLayout()
        g.addWidget(QLabel("입력 (통합 결과 폴더/CSV 또는 단일 데이터셋 결과 폴더)"), 0, 0); self.rf_src = QLineEdit(); g.addWidget(self.rf_src, 0, 1)
        b = QPushButton("폴더…"); b.clicked.connect(lambda: self._pick_dir(self.rf_src)); g.addWidget(b, 0, 2)
        b = QPushButton("CSV…"); b.clicked.connect(self._pick_rf_csv); g.addWidget(b, 0, 3)
        g.addWidget(QLabel("출력 폴더"), 1, 0); self.rf_out = QLineEdit(); g.addWidget(self.rf_out, 1, 1)
        b = QPushButton("…"); b.clicked.connect(lambda: self._pick_dir(self.rf_out)); g.addWidget(b, 1, 2); v.addLayout(g)
        sp = QSplitter(); left = QWidget(); lv = QVBoxLayout(left)
        lv.addWidget(QLabel("판정 기준 (여기서 바꾼 값이 [파라미터] 탭에도 반영됩니다)"))
        self.rf_params = ParamForm(self.cfg, only=["risefall"]); lv.addWidget(self.rf_params, 1)
        b = QPushButton("▶ 특이 입자 분류 실행"); b.clicked.connect(self.run_risefall); lv.addWidget(b); sp.addWidget(left)
        mid = QWidget(); mv = QVBoxLayout(mid)
        self.rf_frac = QTableView(); self.rf_frac_model = DataFrameModel(); self.rf_frac.setModel(self.rf_frac_model); self.rf_frac.setMaximumHeight(190); mv.addWidget(self.rf_frac)
        row = QHBoxLayout(); self.rf_group = QComboBox(); self.rf_group.addItems(["세포 안", "세포 밖", "전체"]); self.rf_show = QComboBox(); self.rf_show.addItems(["선별 입자만", "후보 전체(제외 포함)", "모든 장기 트랙"])
        for wdg in (self.rf_group, self.rf_show): wdg.currentTextChanged.connect(self._rf_filter); row.addWidget(wdg)
        row.addStretch(); mv.addLayout(row)
        self.rf_table = QTableView(); self.rf_model = DataFrameModel(); self.rf_table.setModel(self.rf_model); self.rf_table.setSortingEnabled(True)
        self.rf_table.setSelectionBehavior(QAbstractItemView.SelectRows); self.rf_table.selectionModel().selectionChanged.connect(self._rf_plot_selected)
        mv.addWidget(self.rf_table, 1)
        self.fig = Figure(figsize=(7, 3.2)); self.canvas = FigureCanvasQTAgg(self.fig)
        self.rf_views = QTabWidget(); self.rf_views.addTab(self.canvas, "곡선")
        self.rf_map = TrackMapView(); self.rf_map.trackClicked.connect(self._rf_select_uid); self.rf_views.addTab(self.rf_map, "전체 영상 위치")
        self.rf_crop = CropViewer(); self.rf_views.addTab(self.rf_crop, "크롭 프레임"); self._rf_ctx = {}
        row = QHBoxLayout()
        b = QPushButton("선택 입자 크롭 영상 + 그래프"); b.clicked.connect(lambda: self.run_crops(False)); row.addWidget(b)
        b = QPushButton("현재 목록 전체 크롭"); b.clicked.connect(lambda: self.run_crops(True)); row.addWidget(b)
        b = QPushButton("결과 폴더 열기"); b.clicked.connect(lambda: open_path(self.rf_out.text())); row.addWidget(b); row.addStretch(); mv.addLayout(row)
        sp.addWidget(mid); sp.addWidget(self.rf_views); sp.setSizes([330, 560, 720]); v.addWidget(sp, 1)
        return w

    def _pick_rf_csv(self):
        f, _ = QFileDialog.getOpenFileName(self, "통합 point CSV", "", "CSV (*.csv)")
        if f: self.rf_src.setText(f)

    def run_risefall(self):
        if self._busy(): return
        if not self.rf_src.text() or not self.rf_out.text(): QMessageBox.information(self, "알림", "입력과 출력 폴더를 지정하세요."); return
        self.cfg = self.params.apply_to(self.cfg); self.cfg = self.rf_params.apply_to(self.cfg); self.params.set_config(self.cfg)
        self.rf = {}; self._rf_ctx = {}
        self._start(tasks.risefall, self.rf_src.text(), self.cfg, self.rf_out.text(), self.rf, done=self._after_rf)

    def _after_rf(self, ok, msg):
        if not ok or "R" not in self.rf: return
        F = self.rf["F"]; self.rf_frac_model.set_df(F[["dataset", "group", "n_tracks", "candidates", "kept", "kept_abrupt", "kept_gradual", "kept_pct", "kept_pct_ci_low", "kept_pct_ci_high"]])
        self._rf_filter(); self._refresh_result_dirs()

    def _rf_filter(self, *_):
        R = self.rf.get("R")
        if R is None: return
        g = {"세포 안": ["inside"], "세포 밖": ["outside"], "전체": ["inside", "outside"]}[self.rf_group.currentText()]
        d = R[R.group.isin(g)]; s = self.rf_show.currentText()
        d = d[d.keep] if s == "선별 입자만" else (d[d.candidate] if s.startswith("후보") else d)
        cols = ["track_uid", "dataset", "group", "category", "keep", "filter_result", "base", "peak", "end", "peak_time_h", "decline_time_h", "n_valid"]
        self.rf_model.set_df(d[cols].rename(columns={"base": "ratio_start", "peak": "ratio_peak", "end": "ratio_end"}))
        self._rf_update_map()

    def _selected_uids(self):
        rows = sorted({i.row() for i in self.rf_table.selectionModel().selectedRows()}); df = self.rf_model.df()
        return [df.track_uid.iloc[r] for r in rows]

    def _rf_plot_selected(self, *_):
        uids = self._selected_uids(); pts = self.rf.get("points")
        if not uids or pts is None: return
        u = uids[0]; t = pts[pts.track_uid == u]; R = self.rf["R"]; r = R[R.track_uid == u].iloc[0] if u in set(R.track_uid) else None
        self.fig.clear(); ax = self.fig.add_subplot(111); track_trace(ax, t, r, show_channels=True); ax.set_title(u, fontsize=9); self.fig.tight_layout(); self.canvas.draw_idle()
        ds = t.dataset.iloc[0]; ctx = self._rf_context(ds); self._rf_update_map(ds)
        pk = int(r.peak_frame) if r is not None and pd.notna(r.peak_frame) else int(t.frame.min())
        self.rf_map.select(u, frame=pk); cfg = self.params.apply_to(Config())
        info = f"{ds} | {u.split('::')[-1]} | {r.category if r is not None else ''} {'' if r is None or r.keep else '(제외: ' + str(r.filter_result) + ')'}"
        self.rf_crop.set_track(ctx, t, r, cfg.output.crop_half, cfg.channel.frame_interval_min, info, cfg.output.movie_fps)

    def _rf_context(self, ds):
        if ds not in self._rf_ctx:
            d = (self.rf.get("dirs") or {}).get(ds)
            try: self._rf_ctx[ds] = DatasetContext(d) if d else None
            except Exception: self._rf_ctx[ds] = None   # noqa: BLE001
        return self._rf_ctx[ds]

    def _rf_update_map(self, ds=None):
        """현재 표에 보이는 입자들(선택한 입자의 데이터셋)을 전체 영상 위에 표시. ★ = 최고점 위치."""
        pts = self.rf.get("points"); df = self.rf_model.df()
        if pts is None or not len(df): self.rf_map.set_tracks(pd.DataFrame()); return
        ds = ds or (self._selected_uids()[0].split("::")[0] if self._selected_uids() else df.dataset.iloc[0])
        sel = df[df.dataset == ds]; P = pts[pts.track_uid.isin(set(sel.track_uid))]
        R = self.rf["R"].set_index("track_uid"); mk = {u: int(R.peak_frame[u]) for u in sel.track_uid if pd.notna(R.peak_frame.get(u))}
        self.rf_map.set_context(self._rf_context(ds))
        self.rf_map.set_tracks(P[["track_uid", "frame", "x_fl", "y_fl", "x_ht", "y_ht", "group"]], mk)

    def _rf_select_uid(self, uid):
        df = self.rf_model.df(); hit = np.nonzero(df.track_uid.values == uid)[0]
        if len(hit): self.rf_table.selectRow(int(hit[0])); self.rf_table.scrollTo(self.rf_model.index(int(hit[0]), 0))

    def run_crops(self, all_rows):
        if self._busy() or "R" not in self.rf: return
        uids = list(self.rf_model.df().track_uid) if all_rows else self._selected_uids()
        if not uids: QMessageBox.information(self, "알림", "입자를 선택하세요."); return
        if not self.rf.get("dirs"): QMessageBox.warning(self, "원본 위치 없음", "원본 데이터셋 폴더 정보(datasets_map.json)가 없습니다. [통합 분석]으로 통합한 결과를 입력하세요."); return
        self.cfg = self.params.apply_to(self.cfg)
        self._start(tasks.crops, uids, self.rf["points"], self.rf["R"], self.rf["dirs"], self.cfg, str(Path(self.rf_out.text()) / "crops"),
                    done=lambda ok, m: (self._refresh_result_dirs(), self.res_dir.setCurrentText(self.rf_out.text())))

    # ================================================================ common
    def log(self, msg):
        self.logbox.appendPlainText(msg)

    def _busy(self):
        if self._thread is not None:
            QMessageBox.information(self, "작업 중", "다른 작업이 실행 중입니다."); return True
        return False

    def _start(self, fn, *args, done=None):
        self.prog.setValue(0); self.btn_stop.setEnabled(True); self._done_cb = done
        for b in (self.btn_run_all, self.btn_run_sel): b.setEnabled(False)
        if fn is tasks.run_datasets: self.tabs.setCurrentIndex(0)
        # 슬롯은 모두 MainWindow의 메서드 → 워커 스레드 신호가 메인 스레드에서 안전하게 처리됨
        self._thread, self._worker = start_worker(self, fn, *args, on_log=self.log, on_progress=self._on_progress, on_finished=self._on_finished)

    def _on_progress(self, p, m):
        self.prog.setValue(p); self.prog_lbl.setText(m); self.statusBar().showMessage(m)

    def _on_finished(self, ok, msg):
        self._thread = None; self._worker = None; self.btn_stop.setEnabled(False)
        for b in (self.btn_run_all, self.btn_run_sel): b.setEnabled(True)
        self.log(("✔ " if ok else "✖ ") + msg); self.statusBar().showMessage(msg)
        cb, self._done_cb = self._done_cb, None
        if cb: cb(ok, msg)

    def stop(self):
        if self._worker is not None: self._worker.cancel_event.set(); self.log("중지 요청… (현재 프레임 처리 후 멈춥니다)")

    def closeEvent(self, e):
        self.cfg = self.params.apply_to(self.cfg)
        p = Path.home() / ".particletracker_last.yaml"
        try: self.cfg.save(p); self.settings.setValue("last_config", str(p))
        except Exception: pass   # noqa: BLE001
        if self._worker is not None: self._worker.cancel_event.set()
        self.preview.shutdown(); self.explorer.shutdown(); self.rf_crop.shutdown()
        super().closeEvent(e)

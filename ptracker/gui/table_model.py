"""pandas DataFrame을 QTableView에 보여주는 모델."""
import numpy as np, pandas as pd
from PySide6.QtCore import QAbstractTableModel, Qt, QModelIndex


class DataFrameModel(QAbstractTableModel):
    def __init__(self, df: pd.DataFrame | None = None, parent=None):
        super().__init__(parent); self._df = df if df is not None else pd.DataFrame()

    def set_df(self, df):
        self.beginResetModel(); self._df = df.reset_index(drop=True); self.endResetModel()

    def df(self): return self._df

    def rowCount(self, parent=QModelIndex()): return len(self._df)

    def columnCount(self, parent=QModelIndex()): return len(self._df.columns)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or role != Qt.DisplayRole: return None
        v = self._df.iat[index.row(), index.column()]
        if isinstance(v, (float, np.floating)): return "" if np.isnan(v) else f"{v:.4g}"
        return str(v)

    def headerData(self, s, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole: return None
        return str(self._df.columns[s]) if orientation == Qt.Horizontal else str(s + 1)

    def sort(self, column, order=Qt.AscendingOrder):
        if self._df.empty: return
        self.beginResetModel()
        self._df = self._df.sort_values(self._df.columns[column], ascending=order == Qt.AscendingOrder, kind="mergesort").reset_index(drop=True)
        self.endResetModel()

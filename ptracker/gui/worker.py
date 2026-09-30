"""백그라운드 작업 스레드. GUI가 멈추지 않도록 분석은 QThread에서 실행합니다."""
from __future__ import annotations
import threading, traceback
from PySide6.QtCore import QObject, QThread, Signal


class Worker(QObject):
    log = Signal(str)
    progress = Signal(int, str)          # 0~100, 메시지
    finished = Signal(bool, str)         # 성공 여부, 메시지

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn, self.args, self.kwargs = fn, args, kwargs
        self.cancel_event = threading.Event()

    def run(self):
        try:
            self.fn(*self.args, worker=self, **self.kwargs)
            self.finished.emit(True, "완료")
        except Exception as e:           # noqa: BLE001
            from ..pipeline import Cancelled
            if isinstance(e, Cancelled):
                self.finished.emit(False, "사용자가 중지함")
            else:
                self.log.emit(traceback.format_exc()); self.finished.emit(False, f"오류: {e}")


_alive: set = set()   # 실행 중인 (thread, worker). 스레드가 끝날 때까지 Python 참조를 유지


def start_worker(parent, fn, *args, on_log=None, on_progress=None, on_finished=None, **kwargs):
    """Worker를 새 QThread에서 시작하고 (thread, worker)를 반환.

    Worker는 Python이 소유합니다(deleteLater 사용 안 함). 예전에는 deleteLater와 Python 참조 해제가
    같은 객체를 두 번 지워 작업이 끝난 직후 프로그램이 강제 종료(segfault)되었습니다.
    """
    th = QThread(parent); w = Worker(fn, *args, **kwargs); w.moveToThread(th)
    th.started.connect(w.run)
    if on_log: w.log.connect(on_log)
    if on_progress: w.progress.connect(on_progress)
    if on_finished: w.finished.connect(on_finished)
    w.finished.connect(th.quit)
    pair = (th, w); _alive.add(pair)
    th.finished.connect(lambda: _alive.discard(pair)); th.finished.connect(th.deleteLater)
    th.start()
    return th, w

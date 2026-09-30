import sys


def main():
    from PySide6.QtWidgets import QApplication
    from .main_window import MainWindow
    app = QApplication(sys.argv); app.setApplicationName("ParticleTracker")
    w = MainWindow(); w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

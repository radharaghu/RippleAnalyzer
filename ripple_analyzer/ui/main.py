"""Main window: pick a folder, batch-detect ripples, inspect any recording."""

from __future__ import annotations

import sys
import tempfile
import threading
from pathlib import Path
from typing import List, Optional

from PyQt5.QtCore import QObject, Qt, pyqtSignal
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (
    QAbstractItemView, QApplication, QDoubleSpinBox, QFileDialog, QHBoxLayout, QHeaderView,
    QLabel, QMainWindow, QProgressBar, QPushButton, QShortcut, QSpinBox, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from ripple_analyzer import __version__
from ripple_analyzer.core.batch_processor import BatchProcessor, write_results_csv
from ripple_analyzer.core.detector import RippleDetector
from ripple_analyzer.core.file_handlers import FileLoader
from ripple_analyzer.ui.visualization import RippleViewer

COLUMNS = ["File", "Duration (s)", "Rate (Hz)", "Ripples", "Ripples / min", "Event / baseline", "Status"]


class _Signals(QObject):
    progress = pyqtSignal(int, int, str, str)
    finished = pyqtSignal(list)
    failed = pyqtSignal(str)


class _NumericItem(QTableWidgetItem):
    """Table cell that sorts by its numeric value."""

    def __init__(self, value: Optional[float], fmt: str):
        super().__init__("—" if value is None else format(value, fmt))
        self.value = float("-inf") if value is None else float(value)
        self.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)

    def __lt__(self, other):
        return self.value < getattr(other, "value", float("-inf"))


class RippleAnalyzerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"RippleAnalyzer {__version__}")
        self.resize(1100, 650)
        self.directory: Optional[Path] = None
        self.results: List[dict] = []
        self.viewers: List[RippleViewer] = []
        self.signals = _Signals()
        self.signals.progress.connect(self._on_progress)
        self.signals.finished.connect(self._on_finished)
        self.signals.failed.connect(self._on_failed)
        self._build_ui()

    # ------------------------------------------------------------------ layout
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        row = QHBoxLayout()
        self.open_button = QPushButton("Open folder…")
        self.open_button.clicked.connect(self.choose_folder)
        self.demo_button = QPushButton("Try demo data")
        self.demo_button.setToolTip("Generate four synthetic recordings with known ripple times")
        self.demo_button.clicked.connect(self.load_demo)
        self.folder_label = QLabel("No folder selected")
        self.folder_label.setStyleSheet("color:#666")
        row.addWidget(self.open_button)
        row.addWidget(self.demo_button)
        row.addWidget(self.folder_label, 1)
        layout.addLayout(row)

        settings = QHBoxLayout()
        settings.addWidget(QLabel("Threshold (σ)"))
        self.sigma_box = QDoubleSpinBox()
        self.sigma_box.setRange(0.5, 6.0)
        self.sigma_box.setSingleStep(0.1)
        self.sigma_box.setValue(2.0)
        settings.addWidget(self.sigma_box)
        settings.addSpacing(16)
        settings.addWidget(QLabel("Sampling rate for CSV/BIN without timestamps (Hz)"))
        self.fs_box = QSpinBox()
        self.fs_box.setRange(500, 100000)
        self.fs_box.setSingleStep(250)
        self.fs_box.setValue(30000)
        settings.addWidget(self.fs_box)
        settings.addStretch(1)
        self.run_button = QPushButton("Run analysis")
        self.run_button.setEnabled(False)
        self.run_button.setDefault(True)
        self.run_button.clicked.connect(self.run_analysis)
        settings.addWidget(self.run_button)
        layout.addLayout(settings)

        self.progress = QProgressBar()
        self.progress.setFormat("%v / %m files")
        self.progress.setValue(0)
        layout.addWidget(self.progress)

        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for c in range(1, len(COLUMNS)):
            self.table.horizontalHeader().setSectionResizeMode(c, QHeaderView.ResizeToContents)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.cellDoubleClicked.connect(lambda r, _c: self.open_row(r))
        layout.addWidget(self.table, 1)

        bottom = QHBoxLayout()
        self.summary = QLabel("")
        bottom.addWidget(self.summary, 1)
        self.view_button = QPushButton("Inspect selected")
        self.view_button.setEnabled(False)
        self.view_button.clicked.connect(lambda: self.open_row(self.table.currentRow()))
        self.export_button = QPushButton("Export CSV…")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export_results)
        bottom.addWidget(self.view_button)
        bottom.addWidget(self.export_button)
        layout.addLayout(bottom)
        self.table.itemSelectionChanged.connect(self._update_view_button)

        QShortcut(QKeySequence.Open, self, activated=self.choose_folder)
        QShortcut(QKeySequence("Ctrl+R"), self, activated=self.run_analysis)
        QShortcut(QKeySequence("Ctrl+E"), self, activated=self.export_results)
        QShortcut(QKeySequence.Quit, self, activated=self.close)
        self.statusBar().showMessage("Open a folder of .csv / .nwb / .bin recordings, or try the demo data.")

    # ----------------------------------------------------------------- actions
    def set_directory(self, directory: str | Path):
        self.directory = Path(directory)
        self.folder_label.setText(f"📁 {self.directory.name}")
        self.folder_label.setToolTip(str(self.directory))
        self.run_button.setEnabled(True)

    def choose_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select data folder")
        if folder:
            self.set_directory(folder)
            self.run_analysis()

    def load_demo(self):
        from ripple_analyzer.demo import write_demo_dataset

        target = Path(tempfile.gettempdir()) / "ripple_analyzer_demo"
        self.statusBar().showMessage("Generating demo recordings…")
        QApplication.processEvents()
        write_demo_dataset(target)
        self.set_directory(target)
        self.run_analysis()

    def _detector(self) -> RippleDetector:
        return RippleDetector(threshold_sigma=self.sigma_box.value())

    def run_analysis(self):
        if not self.directory or not self.run_button.isEnabled():
            return
        self.run_button.setEnabled(False)
        self.export_button.setEnabled(False)
        self.table.setRowCount(0)
        self.progress.setValue(0)
        processor = BatchProcessor(self._detector(), fs=float(self.fs_box.value()))

        def work():
            try:
                results = processor.process_directory(
                    self.directory, progress_callback=self.signals.progress.emit
                )
                self.signals.finished.emit(results)
            except Exception as exc:  # reported on the GUI thread
                self.signals.failed.emit(str(exc))

        threading.Thread(target=work, daemon=True).start()

    def _on_progress(self, done: int, total: int, filename: str, status: str):
        self.progress.setMaximum(max(total, 1))
        self.progress.setValue(done)
        self.statusBar().showMessage(f"{filename}: {status}")

    def _on_failed(self, message: str):
        self.run_button.setEnabled(True)
        self.statusBar().showMessage(f"Error: {message}")

    def _on_finished(self, results: list):
        self.results = results
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(results))
        for r, res in enumerate(results):
            ok = res["status"] == "success"
            name = QTableWidgetItem(Path(res["file_path"]).name)
            name.setData(Qt.UserRole, res["file_path"])
            name.setToolTip(res["file_path"])
            status = QTableWidgetItem("OK" if ok else "Error")
            if not ok:
                status.setForeground(Qt.red)
                status.setToolTip(res["error"])
                name.setToolTip(res["error"])
            cells = [
                name,
                _NumericItem(res["duration_sec"], ".1f"),
                _NumericItem(res["fs"], ".0f"),
                _NumericItem(res["n_ripples"] if ok else None, "d"),
                _NumericItem(res["ripples_per_minute"] if ok else None, ".1f"),
                _NumericItem(res["ripple_power_ratio"] if ok else None, ".1f"),
                status,
            ]
            for c, item in enumerate(cells):
                self.table.setItem(r, c, item)
        self.table.setSortingEnabled(True)
        self.table.sortItems(0, Qt.AscendingOrder)

        n_ok = sum(r["status"] == "success" for r in results)
        n_rip = sum(r["n_ripples"] for r in results)
        errors = len(results) - n_ok
        self.summary.setText(
            f"{len(results)} files · {n_rip} ripples"
            + (f" · {errors} could not be read (hover for details)" if errors else "")
        )
        self.progress.setMaximum(max(len(results), 1))
        self.progress.setValue(len(results))
        self.run_button.setEnabled(True)
        self.export_button.setEnabled(bool(results))
        self.statusBar().showMessage(
            "Done. Double-click a recording to inspect it." if results
            else "No supported files (.csv, .nwb, .bin, .dat) found in this folder."
        )

    def _update_view_button(self):
        r = self.table.currentRow()
        self.view_button.setEnabled(r >= 0 and self.table.item(r, 6) is not None
                                    and self.table.item(r, 6).text() == "OK")

    def open_row(self, row: int):
        if row < 0 or self.table.item(row, 6) is None or self.table.item(row, 6).text() != "OK":
            return
        return self.open_file(self.table.item(row, 0).data(Qt.UserRole))

    def open_file(self, path: str) -> Optional[RippleViewer]:
        try:
            signal, fs = FileLoader.load(path, fs=float(self.fs_box.value()))
            viewer = RippleViewer(signal, fs, Path(path).name, self._detector())
        except Exception as exc:
            self.statusBar().showMessage(f"Could not open {Path(path).name}: {exc}")
            return None
        viewer.show()
        self.viewers = [v for v in self.viewers if v.isVisible()] + [viewer]
        return viewer

    def export_results(self):
        if not self.results:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export results", "ripple_results.csv", "CSV (*.csv)")
        if path:
            write_results_csv(self.results, path)
            self.statusBar().showMessage(f"Saved {path}")


def main(directory: str | None = None) -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("RippleAnalyzer")
    window = RippleAnalyzerGUI()
    window.show()
    if directory:
        window.set_directory(directory)
        window.run_analysis()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else None))

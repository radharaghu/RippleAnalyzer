"""Smoke tests for the Qt interface (run headless)."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PyQt5.QtWidgets")


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_viewer_slider_updates_counts(app, recording):
    from ripple_analyzer.ui.visualization import RippleViewer

    signal, centres, fs = recording
    viewer = RippleViewer(signal, fs, "test.csv")
    viewer.slider.setValue(10)
    loose = viewer.result["n_ripples"]
    viewer.slider.setValue(30)
    strict = viewer.result["n_ripples"]
    assert loose > strict >= len(centres) - 1
    assert viewer.stat_labels["count"].text() == str(strict)
    assert viewer.show_stacked_ripples() is not None


def test_main_window_batch_run(app, tmp_path):
    from ripple_analyzer.core.batch_processor import BatchProcessor
    from ripple_analyzer.demo import write_demo_dataset
    from ripple_analyzer.ui.main import RippleAnalyzerGUI

    write_demo_dataset(tmp_path)
    win = RippleAnalyzerGUI()
    win.set_directory(tmp_path)
    win._on_finished(BatchProcessor().process_directory(tmp_path))
    assert win.table.rowCount() == 4
    assert win.open_row(0) is not None

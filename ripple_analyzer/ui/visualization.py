"""Interactive ripple viewer: linked signal panels, live threshold slider,
statistics, threshold-sensitivity curve and peak-aligned ripple stacking."""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.collections import PolyCollection
from matplotlib.figure import Figure
from matplotlib.transforms import blended_transform_factory
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QFontDatabase
from PyQt5.QtWidgets import (
    QFrame, QGridLayout, QHBoxLayout, QLabel, QMainWindow, QPushButton, QSlider,
    QVBoxLayout, QWidget,
)

from ripple_analyzer.core.detector import RippleDetector
from ripple_analyzer.utils.ripple_statistics import RippleStatistics
from ripple_analyzer.utils.waveform_template import WaveformTemplate

RAW = "#3a5a80"
BAND = "#5aa469"
ENV = "#e08e2b"
EVENT = "#d1495b"
MUTED = "#8a94a6"
SIGMAS = np.round(np.arange(1.0, 4.01, 0.1), 2)
ROBUSTNESS_COLORS = {"robust": "#2e8b57", "mixed": "#d4a017", "noise-like": EVENT, "no events": MUTED}


def _style(ax, ylabel=None, xlabel=None):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8, colors="#444")
    ax.grid(True, alpha=0.25, linewidth=0.6)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9)


class RippleViewer(QMainWindow):
    """Inspect one recording. Drag the threshold slider to watch detections,
    statistics and robustness update in real time."""

    def __init__(self, signal, fs, filename: str = "", detector: RippleDetector | None = None):
        super().__init__()
        self.signal = np.asarray(signal, dtype=float)
        self.fs = float(fs)
        self.detector = (detector or RippleDetector()).with_params(fs=self.fs)
        self.filename = filename
        self.time = np.arange(len(self.signal)) / self.fs

        base = self.detector.detect(self.signal)
        self.filtered, self.envelope = base["filtered"], base["envelope"]
        self.sigma = self.detector.threshold_sigma
        self.counts = [
            self.detector.detect_from_envelope(self.envelope, threshold_sigma=s)["n_ripples"]
            for s in SIGMAS
        ]
        self.result = base
        self._child_windows = []

        self.setWindowTitle(f"RippleAnalyzer — {filename}")
        self.resize(1500, 900)
        self._build_ui()
        self._draw_static()
        self._update(self.sigma)

    # ------------------------------------------------------------------ layout
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)

        # Left: linked panels + controls
        left = QVBoxLayout()
        self.fig = Figure(figsize=(11, 7), dpi=100, constrained_layout=True)
        self.ax_raw, self.ax_band, self.ax_evt = self.fig.subplots(3, 1, sharex=True)
        self.canvas = FigureCanvas(self.fig)
        left.addWidget(NavigationToolbar(self.canvas, self))
        left.addWidget(self.canvas, 1)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Detection threshold"))
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(int(SIGMAS[0] * 10), int(SIGMAS[-1] * 10))
        self.slider.setValue(int(round(self.sigma * 10)))
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval(5)
        self.slider.valueChanged.connect(lambda v: self._update(v / 10))
        controls.addWidget(self.slider, 1)
        self.sigma_label = QLabel()
        self.sigma_label.setMinimumWidth(60)
        self.sigma_label.setFont(self._bold(11))
        controls.addWidget(self.sigma_label)
        self.stack_button = QPushButton("Stack ripples")
        self.stack_button.setToolTip("Overlay every detected ripple, aligned on its deepest trough")
        self.stack_button.clicked.connect(self.show_stacked_ripples)
        controls.addWidget(self.stack_button)
        left.addLayout(controls)
        root.addLayout(left, 3)

        # Right: statistics + sensitivity curve
        side = QFrame()
        side.setFrameShape(QFrame.StyledPanel)
        side.setMinimumWidth(330)
        side_l = QVBoxLayout(side)
        title = QLabel("Ripple statistics")
        title.setFont(self._bold(13))
        side_l.addWidget(title)

        grid = QGridLayout()
        mono = QFontDatabase.systemFont(QFontDatabase.FixedFont)
        mono.setPointSize(11)
        self.stat_labels = {}
        rows = [("count", "Ripples"), ("rate", "Rate"), ("duration", "Duration"),
                ("power", "Peak power"), ("ratio", "Event / baseline")]
        for r, (key, text) in enumerate(rows):
            name = QLabel(text)
            name.setStyleSheet("color:#555")
            value = QLabel("—")
            value.setFont(mono)
            value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            grid.addWidget(name, r, 0)
            grid.addWidget(value, r, 1)
            self.stat_labels[key] = value
        side_l.addLayout(grid)

        side_l.addSpacing(10)
        self.robust_label = QLabel()
        self.robust_label.setFont(self._bold(12))
        self.robust_label.setWordWrap(True)
        side_l.addWidget(self.robust_label)
        hint = QLabel("Share of events at this threshold that survive one σ stricter. "
                      "Real ripples stand far above background and survive; noise crossings vanish.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#777; font-size:11px")
        side_l.addWidget(hint)

        self.sens_fig = Figure(figsize=(3.4, 2.8), dpi=100, constrained_layout=True)
        self.ax_sens = self.sens_fig.add_subplot(111)
        self.sens_canvas = FigureCanvas(self.sens_fig)
        side_l.addWidget(self.sens_canvas, 1)
        root.addWidget(side, 1)

    @staticmethod
    def _bold(size):
        f = QFont()
        f.setBold(True)
        f.setPointSize(size)
        return f

    # ---------------------------------------------------------------- drawing
    def _draw_static(self):
        t = self.time
        self.ax_raw.plot(t, self.signal, color=RAW, lw=0.5)
        self.ax_raw.set_title("Raw signal", fontsize=10, loc="left", fontweight="bold")
        _style(self.ax_raw, "µV")

        self.ax_band.plot(t, self.filtered, color=BAND, lw=0.5, alpha=0.8, label="Ripple band")
        self.ax_band.plot(t, self.envelope, color=ENV, lw=1.1, label="Envelope")
        self.thr_line = self.ax_band.axhline(0, color=EVENT, ls="--", lw=1.4, label="Threshold")
        lo, hi = self.detector.ripple_band
        self.ax_band.set_title(f"Ripple band ({lo:g}–{hi:g} Hz) · envelope · threshold",
                               fontsize=10, loc="left", fontweight="bold")
        self.ax_band.legend(loc="upper right", fontsize=7, frameon=False, ncol=3)
        _style(self.ax_band, "µV")

        self.ax_evt.plot(t, self.signal, color=RAW, lw=0.5, alpha=0.35)
        self.ax_evt.set_title("Detected ripples", fontsize=10, loc="left", fontweight="bold")
        self.spans = PolyCollection([], facecolors=EVENT, alpha=0.18, edgecolors="none",
                                    transform=blended_transform_factory(self.ax_evt.transData,
                                                                        self.ax_evt.transAxes))
        self.ax_evt.add_collection(self.spans)
        self.peak_marks = self.ax_evt.scatter([], [], marker="v", s=36, color=EVENT, zorder=5)
        _style(self.ax_evt, "µV", "Time (s)")
        self.ax_evt.set_xlim(t[0], t[-1])

        ax = self.ax_sens
        ax.plot(SIGMAS, self.counts, color=RAW, lw=1.8)
        ax.fill_between(SIGMAS, self.counts, color=RAW, alpha=0.08)
        self.sens_line = ax.axvline(self.sigma, color=EVENT, lw=1.4)
        self.sens_dot, = ax.plot([], [], "o", color=EVENT, ms=6)
        ax.set_yscale("symlog", linthresh=10)
        ax.set_title("Threshold sensitivity", fontsize=10, loc="left", fontweight="bold")
        _style(ax, "Ripples detected", "Threshold (σ)")

    def _update(self, sigma: float):
        self.sigma = float(sigma)
        res = self.detector.detect_from_envelope(self.envelope, self.filtered, self.sigma)
        self.result = res
        bounds = res["event_bounds"]

        self.sigma_label.setText(f"{self.sigma:.1f} σ")
        self.thr_line.set_ydata([res["threshold"]] * 2)
        verts = [[(s / self.fs, 0), (s / self.fs, 1), (e / self.fs, 1), (e / self.fs, 0)]
                 for s, e in bounds]
        self.spans.set_verts(verts)
        peaks = res["peaks"]
        self.peak_marks.set_offsets(
            np.column_stack([peaks / self.fs, self.signal[peaks]]) if len(peaks) else np.empty((0, 2))
        )
        self.ax_evt.set_title(f"Detected ripples (n = {res['n_ripples']})",
                              fontsize=10, loc="left", fontweight="bold")
        self.canvas.draw_idle()

        stats = RippleStatistics.calculate(bounds, self.envelope, self.fs)
        L = self.stat_labels
        L["count"].setText(f"{stats['n_ripples']}")
        L["rate"].setText(f"{stats['ripples_per_minute']:.1f} /min")
        L["duration"].setText(f"{stats['avg_duration_ms']:.0f} ± {stats['std_duration_ms']:.0f} ms")
        L["power"].setText(f"{stats['avg_power']:.3g} µV²")
        L["ratio"].setText(f"{res['power_ratio']:.1f} ×")

        frac, label = RippleStatistics.robustness(self.counts, SIGMAS, self.sigma)
        color = ROBUSTNESS_COLORS[label]
        self.robust_label.setText(f"Robustness: {frac:.0%} — {label}" if label != "no events"
                                  else "Robustness: no events")
        self.robust_label.setStyleSheet(f"color:{color}")

        self.sens_line.set_xdata([self.sigma] * 2)
        self.sens_dot.set_data([self.sigma], [res["n_ripples"]])
        self.sens_canvas.draw_idle()

    # ---------------------------------------------------------------- stacking
    def show_stacked_ripples(self):
        bounds = self.result["event_bounds"]
        if len(bounds) == 0:
            self.statusBar().showMessage("No ripples at this threshold — lower the slider.", 5000)
            return
        centres = WaveformTemplate.align_points(bounds, self.filtered)
        raw = WaveformTemplate.extract(self.signal, centres, self.fs, window_ms=120)
        band = WaveformTemplate.extract(self.filtered, centres, self.fs, window_ms=120)

        fig = Figure(figsize=(10, 4.2), dpi=100, constrained_layout=True)
        axes = fig.subplots(1, 2, sharex=True)
        shown = slice(0, 300)  # keep drawing fast for very large event counts
        for ax, data, title, color in ((axes[0], raw, "Raw LFP", RAW),
                                       (axes[1], band, "Ripple band", BAND)):
            ax.plot(data["times_ms"], data["waveforms"][shown].T, color=color, lw=0.5, alpha=0.18)
            ax.fill_between(data["times_ms"], data["mean"] - data["std"], data["mean"] + data["std"],
                            color=EVENT, alpha=0.18, lw=0)
            ax.plot(data["times_ms"], data["mean"], color=EVENT, lw=2, label="Mean ± SD")
            ax.axvline(0, color=MUTED, lw=0.8, ls=":")
            ax.set_title(title, fontsize=10, loc="left", fontweight="bold")
            _style(ax, "µV", "Time from ripple trough (ms)")
        axes[1].legend(loc="upper right", fontsize=8, frameon=False)
        fig.suptitle(f"{self.filename} · {len(bounds)} ripples at {self.sigma:.1f} σ",
                     fontsize=11, fontweight="bold")

        win = QMainWindow(self)
        win.setWindowTitle(f"Stacked ripples — {self.filename}")
        canvas = FigureCanvas(fig)
        body = QWidget()
        lay = QVBoxLayout(body)
        lay.addWidget(NavigationToolbar(canvas, win))
        lay.addWidget(canvas, 1)
        win.setCentralWidget(body)
        win.resize(1100, 520)
        win.show()
        self._child_windows.append(win)
        return win

import numpy as np

from ripple_analyzer.utils.ripple_statistics import RippleStatistics
from ripple_analyzer.utils.waveform_template import WaveformTemplate


def test_statistics_from_bounds():
    env = np.ones(2000)
    env[100:150] = 3.0
    stats = RippleStatistics.calculate(np.array([[100, 150], [1000, 1100]]), env, fs=1000)
    assert stats["n_ripples"] == 2
    assert stats["avg_duration_ms"] == 75.0
    assert stats["ripples_per_minute"] == 60.0
    assert stats["avg_power"] == (9 + 1) / 2


def test_statistics_with_no_events():
    stats = RippleStatistics.calculate(np.empty((0, 2)), np.ones(100), fs=1000)
    assert stats["n_ripples"] == 0 and stats["ripples_per_minute"] == 0


def test_robustness_labels():
    sigmas = [1, 2, 3, 4]
    assert RippleStatistics.robustness([50, 20, 19, 18], sigmas, 2)[1] == "robust"
    assert RippleStatistics.robustness([50, 20, 2, 0], sigmas, 2)[1] == "noise-like"
    assert RippleStatistics.robustness([0, 0, 0, 0], sigmas, 2)[1] == "no events"


def test_waveform_extraction_pads_edges():
    sig = np.arange(100, dtype=float)
    out = WaveformTemplate.extract(sig, np.array([2, 50]), fs=1000, window_ms=10)
    assert out["waveforms"].shape == (2, 11)
    assert np.isnan(out["waveforms"][0, 0])
    assert out["waveforms"][1, 5] == 50
    assert out["times_ms"][5] == 0

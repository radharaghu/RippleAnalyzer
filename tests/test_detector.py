import numpy as np
import pytest

from ripple_analyzer.core.detector import RippleDetector
from tests.conftest import matched


def test_default_threshold_finds_every_ground_truth_ripple(recording):
    signal, centres, fs = recording
    result = RippleDetector(fs=fs, threshold_sigma=2.0).detect(signal)
    assert matched(result["peaks"] / fs, centres) == len(centres)


def test_strict_threshold_has_no_false_positives(recording):
    signal, centres, fs = recording
    peaks_s = RippleDetector(fs=fs, threshold_sigma=3.0).detect(signal)["peaks"] / fs
    assert all(abs(centres - p).min() < 0.05 for p in peaks_s)
    assert len(peaks_s) >= len(centres) - 1


def test_result_fields_are_consistent(recording):
    signal, _, fs = recording
    r = RippleDetector(fs=fs).detect(signal)
    bounds = r["event_bounds"]
    assert bounds.shape == (r["n_ripples"], 2)
    assert np.all(bounds[:, 1] > bounds[:, 0])
    assert np.all((r["peaks"] >= bounds[:, 0]) & (r["peaks"] < bounds[:, 1]))
    assert np.array_equal(r["events"], bounds[:, 0])
    assert r["power_ratio"] > 3  # ripples carry far more band power than baseline


def test_higher_threshold_never_finds_more(recording):
    signal, _, fs = recording
    d = RippleDetector(fs=fs)
    env = d.detect(signal)["envelope"]
    counts = [d.detect_from_envelope(env, threshold_sigma=s)["n_ripples"] for s in (1, 2, 3, 4)]
    assert counts == sorted(counts, reverse=True)


def test_events_merge_and_respect_minimum_duration():
    d = RippleDetector(fs=1000, min_duration_ms=15, merge_gap_ms=10)
    env = np.zeros(1000)
    env[0:20] = 1       # starts at sample 0 -> must still be found
    env[100:110] = 1    # two 10 ms runs 5 ms apart -> merged to 25 ms
    env[115:130] = 1
    env[500:505] = 1    # too short -> dropped
    assert d.find_events(env, 0.5).tolist() == [[0, 20], [100, 130]]


def test_pure_noise_is_flagged_as_noise_like():
    from ripple_analyzer.utils.ripple_statistics import RippleStatistics

    rng = np.random.default_rng(0)
    d = RippleDetector(fs=2500)
    env = d.detect(rng.standard_normal(2500 * 60))["envelope"]
    sigmas = np.round(np.arange(1, 4.01, 0.1), 2)
    counts = [d.detect_from_envelope(env, threshold_sigma=s)["n_ripples"] for s in sigmas]
    assert RippleStatistics.robustness(counts, sigmas, 2.0)[1] in ("noise-like", "no events")


@pytest.mark.parametrize("kwargs", [dict(fs=0), dict(ripple_band=(200, 130)),
                                    dict(fs=300), dict(threshold_sigma=-1)])
def test_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        RippleDetector(**kwargs)


def test_rejects_bad_signals():
    d = RippleDetector(fs=2500)
    with pytest.raises(ValueError):
        d.detect(np.zeros((10, 10)))
    with pytest.raises(ValueError):
        d.detect(np.zeros(50))
    with pytest.raises(ValueError):
        d.detect(np.full(5000, np.nan))

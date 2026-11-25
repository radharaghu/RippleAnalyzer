import numpy as np
import pytest

from ripple_analyzer.demo import DEMO_FS, synthetic_lfp


@pytest.fixture
def recording():
    """30 s of NREM-like LFP with 12 known ripples."""
    signal, centres = synthetic_lfp(DEMO_FS, 30.0, 12, "nrem", 60.0, 6.0, seed=1)
    return signal, centres, DEMO_FS


def matched(peaks_s, centres_s, tol=0.05):
    return sum(np.any(np.abs(np.asarray(peaks_s) - c) < tol) for c in centres_s)

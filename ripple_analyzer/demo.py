"""Synthetic hippocampal LFP with known ripple times, for demos and tests."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

DEMO_FS = 2500.0  # typical LFP sampling rate

# name: (state, duration_s, n_ripples, ripple_amplitude_uV, noise_uV)
DEMO_RECORDINGS: Dict[str, Tuple[str, float, int, float, float]] = {
    "nrem_sleep": ("nrem", 60.0, 24, 60.0, 6.0),
    "quiet_wake": ("awake", 60.0, 10, 45.0, 8.0),
    "rem_sleep": ("rem", 45.0, 3, 35.0, 8.0),
    "noise_control": ("awake", 30.0, 0, 0.0, 8.0),
}


def _pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    spectrum = np.fft.rfft(rng.standard_normal(n))
    freqs = np.arange(len(spectrum), dtype=float)
    freqs[0] = 1.0
    pink = np.fft.irfft(spectrum / np.sqrt(freqs), n)
    return pink / pink.std()


def synthetic_lfp(
    fs: float = DEMO_FS,
    duration: float = 30.0,
    n_ripples: int = 10,
    state: str = "nrem",
    ripple_amplitude: float = 50.0,
    noise_level: float = 6.0,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(signal_uV, ripple_centres_s)``.

    Background rhythms depend on ``state`` ("nrem": delta + spindles,
    "rem"/"awake": theta). Ripples are 140-190 Hz Gaussian-windowed bursts of
    30-90 ms riding on a negative sharp wave. White noise of ``noise_level`` µV
    sets the ripple-band floor; pink noise and 60 Hz hum add realism.
    """
    rng = np.random.default_rng(seed)
    n = int(duration * fs)
    t = np.arange(n) / fs

    if state == "nrem":
        sig = 120 * np.sin(2 * np.pi * 1.5 * t + rng.uniform(0, 6.3))
        sig += 25 * np.sin(2 * np.pi * 12 * t) * (np.sin(2 * np.pi * 0.2 * t) > 0.6)
    else:
        sig = 80 * np.sin(2 * np.pi * 7.5 * t + 0.5 * np.sin(2 * np.pi * 0.3 * t))
        sig += 10 * np.sin(2 * np.pi * 40 * t + rng.uniform(0, 6.3))
    sig += 40 * _pink_noise(n, rng)
    sig += noise_level * rng.standard_normal(n)
    sig += 3 * np.sin(2 * np.pi * 60 * t)

    centres: List[float] = []
    if n_ripples:
        margin = 0.3
        slots = np.linspace(margin, duration - margin, n_ripples + 1)
        for a, b in zip(slots[:-1], slots[1:]):
            c = rng.uniform(a + 0.1 * (b - a), b - 0.1 * (b - a))
            dur = rng.uniform(0.03, 0.09)
            freq = rng.uniform(140, 190)
            amp = ripple_amplitude * rng.uniform(0.7, 1.3)
            w = np.abs(t - c) < dur * 1.5
            env = np.exp(-0.5 * ((t[w] - c) / (dur / 4)) ** 2)
            sig[w] += amp * env * np.sin(2 * np.pi * freq * (t[w] - c) + rng.uniform(0, 6.3))
            sig[w] -= 3 * amp * np.exp(-0.5 * ((t[w] - c) / (dur / 2)) ** 2)  # sharp wave
            centres.append(c)
    return sig, np.asarray(centres)


def write_demo_dataset(directory: str | Path) -> List[Path]:
    """Write the demo recordings as two-column CSVs (``time_s, lfp_uV``)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, (name, (state, dur, n_rip, amp, noise)) in enumerate(DEMO_RECORDINGS.items()):
        sig, _ = synthetic_lfp(DEMO_FS, dur, n_rip, state, amp, noise, seed=i)
        path = directory / f"{name}.csv"
        t = np.arange(len(sig)) / DEMO_FS
        np.savetxt(path, np.column_stack([t, sig]), delimiter=",", fmt=["%.5f", "%.3f"],
                   header="time_s,lfp_uV", comments="")
        paths.append(path)
    return paths

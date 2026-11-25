"""Extract peak-aligned ripple waveforms and compute a mean template."""

from __future__ import annotations

from typing import Dict

import numpy as np


class WaveformTemplate:
    @staticmethod
    def align_points(event_bounds: np.ndarray, filtered: np.ndarray) -> np.ndarray:
        """Deepest ripple-band trough inside each event (the standard alignment point)."""
        return np.array([s + int(np.argmin(filtered[s:e])) for s, e in event_bounds], dtype=int)

    @staticmethod
    def extract(signal: np.ndarray, centers: np.ndarray, fs: float, window_ms: float = 50.0) -> Dict:
        """Cut ``window_ms`` around each centre. Edge windows are NaN-padded.

        Returns ``waveforms`` (n x samples), ``times_ms`` (relative to centre),
        ``mean`` and ``std`` (NaN-aware).
        """
        signal = np.asarray(signal, dtype=float)
        half = int(round(window_ms / 2 * fs / 1000))
        width = 2 * half + 1
        waves = np.full((len(centers), width), np.nan)
        for i, c in enumerate(np.asarray(centers, dtype=int)):
            lo, hi = c - half, c + half + 1
            src_lo, src_hi = max(lo, 0), min(hi, len(signal))
            waves[i, src_lo - lo: width - (hi - src_hi)] = signal[src_lo:src_hi]
        times = (np.arange(width) - half) / fs * 1000.0
        if len(waves):
            with np.errstate(all="ignore"):
                mean, std = np.nanmean(waves, axis=0), np.nanstd(waves, axis=0)
        else:
            mean = std = np.full(width, np.nan)
        return {"waveforms": waves, "times_ms": times, "mean": mean, "std": std, "n": len(waves)}

"""Per-recording ripple statistics and threshold-sensitivity analysis."""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np


class RippleStatistics:
    """Summaries computed from detected event bounds."""

    @staticmethod
    def calculate(event_bounds: np.ndarray, envelope: np.ndarray, fs: float) -> Dict[str, float]:
        """Count, duration, power and rate for a set of ``[start, end)`` events."""
        bounds = np.asarray(event_bounds, dtype=int).reshape(-1, 2)
        envelope = np.asarray(envelope, dtype=float)
        total_sec = len(envelope) / fs if fs else 0.0
        n = len(bounds)
        if n == 0:
            return dict(n_ripples=0, avg_duration_ms=0.0, std_duration_ms=0.0,
                        avg_power=0.0, std_power=0.0, ripples_per_minute=0.0)

        durations_ms = (bounds[:, 1] - bounds[:, 0]) / fs * 1000.0
        peak_power = np.array([np.max(envelope[s:e]) ** 2 for s, e in bounds])
        return dict(
            n_ripples=int(n),
            avg_duration_ms=float(durations_ms.mean()),
            std_duration_ms=float(durations_ms.std()),
            avg_power=float(peak_power.mean()),
            std_power=float(peak_power.std()),
            ripples_per_minute=float(n / total_sec * 60.0) if total_sec else 0.0,
        )

    @staticmethod
    def robustness(counts: Sequence[int], sigmas: Sequence[float], sigma: float,
                   step: float = 1.0) -> Tuple[float, str]:
        """Fraction of events at ``sigma`` that survive at ``sigma + step``.

        Genuine ripples are much larger than the background, so most of them
        survive a stricter threshold. Noise crossings disappear quickly
        (Rayleigh-distributed envelope: roughly 20% survive one extra sigma).
        """
        sigmas = np.asarray(sigmas, dtype=float)
        counts = np.asarray(counts, dtype=float)
        here = np.interp(sigma, sigmas, counts)
        if here <= 0:
            return 0.0, "no events"
        stricter = np.interp(min(sigma + step, sigmas[-1]), sigmas, counts)
        frac = float(min(stricter / here, 1.0))
        if frac >= 0.7:
            label = "robust"
        elif frac >= 0.4:
            label = "mixed"
        else:
            label = "noise-like"
        return frac, label

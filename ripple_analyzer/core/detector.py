"""Ripple detection: band-pass -> Hilbert envelope -> adaptive threshold -> events."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Tuple

import numpy as np
import scipy.signal


@dataclass
class RippleDetector:
    """Detect sharp-wave ripples (default 130-200 Hz) in a 1-D recording.

    Pipeline
    --------
    1. Zero-phase Butterworth band-pass (``ripple_band``)
    2. Hilbert envelope of the band-passed signal
    3. Adaptive threshold: ``mean(envelope) + threshold_sigma * std(envelope)``
    4. Contiguous supra-threshold runs become events; runs separated by less
       than ``merge_gap_ms`` are merged and events shorter than
       ``min_duration_ms`` are discarded.

    Example
    -------
    >>> detector = RippleDetector(fs=2500)
    >>> result = detector.detect(signal)
    >>> result["n_ripples"], result["event_bounds"][:3]
    """

    fs: float = 30000.0
    ripple_band: Tuple[float, float] = (130.0, 200.0)
    threshold_sigma: float = 2.0
    filter_order: int = 4
    min_duration_ms: float = 15.0
    merge_gap_ms: float = 10.0

    def __post_init__(self) -> None:
        low, high = self.ripple_band
        if self.fs <= 0:
            raise ValueError(f"Sampling rate must be positive, got {self.fs}")
        if not 0 < low < high:
            raise ValueError(f"Band must satisfy 0 < low < high, got {self.ripple_band}")
        if high >= self.fs / 2:
            raise ValueError(
                f"Upper band edge {high} Hz must be below Nyquist ({self.fs / 2:g} Hz)"
            )
        if self.threshold_sigma < 0:
            raise ValueError(f"threshold_sigma must be non-negative, got {self.threshold_sigma}")

    def with_params(self, **changes) -> "RippleDetector":
        """Return a copy with some parameters changed (e.g. ``fs`` for a new file)."""
        return replace(self, **changes)

    # ------------------------------------------------------------------ steps
    def bandpass(self, signal: np.ndarray) -> np.ndarray:
        sos = scipy.signal.butter(
            self.filter_order, self.ripple_band, btype="band", fs=self.fs, output="sos"
        )
        return scipy.signal.sosfiltfilt(sos, signal)

    @staticmethod
    def envelope(filtered: np.ndarray) -> np.ndarray:
        return np.abs(scipy.signal.hilbert(filtered))

    @staticmethod
    def threshold_for(envelope: np.ndarray, sigma: float) -> float:
        return float(np.mean(envelope) + sigma * np.std(envelope))

    def find_events(self, envelope: np.ndarray, threshold: float) -> np.ndarray:
        """Return an ``(n, 2)`` int array of ``[start, end)`` sample bounds."""
        above = np.concatenate(([False], envelope > threshold, [False]))
        edges = np.flatnonzero(np.diff(above.astype(np.int8)))
        bounds = edges.reshape(-1, 2)
        if len(bounds) == 0:
            return np.empty((0, 2), dtype=int)

        max_gap = int(round(self.merge_gap_ms * self.fs / 1000))
        merged = [list(bounds[0])]
        for start, end in bounds[1:]:
            if start - merged[-1][1] <= max_gap:
                merged[-1][1] = end
            else:
                merged.append([start, end])
        bounds = np.asarray(merged, dtype=int)

        min_len = int(round(self.min_duration_ms * self.fs / 1000))
        return bounds[(bounds[:, 1] - bounds[:, 0]) >= max(min_len, 1)]

    # ------------------------------------------------------------- main entry
    def detect(self, signal: np.ndarray, threshold_sigma: float | None = None) -> Dict:
        """Run the full pipeline.

        Returns a dict with ``event_bounds`` (n x 2), ``events`` (start indices),
        ``peaks`` (envelope-peak index per event), ``envelope``, ``filtered``,
        ``threshold``, ``n_ripples``, ``power_ratio`` (mean envelope power inside
        events / outside events), ``ripple_power``, ``baseline_power``,
        ``duration_sec`` and ``ripples_per_minute``.
        """
        signal = np.asarray(signal, dtype=float)
        if signal.ndim != 1:
            raise ValueError(f"Signal must be 1-D, got shape {signal.shape}")
        min_samples = int(0.1 * self.fs)
        if len(signal) < max(min_samples, 100):
            raise ValueError(f"Signal too short ({len(signal)} samples) for reliable detection")
        if not np.all(np.isfinite(signal)):
            raise ValueError("Signal contains NaN or infinite values")

        filtered = self.bandpass(signal)
        envelope = self.envelope(filtered)
        return self.detect_from_envelope(
            envelope,
            filtered=filtered,
            threshold_sigma=self.threshold_sigma if threshold_sigma is None else threshold_sigma,
        )

    def detect_from_envelope(
        self, envelope: np.ndarray, filtered: np.ndarray | None = None, threshold_sigma: float = 2.0
    ) -> Dict:
        """Re-threshold a precomputed envelope (cheap; used by the GUI slider)."""
        threshold = self.threshold_for(envelope, threshold_sigma)
        bounds = self.find_events(envelope, threshold)

        peaks = np.array(
            [s + int(np.argmax(envelope[s:e])) for s, e in bounds], dtype=int
        )
        inside = np.zeros(len(envelope), dtype=bool)
        for s, e in bounds:
            inside[s:e] = True
        power = envelope**2
        ripple_power = float(power[inside].mean()) if inside.any() else 0.0
        baseline_power = float(power[~inside].mean()) if (~inside).any() else 0.0
        power_ratio = ripple_power / baseline_power if baseline_power > 0 else 0.0

        duration_sec = len(envelope) / self.fs
        return {
            "event_bounds": bounds,
            "events": bounds[:, 0].copy(),
            "peaks": peaks,
            "envelope": envelope,
            "filtered": filtered,
            "threshold": threshold,
            "threshold_sigma": threshold_sigma,
            "n_ripples": int(len(bounds)),
            "power_ratio": float(power_ratio),
            "ripple_power": ripple_power,
            "baseline_power": baseline_power,
            "duration_samples": int(len(envelope)),
            "duration_sec": float(duration_sec),
            "ripples_per_minute": float(len(bounds) / duration_sec * 60) if duration_sec else 0.0,
        }

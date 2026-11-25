"""Loaders for NWB, raw binary and CSV recordings.

Every loader returns ``(signal, fs)``: a 1-D float array and the sampling rate in Hz.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import pandas as pd

PathLike = Union[str, Path]
DEFAULT_FS = 30000.0
_TIME_COLUMNS = {"time", "t", "time_s", "times", "timestamp", "timestamps", "seconds"}


def _pick_channel(data: np.ndarray, channel: int) -> np.ndarray:
    data = np.asarray(data)
    if data.ndim == 1:
        return data
    if data.ndim != 2:
        raise ValueError(f"Expected 1-D or 2-D data, got shape {data.shape}")
    # Assume (time, channels); transpose if the long axis is second.
    if data.shape[0] < data.shape[1]:
        data = data.T
    if not 0 <= channel < data.shape[1]:
        raise ValueError(f"Channel {channel} out of range (file has {data.shape[1]} channels)")
    return data[:, channel]


class CSVHandler:
    """CSV with one sample per row.

    * Headerless single-column files are read as-is (``fs`` from the caller).
    * If a column is named ``time``/``time_s``/``timestamp`` the sampling rate
      is inferred from it and the first other column is used as the signal.
    """

    extensions = (".csv",)

    @staticmethod
    def load(path: PathLike, column: int | str | None = None, fs: float = DEFAULT_FS, **_) -> Tuple[np.ndarray, float]:
        first = pd.read_csv(path, header=None, nrows=1).iloc[0]
        has_header = pd.to_numeric(first, errors="coerce").isna().any()
        df = pd.read_csv(path, header=0 if has_header else None)
        if df.empty:
            raise ValueError("CSV file contains no samples")

        names = [str(c).strip().lower() for c in df.columns]
        time_idx = next((i for i, n in enumerate(names) if n in _TIME_COLUMNS), None)
        if time_idx is not None:
            t = pd.to_numeric(df.iloc[:, time_idx], errors="coerce").to_numpy()
            dt = np.nanmedian(np.diff(t))
            if np.isfinite(dt) and dt > 0:
                fs = 1.0 / dt

        if column is None:
            candidates = [i for i in range(df.shape[1]) if i != time_idx]
            if not candidates:
                raise ValueError("CSV has a time column but no signal column")
            col = candidates[0]
        elif isinstance(column, str):
            col = list(df.columns).index(column)
        else:
            col = column

        values = pd.to_numeric(df.iloc[:, col], errors="coerce").to_numpy(dtype=float)
        if np.isnan(values).any():
            raise ValueError(
                f"Column '{df.columns[col]}' contains non-numeric values; "
                "is this a recording (one sample per row)?"
            )
        return values, float(fs)


class BinaryHandler:
    """Raw interleaved binary (e.g. ``.bin``/``.dat``)."""

    extensions = (".bin", ".dat")

    @staticmethod
    def load(
        path: PathLike,
        dtype: str = "int16",
        fs: float = DEFAULT_FS,
        n_channels: int = 1,
        channel: int = 0,
        offset: int = 0,
        **_,
    ) -> Tuple[np.ndarray, float]:
        data = np.fromfile(path, dtype=dtype, offset=offset)
        if n_channels > 1:
            usable = len(data) - len(data) % n_channels
            data = data[:usable].reshape(-1, n_channels)
        return _pick_channel(data, channel).astype(float), float(fs)


class NWBHandler:
    """NWB 2.x: uses the first LFP / ElectricalSeries found in the file."""

    extensions = (".nwb",)

    @staticmethod
    def load(path: PathLike, channel: int = 0, **_) -> Tuple[np.ndarray, float]:
        try:
            from pynwb import NWBHDF5IO
            from pynwb.ecephys import ElectricalSeries
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError("Reading NWB requires pynwb:  pip install pynwb") from exc

        with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
            nwb = io.read()
            series = [o for o in nwb.objects.values() if isinstance(o, ElectricalSeries)]
            if not series:
                from pynwb import TimeSeries
                from hdmf.common import DynamicTable

                found = sorted({
                    o.name for o in nwb.objects.values()
                    if isinstance(o, (TimeSeries, DynamicTable)) and o.name != "electrodes"
                })
                raise ValueError(
                    "No LFP/ElectricalSeries in this NWB file"
                    + (f" (it contains: {', '.join(found[:6])})" if found else "")
                    + ". Ripple detection needs raw voltage data."
                )
            # Prefer series that look like LFP, then the longest.
            series.sort(key=lambda s: ("lfp" not in s.name.lower(), -s.data.shape[0]))
            es = series[0]
            data = es.data[:, channel] if len(es.data.shape) == 2 else es.data[:]
            signal = np.asarray(data, dtype=float)
            if es.conversion and es.conversion != 1.0:
                signal = signal * es.conversion

            if es.rate is not None:
                fs = float(es.rate)
            elif es.timestamps is not None and len(es.timestamps) > 1:
                fs = 1.0 / float(np.median(np.diff(es.timestamps[:1000])))
            else:
                raise ValueError(f"Series '{es.name}' has neither a rate nor timestamps")
        return signal, fs


class FileLoader:
    """Pick a loader by file extension."""

    HANDLERS = (NWBHandler, BinaryHandler, CSVHandler)

    @classmethod
    def load(cls, path: PathLike, **kwargs) -> Tuple[np.ndarray, float]:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        for handler in cls.HANDLERS:
            if path.suffix.lower() in handler.extensions:
                signal, fs = handler.load(path, **kwargs)
                return np.asarray(signal, dtype=float).ravel(), fs
        raise ValueError(f"Unsupported file format: {path.suffix}")

    @classmethod
    def get_supported_formats(cls) -> List[str]:
        return [ext for h in cls.HANDLERS for ext in h.extensions]

    @classmethod
    def is_supported(cls, path: PathLike) -> bool:
        return Path(path).suffix.lower() in cls.get_supported_formats()

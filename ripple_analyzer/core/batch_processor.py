"""Run ripple detection over every supported file in a folder."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional

from ripple_analyzer.core.detector import RippleDetector
from ripple_analyzer.core.file_handlers import FileLoader

logger = logging.getLogger(__name__)

# callback(n_done, n_total, filename, status) with status in {"processing", "complete", "error"}
ProgressCallback = Callable[[int, int, str, str], None]


class BatchProcessor:
    """Detect ripples in many files; one bad file never stops the batch."""

    def __init__(self, detector: Optional[RippleDetector] = None, **loader_kwargs):
        self.detector = detector or RippleDetector()
        self.loader_kwargs = loader_kwargs

    def find_files(self, directory: str | Path, recursive: bool = True) -> List[Path]:
        directory = Path(directory)
        if not directory.is_dir():
            raise ValueError(f"Directory not found: {directory}")
        candidates = directory.rglob("*") if recursive else directory.iterdir()
        return sorted(
            p for p in candidates
            if p.is_file() and not p.name.startswith(".") and FileLoader.is_supported(p)
        )

    def process_directory(
        self,
        directory: str | Path,
        progress_callback: Optional[ProgressCallback] = None,
        recursive: bool = True,
    ) -> List[Dict]:
        files = self.find_files(directory, recursive)
        total = len(files)
        results = []
        for i, path in enumerate(files):
            if progress_callback:
                progress_callback(i, total, path.name, "processing")
            result = self.process_file(path)
            results.append(result)
            if progress_callback:
                status = "complete" if result["status"] == "success" else "error"
                progress_callback(i + 1, total, path.name, status)
        return results

    def process_file(self, path: str | Path) -> Dict:
        path = Path(path)
        try:
            signal, fs = FileLoader.load(path, **self.loader_kwargs)
            res = self.detector.with_params(fs=fs).detect(signal)
            return {
                "file_path": str(path),
                "fs": fs,
                "duration_sec": res["duration_sec"],
                "n_ripples": res["n_ripples"],
                "ripples_per_minute": res["ripples_per_minute"],
                "ripple_power_ratio": res["power_ratio"],
                "status": "success",
                "error": "",
            }
        except Exception as exc:
            logger.debug("Failed to process %s: %s", path, exc)
            return {
                "file_path": str(path),
                "fs": None,
                "duration_sec": None,
                "n_ripples": 0,
                "ripples_per_minute": 0.0,
                "ripple_power_ratio": 0.0,
                "status": "error",
                "error": str(exc),
            }


RESULT_FIELDS = ["file_path", "status", "fs", "duration_sec", "n_ripples",
                 "ripples_per_minute", "ripple_power_ratio", "error"]


def write_results_csv(results: List[Dict], path: str | Path) -> None:
    """Save batch results (one row per file) as CSV."""
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=RESULT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

"""RippleAnalyzer: sharp-wave ripple detection and interactive inspection for LFP recordings."""

__version__ = "0.2.0"

from ripple_analyzer.core.batch_processor import BatchProcessor
from ripple_analyzer.core.detector import RippleDetector
from ripple_analyzer.core.file_handlers import FileLoader

__all__ = ["BatchProcessor", "FileLoader", "RippleDetector", "__version__"]

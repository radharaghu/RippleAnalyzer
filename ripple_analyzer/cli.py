"""Command-line entry point.

    ripple-analyzer                      # open the GUI
    ripple-analyzer gui DATA_DIR         # open the GUI and analyse a folder
    ripple-analyzer detect PATH [-o results.csv] [--sigma 2.5]
    ripple-analyzer demo OUT_DIR         # write synthetic recordings
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ripple_analyzer import __version__
from ripple_analyzer.core.batch_processor import BatchProcessor, write_results_csv
from ripple_analyzer.core.detector import RippleDetector


def _detect(args) -> int:
    detector = RippleDetector(threshold_sigma=args.sigma, ripple_band=(args.low, args.high))
    processor = BatchProcessor(detector, fs=args.fs)
    path = Path(args.path)
    results = processor.process_directory(path) if path.is_dir() else [processor.process_file(path)]
    if not results:
        print(f"No supported files found in {path}", file=sys.stderr)
        return 1

    width = max(len(Path(r["file_path"]).name) for r in results)
    print(f"{'file':<{width}}  {'dur (s)':>8}  {'ripples':>7}  {'per min':>7}  {'evt/base':>8}")
    for r in results:
        name = Path(r["file_path"]).name
        if r["status"] == "success":
            print(f"{name:<{width}}  {r['duration_sec']:8.1f}  {r['n_ripples']:7d}  "
                  f"{r['ripples_per_minute']:7.1f}  {r['ripple_power_ratio']:8.1f}")
        else:
            print(f"{name:<{width}}  error: {r['error']}")

    if args.output:
        write_results_csv(results, args.output)
        print(f"\nSaved {args.output}")
    return 0 if any(r["status"] == "success" for r in results) else 1


def _demo(args) -> int:
    from ripple_analyzer.demo import write_demo_dataset

    for p in write_demo_dataset(args.out_dir):
        print(p)
    return 0


def _gui(args) -> int:
    try:
        from ripple_analyzer.ui.main import main as gui_main
    except ImportError as exc:
        print(f"The GUI needs PyQt5 and matplotlib ({exc}).  pip install PyQt5 matplotlib",
              file=sys.stderr)
        return 1
    return gui_main(getattr(args, "directory", None))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="ripple-analyzer", description=__doc__.split("\n")[0])
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command")

    g = sub.add_parser("gui", help="open the desktop app (default)")
    g.add_argument("directory", nargs="?", help="folder to analyse on start-up")
    g.set_defaults(func=_gui)

    d = sub.add_parser("detect", help="detect ripples in a file or folder")
    d.add_argument("path")
    d.add_argument("-o", "--output", help="write results to this CSV")
    d.add_argument("--sigma", type=float, default=2.0, help="threshold in SDs above mean (default 2)")
    d.add_argument("--low", type=float, default=130.0, help="band low edge, Hz")
    d.add_argument("--high", type=float, default=200.0, help="band high edge, Hz")
    d.add_argument("--fs", type=float, default=30000.0,
                   help="sampling rate for CSV/BIN files without timestamps")
    d.set_defaults(func=_detect)

    m = sub.add_parser("demo", help="write synthetic demo recordings")
    m.add_argument("out_dir")
    m.set_defaults(func=_demo)

    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        return _gui(args)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

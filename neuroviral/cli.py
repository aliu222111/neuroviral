"""`neuroviral` command-line interface.

Usage:
    neuroviral score video.mp4 [--platform tiktok|reels|both]
                               [--type auto|talking_head|edit|vlog]
                               [--json out.json] [--markdown out.md]
                               [--whisper base|small|tiny]
    neuroviral calibrate data.jsonl [--platform tiktok|reels] [--out weights.json]
    neuroviral info
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__


def _cmd_score(args) -> int:
    from .pipeline import analyze

    path = args.video
    if not Path(path).exists():
        print(f"error: file not found: {path}", file=sys.stderr)
        return 2

    platforms = ["tiktok", "reels"] if args.platform == "both" else [args.platform]
    try:
        result = analyze(
            path,
            platforms=platforms,
            content_type=args.type,
            whisper_model=args.whisper,
        )
    except Exception as exc:  # pragma: no cover - defensive top-level
        print(f"error: analysis failed: {exc}", file=sys.stderr)
        return 1

    if args.quiet_terminal:
        pass
    else:
        print(result.to_terminal())

    if args.json:
        Path(args.json).write_text(json.dumps(result.to_dict(), indent=2))
        print(f"\nWrote JSON result -> {args.json}")
    if args.markdown:
        Path(args.markdown).write_text(result.to_markdown())
        print(f"Wrote Markdown report -> {args.markdown}")
    if args.calibration_log:
        rec = result.calibration_record().to_json(indent=None)
        with open(args.calibration_log, "a") as fh:
            fh.write(rec + "\n")
        print(f"Appended calibration record -> {args.calibration_log}")
    return 0


def _cmd_calibrate(args) -> int:
    from .calibration.fit import fit

    platform = None if args.platform == "both" else args.platform
    result = fit(args.data, platform=platform, out_path=args.out)
    if result is None:
        print("No usable labeled data (need >=3 rows with outcomes). "
              "Weights unchanged, still using research-derived defaults.")
        return 0
    print("Re-fit platform weights from labeled data:")
    print(json.dumps(result, indent=2))
    return 0


def _cmd_info(args) -> int:
    from .brain.weights import get_weights

    w = get_weights()
    print(f"NeuroViral {__version__}")
    print(f"  calibrated weights active: {w['_calibrated']}")
    print(f"  content-type profiles: {', '.join(w['content_type_profiles'].keys())}")
    print(f"  platforms: {', '.join(w['platform_weights'].keys())}")
    # dependency availability
    def _has(mod):
        import importlib.util
        return importlib.util.find_spec(mod) is not None
    import shutil
    print("  optional deps:")
    for label, mod in [("librosa", "librosa"), ("faster-whisper", "faster_whisper"),
                       ("scenedetect", "scenedetect"), ("opencv", "cv2"),
                       ("mediapipe", "mediapipe"),
                       ("sentence-transformers", "sentence_transformers"),
                       ("scikit-learn", "sklearn"), ("fastapi", "fastapi")]:
        print(f"    {label:22s} {'available' if _has(mod) else 'MISSING (graceful fallback)'}")
    print(f"    {'ffmpeg':22s} {'available' if shutil.which('ffmpeg') else 'MISSING (required for real media)'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="neuroviral",
        description="Fully-local, research-informed video virality predictor "
                    "(a simulation of brain response, not a real brain scan).",
    )
    p.add_argument("--version", action="version", version=f"neuroviral {__version__}")
    sub = p.add_subparsers(dest="command")

    sc = sub.add_parser("score", help="score a video")
    sc.add_argument("video")
    sc.add_argument("--platform", choices=["tiktok", "reels", "both"], default="both")
    sc.add_argument("--type", choices=["auto", "talking_head", "edit", "vlog"], default="auto")
    sc.add_argument("--json", help="write full JSON result to this path")
    sc.add_argument("--markdown", help="write a Markdown report to this path")
    sc.add_argument("--calibration-log", help="append a calibration record (JSONL)")
    sc.add_argument("--whisper", default="base", help="faster-whisper model size")
    sc.add_argument("--quiet-terminal", action="store_true",
                    help="suppress the terminal report (use with --json)")
    sc.set_defaults(func=_cmd_score)

    ca = sub.add_parser("calibrate", help="re-fit weights from labeled post analytics")
    ca.add_argument("data", help="CSV/JSONL of calibration records with outcomes")
    ca.add_argument("--platform", choices=["tiktok", "reels", "both"], default="both")
    ca.add_argument("--out", help="override output path for calibrated weights JSON")
    ca.set_defaults(func=_cmd_calibrate)

    inf = sub.add_parser("info", help="show version, weights + dependency status")
    inf.set_defaults(func=_cmd_info)
    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

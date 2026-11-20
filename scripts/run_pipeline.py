#!/usr/bin/env python3
"""One-shot autonomous run: ingest -> transcribe -> score hooks -> render -> verify -> cleanup.

Usage:
    python scripts/run_pipeline.py "<youtube-url-or-file>" [--count 5] [--model small] [--hinglish]
                                   [--language auto] [--keep-temp]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import TEMP_DIR  # noqa: E402

HERE = Path(__file__).resolve().parent


def step(*cmd: str) -> None:
    subprocess.run([sys.executable, *cmd], check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source")
    ap.add_argument("--count", type=int, default=5)
    ap.add_argument("--min", type=float, default=30)
    ap.add_argument("--max", type=float, default=58)
    ap.add_argument("--model", default="small")
    ap.add_argument("--language", default="auto")
    ap.add_argument("--hinglish", action="store_true")
    ap.add_argument("--fps", default="auto")
    ap.add_argument("--keep-temp", action="store_true")
    args = ap.parse_args()

    TEMP_DIR.mkdir(exist_ok=True)
    step(str(HERE / "ingest.py"), args.source)
    tr = [str(HERE / "transcribe.py"), str(TEMP_DIR / "source.mp4"), "--model", args.model,
          "--language", args.language] + (["--hinglish"] if args.hinglish else [])
    step(*tr)
    step(str(HERE / "score_hooks.py"), str(TEMP_DIR / "transcript.json"), "--count", str(args.count),
         "--min", str(args.min), "--max", str(args.max))
    mk = [str(HERE / "make_clips.py"), str(TEMP_DIR / "candidates.json"), "--fps", args.fps]
    step(*(mk if args.keep_temp else mk + ["--cleanup"]))


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)

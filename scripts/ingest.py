#!/usr/bin/env python3
"""Ingest a YouTube/stream URL or a local file into ./temp_work/source.mp4.

Usage:
    python scripts/ingest.py <url-or-path> [--out temp_work] [--max-height 1080]

Writes <out>/source.mp4 and <out>/source.json (metadata) and prints the JSON.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import TEMP_DIR, log, media_info, require_bin, save_json  # noqa: E402

# Player-client sets tried in order. The first is the one requested in the spec;
# the rest are fallbacks for 403 / throttling / PO-token failures.
CLIENT_SETS = [
    "youtube:player_client=android,web",
    "youtube:player_client=web_safari,mweb",
    "youtube:player_client=tv,web_embedded",
    "youtube:player_client=default",
]


def is_url(s: str) -> bool:
    return s.startswith(("http://", "https://", "www.", "youtu"))


def download(url: str, out_dir: Path, max_height: int, cookies: str | None, attempts: int,
             live_from_start: bool = False) -> Path:
    target = out_dir / "source.mp4"
    fmt = (
        f"bv*[height<={max_height}][vcodec^=avc1]+ba[ext=m4a]/"
        f"bv*[height<={max_height}]+ba/"
        f"b[height<={max_height}]/b"
    )
    base = [
        sys.executable, "-m", "yt_dlp",
        "--no-playlist", "--no-part", "--force-overwrites",
        "-f", fmt, "--merge-output-format", "mp4",
        "--remux-video", "mp4",
        "--retries", "10", "--fragment-retries", "10", "--extractor-retries", "5",
        "--retry-sleep", "http:exp=1:20", "--retry-sleep", "fragment:exp=1:20",
        "--concurrent-fragments", "4",
        "--socket-timeout", "30",
        "-o", str(out_dir / "source.%(ext)s"),
    ]
    if shutil.which("node"):
        base += ["--js-runtimes", "node"]
    if cookies:
        base += ["--cookies-from-browser", cookies] if "/" not in cookies and not cookies.endswith(".txt") \
            else ["--cookies", cookies]
    if live_from_start:
        # Ongoing livestream: grab from the beginning (blocks until the stream ends).
        base += ["--live-from-start"]

    last_err = None
    for i in range(attempts):
        client = CLIENT_SETS[i % len(CLIENT_SETS)]
        cmd = base + ["--extractor-args", client, url]
        log(f"yt-dlp attempt {i + 1}/{attempts} with {client}")
        proc = subprocess.run(cmd)
        if proc.returncode == 0 and target.exists() and target.stat().st_size > 0:
            return target
        last_err = proc.returncode
        time.sleep(min(2 ** i, 20))
    sys.exit(f"yt-dlp failed after {attempts} attempts (last exit code {last_err})")


def ingest_local(path: Path, out_dir: Path) -> Path:
    if not path.exists():
        sys.exit(f"Input file not found: {path}")
    target = out_dir / "source.mp4"
    if target.exists() or target.is_symlink():
        target.unlink()
    info = media_info(path)
    if path.suffix.lower() in (".mp4", ".m4v", ".mov"):
        # Avoid copying huge files: symlink, the pipeline only reads it.
        target.symlink_to(path.resolve())
    else:
        # Remux (or transcode audio) into mp4 so every later step sees one container.
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(path),
               "-map", "0:v:0", "-c:v", "copy"]
        if info["has_audio"]:
            cmd += ["-map", "0:a:0", "-c:a", "aac", "-b:a", "192k"]
        cmd += ["-movflags", "+faststart", str(target)]
        if subprocess.run(cmd).returncode != 0:
            log("Stream copy failed, transcoding video")
            cmd[cmd.index("copy")] = "libx264"
            subprocess.run(cmd[:-1] + ["-crf", "16", "-preset", "veryfast", str(target)], check=True)
    return target


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="YouTube/stream URL or local video path")
    ap.add_argument("--out", default=str(TEMP_DIR))
    ap.add_argument("--max-height", type=int, default=1080)
    ap.add_argument("--cookies", help="Browser name (e.g. chrome) or cookies.txt path for gated videos")
    ap.add_argument("--attempts", type=int, default=6)
    ap.add_argument("--live-from-start", action="store_true", help="For ongoing livestreams")
    args = ap.parse_args()

    require_bin("ffmpeg")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if is_url(args.source):
        path = download(args.source, out_dir, args.max_height, args.cookies, args.attempts,
                        args.live_from_start)
    else:
        path = ingest_local(Path(args.source).expanduser(), out_dir)

    meta = {"source": args.source, "path": str(path.resolve() if not path.is_symlink() else path.absolute())}
    meta.update(media_info(path))
    save_json(meta, out_dir / "source.json")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

"""Shared helpers for the universal-clipper pipeline."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEMP_DIR = ROOT / "temp_work"
FINAL_DIR = ROOT / "final_clips"
FONTS_DIR = ROOT / "fonts"

OUT_W, OUT_H = 1080, 1920


def log(msg: str) -> None:
    print(f"[clipper] {msg}", file=sys.stderr, flush=True)


def require_bin(name: str) -> str:
    path = shutil.which(name)
    if not path:
        sys.exit(f"Required binary '{name}' not found on PATH")
    return path


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    log("$ " + " ".join(str(c) for c in cmd))
    return subprocess.run(cmd, check=True, **kw)


def ffprobe(path: str | Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout
    return json.loads(out)


def media_info(path: str | Path) -> dict:
    """Return width, height, fps, duration, has_audio for a media file."""
    info = ffprobe(path)
    v = next((s for s in info["streams"] if s.get("codec_type") == "video"), None)
    a = next((s for s in info["streams"] if s.get("codec_type") == "audio"), None)
    if v is None:
        raise ValueError(f"No video stream in {path}")
    num, den = (v.get("avg_frame_rate") or v.get("r_frame_rate") or "30/1").split("/")
    fps = float(num) / float(den) if float(den) else 30.0
    width, height = int(v["width"]), int(v["height"])
    rot = 0
    for sd in v.get("side_data_list", []) or []:
        if "rotation" in sd:
            rot = int(sd["rotation"])
    if abs(rot) in (90, 270):
        width, height = height, width
    duration = float(info["format"].get("duration") or v.get("duration") or 0)
    return {"width": width, "height": height, "fps": fps or 30.0,
            "duration": duration, "has_audio": a is not None}


def slugify(text: str, max_len: int = 40) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    return (text[:max_len].rstrip("-")) or "clip"


def load_json(path: str | Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def words_in_window(transcript: dict, start: float, end: float) -> list[dict]:
    """All words whose midpoint falls in [start, end)."""
    out = []
    for seg in transcript["segments"]:
        if seg["end"] < start or seg["start"] > end:
            continue
        for w in seg.get("words", []):
            mid = (w["start"] + w["end"]) / 2
            if start <= mid < end:
                out.append(w)
    return out

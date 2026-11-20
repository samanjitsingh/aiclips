#!/usr/bin/env python3
"""Sanity check: binaries, ffmpeg filters/encoders, fonts and Python imports inside .venv."""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ok = True


def check(label: str, cond: bool, detail: str = "") -> None:
    global ok
    ok &= bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {label}{(' - ' + detail) if detail else ''}")


def out(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True).stdout
    except FileNotFoundError:
        return ""


check("python >= 3.10", sys.version_info >= (3, 10), sys.version.split()[0])
check("running inside .venv", sys.prefix != sys.base_prefix, sys.prefix)
for b in ("ffmpeg", "ffprobe", "node"):
    check(f"{b} on PATH", shutil.which(b) is not None, shutil.which(b) or "")
check("yt-dlp available", bool(out([sys.executable, "-m", "yt_dlp", "--version"]).strip()),
      out([sys.executable, "-m", "yt_dlp", "--version"]).strip())

filters = out(["ffmpeg", "-hide_banner", "-filters"])
encoders = out(["ffmpeg", "-hide_banner", "-encoders"])
check("ffmpeg filter 'ass'", " ass " in filters)
check("ffmpeg filter 'subtitles'", " subtitles " in filters)
check("ffmpeg encoder libx264", "libx264" in encoders)
check("ffmpeg encoder aac", " aac " in encoders)

fonts = list((ROOT / "fonts").glob("*.ttf"))
check("caption font present", bool(fonts), ", ".join(f.name for f in fonts))

try:
    import faster_whisper
    import ctranslate2
    cuda = ctranslate2.get_cuda_device_count()
    check("import faster_whisper", True,
          f"{faster_whisper.__version__}; CUDA devices={cuda} -> {'cuda/float16' if cuda else 'cpu/int8'}")
except Exception as e:  # noqa: BLE001
    check("import faster_whisper", False, repr(e))
try:
    import mediapipe as mp
    import google.protobuf
    mp.solutions.face_detection.FaceDetection(model_selection=1).close()
    check("import mediapipe + face model", True, f"{mp.__version__} (protobuf {google.protobuf.__version__})")
except Exception as e:  # noqa: BLE001
    check("import mediapipe + face model", False, repr(e))
for mod in ("cv2", "numpy", "pysubs2"):
    try:
        m = __import__(mod)
        check(f"import {mod}", True, getattr(m, "__version__", ""))
    except Exception as e:  # noqa: BLE001
        check(f"import {mod}", False, repr(e))

print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
sys.exit(0 if ok else 1)

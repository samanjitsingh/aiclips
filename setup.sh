#!/usr/bin/env bash
# Non-interactive provisioning for Linux (apt) / macOS (brew). Re-runnable.
set -euo pipefail
cd "$(dirname "$0")"

if [[ "$(uname)" == "Darwin" ]]; then
  command -v brew >/dev/null || { echo "Install Homebrew first: https://brew.sh"; exit 1; }
  brew list python@3.12 >/dev/null 2>&1 || brew install python@3.12
  brew list ffmpeg >/dev/null 2>&1 || brew install ffmpeg   # brew ffmpeg ships libx264 + libass
  command -v node >/dev/null || brew install node
  PY=$(brew --prefix python@3.12)/bin/python3.12
else
  SUDO=$([[ $EUID -eq 0 ]] && echo "" || echo "sudo")
  export DEBIAN_FRONTEND=noninteractive
  $SUDO apt-get update -qq
  $SUDO apt-get install -y -qq ffmpeg python3-venv fonts-noto-core >/dev/null
  command -v node >/dev/null || $SUDO apt-get install -y -qq nodejs >/dev/null
  PY=$(command -v python3.12 || command -v python3.11 || command -v python3.10 || true)
  [[ -n "$PY" ]] || { $SUDO apt-get install -y -qq python3.12 python3.12-venv >/dev/null; PY=python3.12; }
fi

[[ -d .venv ]] || "$PY" -m venv .venv
.venv/bin/python -m pip install -q --upgrade pip
.venv/bin/python -m pip install -q -r requirements.txt
# mediapipe drags in the GUI opencv-contrib build; keep only the headless one (same cv2 module).
.venv/bin/python -m pip uninstall -y -q opencv-contrib-python opencv-python 2>/dev/null || true
.venv/bin/python -m pip install -q --force-reinstall --no-deps opencv-python-headless==4.11.0.86
mkdir -p temp_work final_clips
.venv/bin/python scripts/verify_env.py

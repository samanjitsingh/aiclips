#!/usr/bin/env python3
"""Render a clip plan: smart_crop -> render_captions -> verify, for every clip.

Usage:
    python scripts/make_clips.py temp_work/clips_plan.json [--cleanup]

Plan format (candidates.json from score_hooks.py works as-is):
    {"clips": [{"start": 120.5, "end": 170.2, "title": "why nobody talks about this"}, ...]}

Clip windows are snapped to word boundaries from the transcript so cuts never land
mid-word. Output: final_clips/clip_NN_<slug>.mp4 + final_clips/manifest.json.
With --cleanup, ./temp_work is wiped only if every clip verifies (size > 0, valid
duration, 1080x1920 H.264 + audio).
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FINAL_DIR, OUT_H, OUT_W, TEMP_DIR, ffprobe, load_json, log, save_json, slugify  # noqa: E402

HERE = Path(__file__).resolve().parent
PY = sys.executable


def snap(transcript: dict, start: float, end: float, pad_in=0.12, pad_out=0.35) -> tuple[float, float]:
    words = [w for s in transcript["segments"] for w in s.get("words", [])]
    inside = [w for w in words if w["end"] > start - 0.25 and w["start"] < end + 0.25]
    if not inside:
        return start, end
    first, last = inside[0], inside[-1]
    prev_end = max([w["end"] for w in words if w["end"] <= first["start"]], default=0.0)
    next_start = min([w["start"] for w in words if w["start"] >= last["end"]], default=last["end"] + 5)
    s = max(first["start"] - pad_in, (prev_end + first["start"]) / 2, 0.0)
    e = min(last["end"] + pad_out, (last["end"] + next_start) / 2 if next_start > last["end"] else last["end"] + pad_out)
    return round(s, 3), round(e, 3)


def verify(path: Path, expected: float) -> tuple[bool, str]:
    if not path.exists() or path.stat().st_size == 0:
        return False, "missing or empty"
    try:
        info = ffprobe(path)
    except subprocess.CalledProcessError as e:
        return False, f"ffprobe failed: {e}"
    dur = float(info["format"].get("duration", 0))
    v = next((s for s in info["streams"] if s["codec_type"] == "video"), None)
    a = next((s for s in info["streams"] if s["codec_type"] == "audio"), None)
    problems = []
    if not v or v.get("codec_name") != "h264" or (v["width"], v["height"]) != (OUT_W, OUT_H):
        problems.append(f"bad video stream {v and (v.get('codec_name'), v['width'], v['height'])}")
    if dur <= 0 or abs(dur - expected) > 1.5:
        problems.append(f"duration {dur:.2f}s vs expected {expected:.2f}s")
    if a is None:
        problems.append("no audio")
    return not problems, "; ".join(problems) or f"ok ({dur:.2f}s, {path.stat().st_size / 1e6:.1f} MB)"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("plan")
    ap.add_argument("--source", default=str(TEMP_DIR / "source.mp4"))
    ap.add_argument("--transcript", default=str(TEMP_DIR / "transcript.json"))
    ap.add_argument("--out-dir", default=str(FINAL_DIR))
    ap.add_argument("--fps", default="auto")
    ap.add_argument("--layout", default="auto", choices=["auto", "crop", "pillarbox"])
    ap.add_argument("--cleanup", action="store_true", help="Wipe temp_work after all clips verify")
    ap.add_argument("--no-snap", action="store_true")
    args = ap.parse_args()

    plan = load_json(args.plan)["clips"]
    transcript = load_json(args.transcript)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(exist_ok=True)

    results, all_ok = [], True
    for n, clip in enumerate(plan, 1):
        start, end = float(clip["start"]), float(clip["end"])
        if not args.no_snap:
            start, end = snap(transcript, start, end)
        name = f"clip_{n:02d}_{slugify(clip.get('title') or clip.get('hook') or 'clip')}"
        cropped = TEMP_DIR / f"{name}_cropped.mp4"
        final = out_dir / f"{name}.mp4"
        log(f"=== {name}: {start:.2f}-{end:.2f} ({end - start:.1f}s)")
        try:
            subprocess.run([PY, str(HERE / "smart_crop.py"), args.source, "--start", str(start), "--end", str(end),
                            "--out", str(cropped), "--fps", args.fps, "--layout", args.layout], check=True)
            subprocess.run([PY, str(HERE / "render_captions.py"), "--transcript", args.transcript,
                            "--start", str(start), "--end", str(end), "--video", str(cropped),
                            "--out", str(final)], check=True)
            ok, msg = verify(final, end - start)
        except subprocess.CalledProcessError as e:
            ok, msg = False, f"step failed: {e}"
        all_ok &= ok
        log(f"{'PASS' if ok else 'FAIL'} {final.name}: {msg}")
        results.append({"file": str(final), "start": start, "end": end, "duration": round(end - start, 2),
                        "title": clip.get("title"), "hook": clip.get("hook"), "verified": ok, "check": msg})

    save_json({"clips": results}, out_dir / "manifest.json")
    if args.cleanup:
        if all_ok and results:
            shutil.rmtree(TEMP_DIR, ignore_errors=True)
            TEMP_DIR.mkdir(exist_ok=True)
            (TEMP_DIR / ".gitkeep").touch()
            log("All clips verified; temp_work wiped")
        else:
            log("Not cleaning temp_work: some clips failed verification")
    for r in results:
        print(f"{'OK  ' if r['verified'] else 'FAIL'} {r['file']}  [{r['start']:.1f}-{r['end']:.1f}s]  {r['check']}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()

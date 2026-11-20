#!/usr/bin/env python3
"""Viral word-by-word karaoke captions (.ass) + final 1080x1920 H.264 burn-in.

Usage:
    python scripts/render_captions.py --transcript temp_work/transcript.json \
        --start 120.5 --end 170.2 --video temp_work/clip_01_cropped.mp4 \
        --out final_clips/clip_01_my-hook.mp4

Style: 2-4 words per phrase, Montserrat Black (falls back to Arial Black / any bold sans),
thick black outline + drop shadow, active word in #FFD700, Alignment=2, MarginV=280.
Final encode: libx264 CRF 20, yuv420p, AAC 192k, +faststart.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import pysubs2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FONTS_DIR, OUT_H, OUT_W, load_json, log, words_in_window  # noqa: E402

ACTIVE = r"\1c&H00D7FF&"     # #FFD700 in ASS BGR order
INACTIVE = r"\1c&HFFFFFF&"   # back to white without \r (which would cancel the pop-in)
BREAK_PUNCT = re.compile(r"[.!?,;:।…]$")
STRIP_PUNCT = re.compile(r"^[\"'“”‘’(\[]+|[\"'“”‘’)\],.;:।…]+$")


DEVANAGARI = re.compile(r"[\u0900-\u097F]")
INDIC_FONT = "Noto Sans Devanagari"


def clean(word: str, upper: bool) -> str:
    w = STRIP_PUNCT.sub("", word.strip())
    w = w.replace("{", "(").replace("}", ")").replace("\\", "")
    w = w.upper() if upper else w
    if DEVANAGARI.search(w):
        # Shape the whole word in one Devanagari font; per-glyph fallback breaks matras.
        w = f"{{\\fn{INDIC_FONT}}}{w}{{\\fn}}"
    return w


def vis_len(text: str) -> int:
    return len(re.sub(r"\{[^}]*\}", "", text))


def group_phrases(words: list[dict], max_words: int, max_chars: int, max_gap: float) -> list[list[dict]]:
    phrases, cur = [], []
    for i, w in enumerate(words):
        if cur:
            gap = w["start"] - cur[-1]["end"]
            chars = sum(vis_len(x["text"]) + 1 for x in cur) + vis_len(w["text"])
            if len(cur) >= max_words or gap > max_gap or chars > max_chars or \
                    (len(cur) >= 2 and BREAK_PUNCT.search(cur[-1]["raw"])):
                phrases.append(cur)
                cur = []
        cur.append(w)
    if cur:
        phrases.append(cur)
    # Avoid one-word orphans where we can merge them into a short neighbour.
    merged: list[list[dict]] = []
    for p in phrases:
        if merged and len(p) == 1 and len(merged[-1]) < max_words and \
                p[0]["start"] - merged[-1][-1]["end"] <= max_gap and not BREAK_PUNCT.search(merged[-1][-1]["raw"]):
            merged[-1].extend(p)
        else:
            merged.append(p)
    return merged


def build_ass(words: list[dict], args) -> pysubs2.SSAFile:
    subs = pysubs2.SSAFile()
    subs.info.update({"PlayResX": str(OUT_W), "PlayResY": str(OUT_H), "WrapStyle": "0",
                      "ScaledBorderAndShadow": "yes", "YCbCr Matrix": "TV.709"})
    subs.styles["Karaoke"] = pysubs2.SSAStyle(
        fontname=args.font, fontsize=args.font_size, bold=True,
        primarycolor=pysubs2.Color(255, 255, 255, 0),
        secondarycolor=pysubs2.Color(255, 215, 0, 0),
        outlinecolor=pysubs2.Color(0, 0, 0, 0),
        backcolor=pysubs2.Color(0, 0, 0, 110),
        outline=args.outline, shadow=args.shadow, borderstyle=1,
        alignment=pysubs2.Alignment.BOTTOM_CENTER, marginv=args.margin_v, marginl=70, marginr=70,
        spacing=0.0,  # non-zero letter spacing breaks complex-script (Devanagari) shaping
    )
    phrases = group_phrases(words, args.max_words, args.max_chars, args.max_gap)
    for pi, p in enumerate(phrases):
        next_start = phrases[pi + 1][0]["start"] if pi + 1 < len(phrases) else None
        p_end = p[-1]["end"] + args.hold
        if next_start is not None:
            p_end = min(p_end, next_start)
        for wi, w in enumerate(p):
            s = w["start"]
            e = p[wi + 1]["start"] if wi + 1 < len(p) else p_end
            if e - s < 0.04:
                continue
            parts = [f"{{{ACTIVE}}}{x['text']}{{{INACTIVE}}}" if xi == wi else x["text"] for xi, x in enumerate(p)]
            pop = r"{\fscx85\fscy85\t(0,110,\fscx100\fscy100)}" if wi == 0 else ""
            subs.events.append(pysubs2.SSAEvent(start=int(s * 1000), end=int(e * 1000),
                                                text=pop + " ".join(parts), style="Karaoke"))
    log(f"{len(words)} words -> {len(phrases)} caption phrases")
    return subs


def ff_escape(path: Path) -> str:
    """Escape a path for use inside an ffmpeg filtergraph option value."""
    s = str(path).replace("\\", "/")
    for ch in ("\\", ":", "'", ",", "[", "]", ";"):
        s = s.replace(ch, "\\" + ch)
    return s


def burn(video: Path, ass: Path, out: Path, args) -> None:
    vf = f"ass={ff_escape(ass.resolve())}:fontsdir={ff_escape(FONTS_DIR.resolve())}"
    vf = f"scale={OUT_W}:{OUT_H}:flags=lanczos,setsar=1,{vf}"
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(video), "-vf", vf,
           "-map", "0:v:0", "-map", "0:a:0?",
           "-c:v", "libx264", "-preset", args.preset, "-crf", "20", "-profile:v", "high", "-level", "4.2",
           "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
           "-movflags", "+faststart", str(out)]
    log("$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--transcript", required=True)
    ap.add_argument("--start", type=float, required=True)
    ap.add_argument("--end", type=float, required=True)
    ap.add_argument("--video", help="Cropped 1080x1920 clip to burn into (omit to only write .ass)")
    ap.add_argument("--out", help="Final mp4 path")
    ap.add_argument("--ass", help="Where to write the .ass (default next to --out / --video)")
    ap.add_argument("--font", default="Montserrat Black")
    ap.add_argument("--font-size", type=int, default=86)
    ap.add_argument("--outline", type=float, default=7)
    ap.add_argument("--shadow", type=float, default=4)
    ap.add_argument("--margin-v", type=int, default=280)
    ap.add_argument("--max-words", type=int, default=4)
    ap.add_argument("--max-chars", type=int, default=22)
    ap.add_argument("--max-gap", type=float, default=0.6, help="Pause (s) that forces a new phrase")
    ap.add_argument("--hold", type=float, default=0.3, help="Linger after a phrase's last word")
    ap.add_argument("--no-upper", action="store_true")
    ap.add_argument("--preset", default="medium")
    args = ap.parse_args()

    transcript = load_json(args.transcript)
    raw = words_in_window(transcript, args.start, args.end)
    words = []
    for w in raw:
        text = clean(w["word"], not args.no_upper)
        if not text:
            continue
        words.append({"start": max(0.0, w["start"] - args.start), "end": min(args.end, w["end"]) - args.start,
                      "text": text, "raw": w["word"]})

    subs = build_ass(words, args)
    ass_path = Path(args.ass or Path(args.out or args.video).with_suffix(".ass"))
    if args.out and not args.ass:
        ass_path = Path(args.video).with_suffix(".ass") if args.video else ass_path
    ass_path.parent.mkdir(parents=True, exist_ok=True)
    subs.save(str(ass_path))
    log(f"Wrote {ass_path}")

    if args.video and args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        burn(Path(args.video), ass_path, Path(args.out), args)
        log(f"Final clip -> {args.out}")


if __name__ == "__main__":
    main()

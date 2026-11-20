#!/usr/bin/env python3
"""Word-level transcription with faster-whisper (GPU auto-detect, CPU int8 fallback).

Usage:
    python scripts/transcribe.py temp_work/source.mp4 [--model small] [--language auto|en|hi] [--hinglish]

Outputs:
    temp_work/transcript.json  {language, duration, segments:[{id,start,end,text,words:[{start,end,word,prob}]}]}
    temp_work/transcript.txt   compact "[start-end] text" lines for LLM-based hook selection
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import TEMP_DIR, log, save_json  # noqa: E402

# Romanised Hinglish prompt: nudges Whisper to keep code-switched speech in Latin script
# instead of translating to English or switching to Devanagari.
HINGLISH_PROMPT = (
    "Haan bhai, toh basically main yeh bol raha tha ki this is actually crazy. "
    "Kya scene hai, sach mein? Let's go, chalo shuru karte hain."
)


def pick_device(force: str | None) -> tuple[str, str]:
    if force == "cpu":
        return "cpu", "int8"
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", "float16"
    except Exception as e:  # noqa: BLE001
        log(f"CUDA probe failed ({e}); using CPU")
    if force == "cuda":
        log("CUDA requested but unavailable; falling back to CPU")
    return "cpu", "int8"


def extract_audio(src: Path, dst: Path) -> Path:
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(src),
                    "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(dst)], check=True)
    return dst


def load_model(name: str, device: str, compute: str):
    from faster_whisper import WhisperModel
    try:
        return WhisperModel(name, device=device, compute_type=compute), device
    except Exception as e:  # noqa: BLE001 - e.g. missing cuDNN/cuBLAS libs
        if device == "cpu":
            raise
        log(f"GPU init failed ({e}); falling back to CPU int8")
        return WhisperModel(name, device="cpu", compute_type="int8"), "cpu"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("--model", default="small", help="tiny|base|small|medium|large-v3 (default small)")
    ap.add_argument("--language", default="auto", help="auto, en, hi, ... (default auto)")
    ap.add_argument("--hinglish", action="store_true",
                    help="Code-switched Hindi/English: keep output in Roman script")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--beam-size", type=int, default=5)
    ap.add_argument("--out", default=str(TEMP_DIR / "transcript.json"))
    args = ap.parse_args()

    src = Path(args.input)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    wav = extract_audio(src, out.parent / "audio_16k.wav")

    device, compute = pick_device(None if args.device == "auto" else args.device)
    log(f"Loading faster-whisper '{args.model}' on {device} ({compute})")
    try:
        model, device = load_model(args.model, device, compute)
    except Exception as e:  # noqa: BLE001
        sys.exit(f"Could not load Whisper model '{args.model}': {e}\n"
                 "First run downloads it from huggingface.co - check internet/proxy access, "
                 "or pass a local CTranslate2 model directory via --model.")

    language = None if args.language == "auto" else args.language
    prompt = None
    if args.hinglish:
        # Decoding as 'en' with a romanised prompt keeps Hindi words in Latin script.
        language, prompt = "en", HINGLISH_PROMPT

    t0 = time.time()
    segments, info = model.transcribe(
        str(wav), language=language, initial_prompt=prompt, beam_size=args.beam_size,
        word_timestamps=True, vad_filter=True, vad_parameters={"min_silence_duration_ms": 400},
        condition_on_previous_text=False,
    )

    result = {"language": info.language, "language_probability": round(info.language_probability, 3),
              "duration": round(info.duration, 3), "model": args.model, "device": device,
              "hinglish": args.hinglish, "segments": []}
    lines = []
    for seg in segments:
        words = [{"start": round(w.start, 3), "end": round(w.end, 3), "word": w.word.strip(),
                  "prob": round(w.probability, 3)} for w in (seg.words or []) if w.word.strip()]
        if not words:
            continue
        text = seg.text.strip()
        result["segments"].append({"id": len(result["segments"]), "start": words[0]["start"],
                                   "end": words[-1]["end"], "text": text, "words": words})
        lines.append(f"[{words[0]['start']:.2f}-{words[-1]['end']:.2f}] {text}")
        log(lines[-1])

    save_json(result, out)
    out.with_suffix(".txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"Transcribed {info.duration:.0f}s of audio in {time.time() - t0:.0f}s "
        f"(lang={info.language}, {len(result['segments'])} segments) -> {out}")
    wav.unlink(missing_ok=True)


if __name__ == "__main__":
    main()

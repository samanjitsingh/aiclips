#!/usr/bin/env python3
"""Heuristic viral-segment finder (used for fully autonomous runs, or as a shortlist
for an LLM agent to refine).

Usage:
    python scripts/score_hooks.py temp_work/transcript.json [--count 5] [--min 30] [--max 58]

Builds sentence-aligned windows (start/end on complete sentences), scores them on hook
strength in the first 3 s, emotional/curiosity language, punch (! / ?), speech energy,
and penalises dangling starts and dead air. Writes temp_work/candidates.json.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import TEMP_DIR, load_json, log, save_json  # noqa: E402

END_RE = re.compile(r"[.!?।…]['\"”’)]*$")
HOOK_WORDS = set("""
you your why how what nobody never secret truth mistake biggest worst best insane crazy stop
wait listen imagine actually literally seriously shocking honestly problem reason million
billion money dead died kill scared fear warning mistake hack nobody's everyone everybody
kya kyun kaise kaun sach suno dekho bhai yaar pagal matlab asli galti raaz paisa
क्या क्यों कैसे सच सुनो देखो भाई यार पागल गलती पैसा
""".split())
EMOTION_WORDS = set("""
love hate amazing incredible unbelievable insane crazy wow omg holy hell damn angry cried crying
laugh hilarious funny shocked terrified excited proud regret painful brutal epic legendary
wtf bro dude oh no yes finally impossible
mast zabardast bakwas gussa dar khatarnak kamaal
""".split())
WEAK_STARTS = {"and", "but", "so", "because", "or", "then", "also", "which", "that", "aur", "toh", "lekin", "par"}


def norm(w: str) -> str:
    return re.sub(r"[^\w']", "", w.lower())


def sentences(transcript: dict, max_pause: float) -> list[dict]:
    out, cur = [], []
    words = [w for s in transcript["segments"] for w in s.get("words", [])]
    for i, w in enumerate(words):
        cur.append(w)
        nxt = words[i + 1] if i + 1 < len(words) else None
        if nxt is None or END_RE.search(w["word"]) or nxt["start"] - w["end"] > max_pause:
            out.append({"start": cur[0]["start"], "end": cur[-1]["end"], "words": cur,
                        "text": " ".join(x["word"] for x in cur)})
            cur = []
    return out


def score_window(sents: list[dict]) -> tuple[float, dict]:
    words = [w for s in sents for w in s["words"]]
    t0, t1 = sents[0]["start"], sents[-1]["end"]
    dur = t1 - t0
    hook = [w for w in words if w["start"] < t0 + 3.0]
    hook_tokens = [norm(w["word"]) for w in hook]
    all_tokens = [norm(w["word"]) for w in words]
    hook_text = " ".join(w["word"] for w in hook)

    f = {}
    f["hook_words"] = min(sum(t in HOOK_WORDS for t in hook_tokens), 3) * 1.0
    f["hook_question"] = 1.5 if "?" in hook_text else 0.0
    f["hook_number"] = 0.75 if re.search(r"\d", hook_text) else 0.0
    f["emotion"] = min(sum(t in EMOTION_WORDS for t in all_tokens) * 0.4, 2.5)
    f["punch"] = min(sum(s["text"].rstrip().endswith(("!", "?")) for s in sents) * 0.35, 1.75)
    f["payoff_end"] = 0.5 if sents[-1]["text"].rstrip().endswith(("!", "?")) else 0.0
    wps = len(words) / max(dur, 1)
    f["energy"] = max(min((wps - 2.0) * 0.8, 1.5), -1.0)
    gaps = [b["start"] - a["end"] for a, b in zip(words, words[1:])]
    f["dead_air"] = -min(sum(g for g in gaps if g > 1.2) * 0.3, 2.0)
    f["weak_start"] = -1.0 if all_tokens and all_tokens[0] in WEAK_STARTS else 0.0
    f["low_conf"] = -1.0 if sum(w.get("prob", 1) for w in words) / max(len(words), 1) < 0.6 else 0.0
    return sum(f.values()), f


def title_from(text: str) -> str:
    toks = [t for t in re.findall(r"[\w']+", text) if len(t) > 2][:6]
    return " ".join(toks) or "clip"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("transcript")
    ap.add_argument("--count", type=int, default=5)
    ap.add_argument("--min", type=float, default=30.0)
    ap.add_argument("--max", type=float, default=58.0)
    ap.add_argument("--max-pause", type=float, default=0.9, help="Pause that ends a sentence")
    ap.add_argument("--out", default=str(TEMP_DIR / "candidates.json"))
    args = ap.parse_args()

    sents = sentences(load_json(args.transcript), args.max_pause)
    cands = []
    for i in range(len(sents)):
        for j in range(i, len(sents)):
            dur = sents[j]["end"] - sents[i]["start"]
            if dur > args.max:
                break
            if dur >= args.min:
                sc, feats = score_window(sents[i:j + 1])
                sc += 0.3 * (dur - args.min) / (args.max - args.min)  # mild preference for fuller stories
                cands.append({"start": round(sents[i]["start"], 3), "end": round(sents[j]["end"], 3),
                              "duration": round(dur, 2), "score": round(sc, 3), "features": feats,
                              "hook": " ".join(w["word"] for w in sents[i]["words"] if
                                               w["start"] < sents[i]["start"] + 3),
                              "title": title_from(sents[i]["text"]),
                              "text": " ".join(s["text"] for s in sents[i:j + 1])})
    cands.sort(key=lambda c: c["score"], reverse=True)
    picked = []
    for c in cands:
        if all(c["end"] <= p["start"] or c["start"] >= p["end"] for p in picked):
            picked.append(c)
        if len(picked) >= args.count:
            break
    picked.sort(key=lambda c: c["start"])
    if not picked:
        log("No sentence-aligned window fits the duration limits; source may be too short or silent")
    save_json({"clips": picked}, args.out)
    print(json.dumps([{k: c[k] for k in ("start", "end", "duration", "score", "title", "hook")} for c in picked],
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

---
name: universal-clipper
description: End-to-end autonomous pipeline to ingest YouTube videos, livestreams, or local files, detect high-retention viral hooks, apply smooth AI subject tracking, burn word-by-word karaoke captions, and export 1080x1920 Shorts.
---

# Universal Clipper

Turns any long-form video (VOD, livestream, podcast, gaming, interview) into 3–5 vertical
1080x1920 Shorts with AI subject tracking and viral karaoke captions.

## Deterministic execution rules

- Run every step from the workspace root with the venv interpreter:
  `.venv/bin/python` (Windows: `.venv\Scripts\python.exe`). Never use the system Python.
- Run the steps in order. Do not skip verification. If a step exits non-zero, stop, report
  the stderr, and do not run cleanup.
- Never ask the user to confirm intermediate choices: pick the defaults below.
- All intermediates go in `./temp_work/`; deliverables go only in `./final_clips/`.

## Workflow

### 1. Ingest

```bash
.venv/bin/python scripts/ingest.py "<URL or local path>"
```

- Produces `temp_work/source.mp4` (≤1080p, merged best video + audio) and `temp_work/source.json`.
- yt-dlp uses `--extractor-args "youtube:player_client=android,web"` first, then rotates
  through fallback player clients with exponential back-off on 403/throttling.
- Age-gated or members-only video: add `--cookies chrome` (or a `cookies.txt` path).
- An ongoing livestream: add `--live-from-start`.
- **Long VODs/streams (> ~1 h): never download the whole file.** Download only candidate windows
  (from timestamps the user gives, chapters or comment hot-spots), each with ~30 s of margin:
  `scripts/ingest.py "<URL>" --sections 1:31:40-1:33:15 2:42:15-2:46:00`. This writes
  `temp_work/section_N.mp4`, and `temp_work/sections.json` records each section's `offset` in
  the original timeline. Transcribe each section separately
  (`--out temp_work/section_N.json`).
- "Sign in to confirm you're not a bot" / "The page needs to be reloaded": YouTube is blocking
  this machine's IP. Retry once with `--cookies <browser|cookies.txt>`. If that fails, stop and
  ask the user to run the ingest step on their own machine or provide the file. Never loop on it.

### 2. Transcribe and score for virality

```bash
.venv/bin/python scripts/transcribe.py temp_work/source.mp4 --model small
```

- Language: `--language auto` (default), `en`, or `hi`. For Hinglish (code-switched
  Hindi/English) add `--hinglish`, which keeps Hindi words in Roman script.
- Uses CUDA/float16 when a GPU is present, otherwise CPU/int8 automatically.
- Outputs `temp_work/transcript.json` (segments + word timestamps) and
  `temp_work/transcript.txt` (`[start-end] text` lines).

Then get a heuristic shortlist:

```bash
.venv/bin/python scripts/score_hooks.py temp_work/transcript.json --count 8
```

**Viral selection (agent judgement):** read `temp_work/transcript.txt` and
`temp_work/candidates.json`, then choose **3–5 non-overlapping segments of 30–58 seconds**
that:

1. **Hook in the first 3 seconds**: a bold claim, a question, a number, conflict, "you"-directed
   language, or a pattern interrupt. Never open on filler ("so", "um", "and then").
2. Contain an **emotional peak, curiosity gap, payoff, or punchline** before the end.
3. **Start and end on complete sentences**: the start is a sentence beginning and the end is
   right after a sentence-final word (the renderer also snaps cuts to word boundaries).
4. Stand alone without the surrounding context.

Write the plan to `temp_work/clips_plan.json`:

```json
{"clips": [
  {"start": 312.40, "end": 361.85, "title": "the mistake that cost a million", "hook": "I lost a million dollars in one week"}
]}
```

For section downloads, give each clip its own `source`, `transcript` and `offset` (section start
in seconds). `start`/`end` are then relative to the section, and the manifest reports the
original-timeline `source_start`/`source_end`:

```json
{"clips": [{"start": 31.2, "end": 78.9, "title": "...", "source": "temp_work/section_1.mp4",
            "transcript": "temp_work/section_1.json", "offset": 5500}]}
```

`title` becomes the filename slug (`clip_01_the-mistake-that-cost-a-million.mp4`): keep it short,
lowercase and descriptive. If you cannot judge (no LLM reasoning available), use
`temp_work/candidates.json` directly as the plan.

### 3. Track, caption and render

```bash
.venv/bin/python scripts/make_clips.py temp_work/clips_plan.json
```

For each clip this runs:

Options: `--fps 60` forces 60 fps; `--naming index` names files `clip_1.mp4`, `clip_2.mp4`, ...

Framing is auto-detected per clip and reported as `layout` in the manifest:
- **Split-screen** (streamers/gaming): a face that stays fixed in a frame corner for most of the
  clip is a facecam overlay. The facecam goes in the top 42% and the centre gameplay in the bottom.
- **Speaker-tracking crop** (podcasts, interviews, vlogs, IRL): a smoothed 9:16 crop that
  follows the dominant face, or frames two faces together when they fit.
- **Blurred pillarbox**: no subject for >1.5 s (gameplay with no cam, B-roll, slides).

- `scripts/smart_crop.py`: MediaPipe face detection (pose fallback) at 10 Hz, shot-cut detection,
  dead-zone + zero-phase EMA smoothing so the 9:16 pan never jitters, and a cross-fade to a blurred
  pillarbox when no subject is visible for >1.5 s (gameplay, B-roll, slides). Use `--layout pillarbox`
  to force the full-frame layout (e.g. pure gameplay) or `--layout crop` to force tracking.
- `scripts/render_captions.py`: 2–4-word karaoke phrases in Montserrat Black, thick black outline
  and drop shadow, active word `#FFD700`, `Alignment=2`, `MarginV=280`, burned in at
  1080x1920, H.264 CRF 20, AAC 192k, 30/60 fps (60 when the source is ≥50 fps).

Outputs `final_clips/clip_NN_<slug>.mp4` and `final_clips/manifest.json`.

### 4. Verify and clean up

`make_clips.py` verifies every output (file size > 0, duration matching the window, 1080x1920
H.264 + audio, requested fps, A/V start offset exactly 0 ms, and stream ends within one frame) and prints `OK`/`FAIL` per clip. When all clips print `OK`, wipe the staging area:

```bash
.venv/bin/python scripts/make_clips.py temp_work/clips_plan.json --cleanup
```

(or pass `--cleanup` on the first render run; it only wipes `temp_work/` if every clip verifies).

## Fully autonomous one-liner

Uses the heuristic scorer in place of agent judgement:

```bash
.venv/bin/python scripts/run_pipeline.py "<URL or path>" --count 5 [--hinglish]
```

## Final report to the user

List each file in `final_clips/` with its duration, original-video timestamps
(`source_start`-`source_end` as H:MM:SS), opening hook line, framing `layout`, A/V sync result,
plus a viral title and 5 niche-matched hashtags (from `final_clips/manifest.json`).

# aiclips: universal vertical clipper

Turns long-form video (YouTube VODs, livestreams, podcasts, gaming, interviews) into
1080x1920 Shorts with AI subject tracking and word-by-word karaoke captions.

## Setup

```bash
./setup.sh                                              # Linux / macOS
powershell -ExecutionPolicy Bypass -File setup.ps1      # Windows
```

Requires Python 3.10–3.12 (mediapipe 0.10.21 has no 3.13 wheels). The first transcription
downloads the Whisper model from huggingface.co.

## Run

```bash
.venv/bin/python scripts/run_pipeline.py "https://www.youtube.com/watch?v=..." --count 5
```

Or ask your agent: *"Use the universal-clipper skill on <URL>"*. See
`.agent/skills/universal-clipper/SKILL.md`.

| Script | Purpose |
|---|---|
| `scripts/ingest.py` | yt-dlp (≤1080p, client rotation + retries, `--sections` for long streams) or local file |
| `scripts/transcribe.py` | faster-whisper word timestamps (GPU auto / CPU int8), en / hi / `--hinglish` |
| `scripts/score_hooks.py` | Sentence-aligned 30–58 s candidates ranked by hook/emotion heuristics |
| `scripts/smart_crop.py` | Auto framing: streamer split-screen, smoothed speaker-tracking 9:16 crop, blurred pillarbox |
| `scripts/render_captions.py` | Karaoke `.ass` captions + final H.264 CRF 20 / AAC 192k burn |
| `scripts/make_clips.py` | Renders a clip plan, verifies outputs, optional `--cleanup` |
| `scripts/run_pipeline.py` | All of the above in one command |
| `scripts/verify_env.py` | Environment sanity check |

Fonts: Montserrat Black and Noto Sans Devanagari Bold (SIL OFL, `fonts/OFL.txt`).

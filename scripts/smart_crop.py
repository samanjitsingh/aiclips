#!/usr/bin/env python3
"""AI subject-tracking 9:16 reframer with blurred-pillarbox fallback.

Usage:
    python scripts/smart_crop.py temp_work/source.mp4 --start 120.5 --end 170.2 \
        --out temp_work/clip_01_cropped.mp4 [--fps auto|30|60]

Pass 1 (cheap, 10 Hz @ 640px): MediaPipe face detection (+ pose fallback), shot-cut
detection, dead-zone + zero-phase EMA smoothing of the camera centre.
Pass 2 (full res): sub-pixel crop/scale to 1080x1920 via warpAffine. If no subject is
found for > --no-subject-sec (1.5 s) the frame cross-fades to a blurred-background
pillarbox showing the whole frame. Audio is copied from the source window.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT_H, OUT_W, log, media_info, save_json  # noqa: E402

ANALYSIS_FPS = 10.0
ANALYSIS_W = 640


# --------------------------------------------------------------------------- decode
def ffmpeg_reader(src: str, start: float, dur: float, vf: str, w: int, h: int, pix: str):
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}",
           "-i", src, "-an", "-sn", "-vf", vf, "-f", "rawvideo", "-pix_fmt", pix, "pipe:1"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=w * h * 3 * 4)
    size = w * h * 3
    try:
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    finally:
        proc.stdout.close()
        proc.wait()


# --------------------------------------------------------------------------- analysis
class SubjectDetector:
    def __init__(self, min_face_h: float):
        import mediapipe as mp
        self.face = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.55)
        self.pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=0,
                                           min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.min_face_h = min_face_h

    def faces(self, rgb) -> list[tuple[float, float, float, float]]:
        """Return [(cx, w, h, score)] in normalised coords for faces big enough to track."""
        res = self.face.process(rgb)
        out = []
        for d in res.detections or []:
            bb = d.location_data.relative_bounding_box
            if bb.height < self.min_face_h:
                continue  # tiny face (e.g. gaming facecam) -> not worth reframing on
            cx = min(max(bb.xmin + bb.width / 2, 0.0), 1.0)
            out.append((cx, bb.width, bb.height, float(d.score[0])))
        return out

    def pose_center(self, rgb) -> float | None:
        res = self.pose.process(rgb)
        if not res.pose_landmarks:
            return None
        lm = res.pose_landmarks.landmark
        pts = [lm[i].x for i in (0, 11, 12) if lm[i].visibility > 0.6 and 0 <= lm[i].x <= 1]
        return float(np.mean(pts)) if len(pts) >= 2 else None


def choose_target(faces, prev: float | None, crop_frac: float) -> float:
    """Pick the camera centre: frame a group if it fits, else the dominant face (sticky)."""
    faces = sorted(faces, key=lambda f: f[1] * f[2] * f[3], reverse=True)[:3]
    if len(faces) > 1:
        lo = min(f[0] - f[1] / 2 for f in faces)
        hi = max(f[0] + f[1] / 2 for f in faces)
        if hi - lo < crop_frac * 0.85:
            return (lo + hi) / 2

    def score(f):
        s = f[1] * f[2] * f[3]
        if prev is not None and abs(f[0] - prev) < crop_frac / 2:
            s *= 1.6  # hysteresis: keep the current subject unless another clearly dominates
        return s

    return max(faces, key=score)[0]


def analyse(src: str, start: float, dur: float, info: dict, args) -> dict:
    aw = ANALYSIS_W
    ah = int(round(info["height"] * aw / info["width"] / 2) * 2)
    crop_frac = min(1.0, (info["height"] * 9 / 16) / info["width"])
    det = SubjectDetector(args.min_face_height)

    xs, cuts, prev, prev_hist = [], [], None, None
    vf = f"fps={ANALYSIS_FPS},scale={aw}:{ah}"
    for i, rgb in enumerate(ffmpeg_reader(src, start, dur, vf, aw, ah, "rgb24")):
        small = cv2.resize(rgb, (160, int(160 * ah / aw)))
        hist = cv2.calcHist([cv2.cvtColor(small, cv2.COLOR_RGB2HSV)], [0, 1], None, [32, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        cuts.append(prev_hist is not None and cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL) < args.cut_threshold)
        prev_hist = hist

        faces = det.faces(rgb)
        x = choose_target(faces, None if cuts[-1] else prev, crop_frac) if faces else det.pose_center(rgb)
        xs.append(np.nan if x is None else x)
        if x is not None:
            prev = x
    n = len(xs)
    log(f"Analysed {n} samples: subject found in {np.sum(~np.isnan(xs))}, {sum(cuts)} shot cuts")
    return {"x": np.array(xs, float), "cut": np.array(cuts, bool), "crop_frac": crop_frac}


def smooth_track(x: np.ndarray, cuts: np.ndarray, args) -> tuple[np.ndarray, np.ndarray]:
    """Return (camera_centre, crop_mode) per analysis sample."""
    n = len(x)
    has = ~np.isnan(x)
    mode = np.ones(n, bool)
    # Long subject-less runs -> blurred layout. Short gaps are bridged by holding.
    max_gap = int(round(args.no_subject_sec * ANALYSIS_FPS))
    i = 0
    while i < n:
        if not has[i]:
            j = i
            while j < n and not has[j]:
                j += 1
            if j - i > max_gap:
                mode[i:j] = False
            i = j
        else:
            i += 1
    if not has.any():
        return np.full(n, 0.5), np.zeros(n, bool)

    # Fill gaps with nearest valid value (forward then backward) without crossing shot cuts badly.
    filled = x.copy()
    last = np.nan
    for k in range(n):
        if cuts[k] and not has[k]:
            last = np.nan
        filled[k] = x[k] if has[k] else last
        last = filled[k]
    nxt = np.nan
    for k in range(n - 1, -1, -1):
        if np.isnan(filled[k]):
            filled[k] = nxt
        nxt = filled[k] if not cuts[k] else np.nan
    filled[np.isnan(filled)] = np.nanmean(x)

    # Per-shot: dead-zone follower, then forward-backward EMA (zero phase, no lag / jitter).
    cam = np.empty(n)
    bounds = [0] + [k for k in range(1, n) if cuts[k]] + [n]
    dz, a = args.deadzone, args.smoothing
    for s, e in zip(bounds[:-1], bounds[1:]):
        c = float(np.median(filled[s:min(e, s + 5)]))  # start of shot: snap (cut), don't pan
        seg = np.empty(e - s)
        for k in range(s, e):
            d = filled[k] - c
            if abs(d) > dz:
                c += a * (d - np.sign(d) * dz)
            seg[k - s] = c
        for k in range(1, len(seg)):          # forward EMA
            seg[k] = seg[k - 1] + a * (seg[k] - seg[k - 1])
        for k in range(len(seg) - 2, -1, -1):  # backward EMA
            seg[k] = seg[k + 1] + a * (seg[k] - seg[k + 1])
        cam[s:e] = seg
    return cam, mode


# --------------------------------------------------------------------------- render
class Layouts:
    def __init__(self, w: int, h: int):
        self.w, self.h = w, h
        self.vertical_src = w / h <= OUT_W / OUT_H + 1e-3
        self.cw = min(w, h * OUT_W / OUT_H)          # crop width in source px
        self.scale = OUT_H / h if not self.vertical_src else OUT_W / w
        # Blurred pillarbox geometry
        self.fg_s = min(OUT_W / w, OUT_H / h)
        self.fg_w, self.fg_h = int(round(w * self.fg_s)), int(round(h * self.fg_s))
        self.fg_x, self.fg_y = (OUT_W - self.fg_w) // 2, (OUT_H - self.fg_h) // 2
        self.bg_s = max(OUT_W / w, OUT_H / h)

    def crop(self, frame, cx_norm: float):
        if self.vertical_src:
            return self.pillar(frame)
        x0 = float(np.clip(cx_norm * self.w - self.cw / 2, 0, self.w - self.cw))
        m = np.float32([[self.scale, 0, -x0 * self.scale], [0, self.scale, 0]])
        return cv2.warpAffine(frame, m, (OUT_W, OUT_H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    def pillar(self, frame):
        f = 12
        small = cv2.resize(frame, (max(self.w // f, 2), max(self.h // f, 2)), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), 4)
        s = self.bg_s * f
        tx, ty = (OUT_W - small.shape[1] * s) / 2, (OUT_H - small.shape[0] * s) / 2
        bg = cv2.warpAffine(small, np.float32([[s, 0, tx], [0, s, ty]]), (OUT_W, OUT_H),
                            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        bg = cv2.convertScaleAbs(bg, alpha=0.55)
        fg = cv2.resize(frame, (self.fg_w, self.fg_h), interpolation=cv2.INTER_AREA)
        bg[self.fg_y:self.fg_y + self.fg_h, self.fg_x:self.fg_x + self.fg_w] = fg
        return bg


def render(src: str, start: float, dur: float, info: dict, cam, mode, cuts, out: str, fps: float, args):
    w, h = info["width"], info["height"]
    lay = Layouts(w, h)
    n_s = len(cam)
    t_s = np.arange(n_s) / ANALYSIS_FPS
    ramp = 1.0 / max(args.transition_sec * fps, 1)
    weight = float(mode[0]) if n_s else 0.0

    enc = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{OUT_W}x{OUT_H}", "-r", f"{fps}", "-i", "pipe:0",
           "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", src,
           "-map", "0:v:0", "-map", "1:a:0?",
           "-c:v", "libx264", "-preset", args.preset, "-crf", str(args.crf), "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-shortest", "-movflags", "+faststart", out]
    proc = subprocess.Popen(enc, stdin=subprocess.PIPE)
    cut_times = t_s[cuts] if n_s else np.array([])
    frames = 0
    try:
        for f_idx, frame in enumerate(ffmpeg_reader(src, start, dur, f"fps={fps}", w, h, "bgr24")):
            t = f_idx / fps
            cx = float(np.interp(t, t_s, cam)) if n_s else 0.5
            k = min(int(t * ANALYSIS_FPS), n_s - 1) if n_s else 0
            target = float(mode[k]) if n_s else 0.0
            if cut_times.size and np.any(np.abs(cut_times - t) < 0.5 / fps):
                weight = target  # hard cut in the source -> switch layout on the cut, no fade
            else:
                weight += np.clip(target - weight, -ramp, ramp)
            if weight >= 0.999:
                img = lay.crop(frame, cx)
            elif weight <= 0.001:
                img = lay.pillar(frame)
            else:
                img = cv2.addWeighted(lay.crop(frame, cx), weight, lay.pillar(frame), 1 - weight, 0)
            proc.stdin.write(np.ascontiguousarray(img).tobytes())
            frames += 1
    finally:
        proc.stdin.close()
        rc = proc.wait()
    if rc != 0:
        sys.exit(f"ffmpeg encode failed ({rc})")
    log(f"Rendered {frames} frames @ {fps}fps -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("--start", type=float, required=True)
    ap.add_argument("--end", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", default="auto", help="auto (60 if source>=50fps else 30), 30 or 60")
    ap.add_argument("--no-subject-sec", type=float, default=1.5)
    ap.add_argument("--min-face-height", type=float, default=0.08,
                    help="Ignore faces smaller than this fraction of frame height (facecams)")
    ap.add_argument("--deadzone", type=float, default=0.04, help="Normalised x dead-zone")
    ap.add_argument("--smoothing", type=float, default=0.18, help="EMA alpha at 10 Hz (lower = smoother)")
    ap.add_argument("--transition-sec", type=float, default=0.4)
    ap.add_argument("--cut-threshold", type=float, default=0.55)
    ap.add_argument("--crf", type=int, default=14, help="Intermediate quality (final encode is CRF 20)")
    ap.add_argument("--preset", default="veryfast")
    ap.add_argument("--layout", choices=["auto", "crop", "pillarbox"], default="auto")
    args = ap.parse_args()

    info = media_info(args.input)
    start = max(0.0, args.start)
    end = min(args.end, info["duration"]) if info["duration"] else args.end
    dur = end - start
    if dur <= 0:
        sys.exit("Empty window")
    fps = (60.0 if info["fps"] >= 50 else 30.0) if args.fps == "auto" else float(args.fps)

    if args.layout == "pillarbox":
        n = int(dur * ANALYSIS_FPS) + 1
        a = {"x": np.full(n, np.nan), "cut": np.zeros(n, bool), "crop_frac": 1.0}
        cam, mode = np.full(n, 0.5), np.zeros(n, bool)
    else:
        a = analyse(args.input, start, dur, info, args)
        cam, mode = smooth_track(a["x"], a["cut"], args)
        if args.layout == "crop":
            mode[:] = True
    log(f"Crop-mode coverage: {100 * mode.mean():.0f}% (rest blurred pillarbox)")
    save_json({"start": start, "end": end, "fps": fps, "analysis_fps": ANALYSIS_FPS,
               "raw_x": [None if np.isnan(v) else round(float(v), 4) for v in a["x"]],
               "camera_x": [round(float(v), 4) for v in cam], "crop_mode": mode.astype(int).tolist(),
               "cuts": np.flatnonzero(a["cut"]).tolist()},
              Path(args.out).with_suffix(".track.json"))
    render(args.input, start, dur, info, cam, mode, a["cut"], args.out, fps, args)


if __name__ == "__main__":
    main()

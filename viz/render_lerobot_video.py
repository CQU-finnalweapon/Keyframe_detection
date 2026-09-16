#!/usr/bin/env python3
"""
Render an annotated review video: camera frames + live keypose dashboard.

Plays the dataset videos (``videos/chunk-000/<camera>/episode_XXXXXX.mp4``) while
overlaying, for every frame:

* the active subtask (annotated window, pick/place intent, executing arm, text);
* a prominent banner on the frames where a keypose was detected, naming the event
  (grasp / release / detach) and the arm that performs it;
* a scrolling timeline of subtask spans, detected gripper intervals and keyposes,
  with a playhead;
* live gripper opening / command readouts for both arms.

The camera frames and the parquet rows are index-aligned (both at the dataset fps,
verified on load), so no resampling is involved.

Usage
-----
    # one episode, default camera
    python viz/render_lerobot_video.py --dataset_root /path/to/dataset --episode 0

    # batch
    python viz/render_lerobot_video.py --dataset_root /path/to/dataset --num_episodes 5

    # several camera views side by side (best way to check grasp contact)
    python viz/render_lerobot_video.py --dataset_root /path/to/dataset --episode 0 \
        --cameras cam_high,cam_left_wrist,cam_right_wrist

    # hold each keypose for 15 extra frames so it is easy to see
    python viz/render_lerobot_video.py --dataset_root /path/to/dataset --episode 0 \
        --hold_frames 15
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

from src.lerobot_utils import ARMS, get_profile, load_dataset_meta  # noqa: E402
from visualize_lerobot_episode import process_episode  # noqa: E402

# ---------------------------------------------------------------------------
# Layout constants (canvas is built from these; everything else is derived)
# ---------------------------------------------------------------------------
VIDEO_AREA_W = 960
VIDEO_AREA_H = 720
BANNER_H = 56
PANEL_W = 592
PANEL_EXTRA_H = 208  # room for the full keypose / subtask lists
MARGIN = 16
TIMELINE_H = 126
GAP = 8

CANVAS_W = MARGIN + VIDEO_AREA_W + MARGIN + PANEL_W + MARGIN
VIDEO_Y0 = MARGIN + BANNER_H + GAP
TIMELINE_Y0 = VIDEO_Y0 + VIDEO_AREA_H + MARGIN
CANVAS_H = TIMELINE_Y0 + TIMELINE_H + MARGIN

# Native camera grid.  Resizing the feeds would only burn bitrate, so the canvas
# grows with the number of cameras instead of the feeds being upscaled.
CAM_W, CAM_H = 640, 480


def camera_grid(n_cams: int) -> Tuple[int, int]:
    """Columns/rows for ``n_cams`` feeds: one row when they fit, else 2x2."""
    n = max(1, n_cams)
    if n <= 3:
        return n, 1
    return 2, int(np.ceil(n / 2))


def configure_layout(n_cams: int) -> None:
    """Size the video area / canvas to hold ``n_cams`` native-resolution feeds."""
    global VIDEO_AREA_W, VIDEO_AREA_H, CANVAS_W, CANVAS_H, VIDEO_Y0, TIMELINE_Y0
    cols, rows = camera_grid(n_cams)
    VIDEO_AREA_W = CAM_W * cols
    VIDEO_AREA_H = CAM_H * rows
    VIDEO_Y0 = MARGIN + BANNER_H + GAP
    TIMELINE_Y0 = VIDEO_Y0 + VIDEO_AREA_H + PANEL_EXTRA_H + MARGIN
    CANVAS_W = MARGIN + VIDEO_AREA_W + MARGIN + PANEL_W + MARGIN
    CANVAS_H = TIMELINE_Y0 + TIMELINE_H + MARGIN


configure_layout(1)

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX
CJK_NOTE = "annotation text is shown in English (OpenCV cannot render CJK)"

BG = (28, 28, 32)
PANEL_BG = (42, 42, 48)
FG = (238, 238, 238)
DIM = (168, 168, 176)
ACCENT = (72, 176, 255)

KIND_COLORS = {"pick": (60, 150, 60), "place": (60, 60, 200), "other": (110, 110, 110)}
EVENT_COLORS = {
    "grasp": (80, 200, 80),
    "release": (90, 90, 240),
    "detach": (200, 90, 200),
    "attach_drop": (60, 180, 240),
}
ARM_COLORS = {"left": (235, 130, 40), "right": (60, 60, 235)}


# ---------------------------------------------------------------------------
# small drawing helpers
# ---------------------------------------------------------------------------
def text_size(text: str, scale: float, thickness: int = 1) -> Tuple[int, int]:
    (w, h), base = cv2.getTextSize(text, FONT, scale, thickness)
    return w, h + base


def draw_text(
    img: np.ndarray,
    text: str,
    org: Tuple[int, int],
    scale: float = 0.55,
    color: Tuple[int, int, int] = FG,
    thickness: int = 1,
    font: int = FONT,
) -> int:
    """Draw text at ``org`` (top-left); return the y of the next line."""
    cv2.putText(img, text, (org[0], org[1] + text_size(text, scale)[1] - 6),
                font, scale, color, thickness, cv2.LINE_AA)
    return org[1] + text_size(text, scale, thickness)[1]


def wrap_text(text: str, max_px: int, scale: float) -> List[str]:
    """Greedy word wrap to a pixel width."""
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if text_size(trial, scale)[0] <= max_px or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def truncate_to_px(text: str, max_px: int, scale: float) -> str:
    """Trim ``text`` with an ellipsis so it fits ``max_px`` at ``scale``."""
    if text_size(text, scale)[0] <= max_px:
        return text
    for cut in range(len(text) - 1, 0, -1):
        trial = text[:cut].rstrip() + "..."
        if text_size(trial, scale)[0] <= max_px:
            return trial
    return ""


def fill_rect(img, x0, y0, x1, y1, color, alpha: Optional[float] = None):
    if alpha is None:
        cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), color, -1)
    else:
        overlay = img[int(y0):int(y1), int(x0):int(x1)]
        if overlay.size == 0:
            return
        cv2.addWeighted(np.full_like(overlay, color), alpha, overlay, 1 - alpha, 0, overlay)


def put_label(img, text, x, y, color, scale=0.45, pad=4, text_color=(255, 255, 255)):
    """Filled chip with centred text (used for subtask / keypose markers)."""
    w, h = text_size(text, scale)
    fill_rect(img, x, y, x + w + 2 * pad, y + h + pad, color)
    cv2.putText(img, text, (int(x + pad), int(y + h - 2)), FONT, scale,
                text_color, 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# episode state we need while painting
# ---------------------------------------------------------------------------
class EpisodeRenderState:
    """Pre-computed, frame-indexed view of everything to overlay."""

    def __init__(self, episode: int, rec: Dict[str, Any]):
        self.episode = episode
        self.fps = rec["fps"]
        self.subtasks = rec["subtasks"]
        self.streams = rec["streams"]
        self.timelines = rec["timelines"]
        self.keyposes = rec["keyposes"]
        self.results = rec["results"]
        self.resolved_arms = rec["resolved_arms"]
        self.horizon = next(iter(self.streams.values())).num_frames

        # arm per subtask (use the resolved one so corrections are visible)
        self.subtask_arm = list(self.resolved_arms)
        for i, r in enumerate(self.results):
            if r.arm is not None:
                self.subtask_arm[i] = r.arm
            elif i < len(self.subtask_arm) and self.subtask_arm[i] is None:
                self.subtask_arm[i] = rec["subtasks"][i].arm

        # primary + secondary events in one chronological list
        self.events = sorted(
            [
                {
                    "frame": int(e.frame),
                    "type": e.type,
                    "arm": e.arm,
                    "subtask": e.subtask_index,
                    "primary": bool(e.is_primary),
                    "interval": e.interval,
                }
                for e in rec["events"]
            ],
            key=lambda e: (e["frame"], 0 if e["primary"] else 1),
        )
        self.kp_frames = [int(f) for f in self.keyposes["frames"]]
        self.kp_types = list(self.keyposes["event_types"])
        self.kp_arms = list(self.keyposes["arm"])

    # ---- lookups ------------------------------------------------------
    def subtask_at(self, frame: int) -> Optional[int]:
        for i, s in enumerate(self.subtasks):
            if s.start_frame <= frame < s.end_frame:
                return i
        return None

    def event_at(self, frame: int, radius: int = 0) -> Optional[Dict[str, Any]]:
        best = None
        for e in self.events:
            if abs(e["frame"] - frame) <= radius:
                if best is None or abs(e["frame"] - frame) < abs(best["frame"] - frame):
                    best = e
        return best

    def last_event_before(self, frame: int) -> Optional[Dict[str, Any]]:
        prev = None
        for e in self.events:
            if e["frame"] <= frame:
                prev = e
            else:
                break
        return prev


# ---------------------------------------------------------------------------
# panels
# ---------------------------------------------------------------------------
def paste_videos(
    canvas: np.ndarray,
    frame: np.ndarray,
    cam_index: int,
    n_cams: int,
) -> None:
    """Scale and place camera ``frame`` in the video area (grid, aspect preserved)."""
    cols, rows = camera_grid(n_cams)
    cell_w = VIDEO_AREA_W // cols
    cell_h = VIDEO_AREA_H // rows
    # keep the source aspect inside the cell
    scale = min(cell_w / frame.shape[1], cell_h / frame.shape[0])
    w, h = int(frame.shape[1] * scale), int(frame.shape[0] * scale)

    r, c = divmod(cam_index, cols)
    x0 = MARGIN + c * cell_w + (cell_w - w) // 2
    y0 = VIDEO_Y0 + r * cell_h + (cell_h - h) // 2
    canvas[y0:y0 + h, x0:x0 + w] = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)


def draw_keyframe_banner(
    canvas: np.ndarray,
    state: EpisodeRenderState,
    frame: int,
    hold_radius: int,
    camera_label: str = "",
) -> Optional[Dict[str, Any]]:
    """Dedicated strip above the video announcing the active keypose.

    Lives outside the video area so it never hides the scene, and is drawn on
    every frame (idle state included) so the layout does not jump around.
    """
    x0, y0 = MARGIN, MARGIN
    x1, y1 = MARGIN + VIDEO_AREA_W, MARGIN + BANNER_H

    ev = state.event_at(frame, radius=hold_radius) if hold_radius > 0 else None
    on_kf = ev is not None and ev["frame"] == frame
    color = EVENT_COLORS.get(ev["type"], ACCENT) if ev is not None else (60, 60, 68)

    fill_rect(canvas, x0, y0, x1, y1, color if on_kf else PANEL_BG)
    fill_rect(canvas, x0, y0, x0 + 8, y1, color)

    if ev is not None:
        tag = "KEYPOSE" if ev["primary"] else "secondary"
        label = f"{tag}: {ev['type'].upper()}   ({ev['arm']} arm)"
        scale = 0.95
        tw = text_size(label, scale, 2)[0]
        cv2.putText(canvas, label, (x0 + 24, y0 + 38), FONT_BOLD, scale,
                    (255, 255, 255) if on_kf else color, 2, cv2.LINE_AA)
        sub = (f"frame {ev['frame']}  ({ev['frame'] - frame:+d})   "
               f"subtask #{ev['subtask']}   "
               f"{'detected here' if on_kf else 'approaching'}")
        cv2.putText(canvas, sub, (x0 + 26, y0 + 52), FONT, 0.45,
                    (245, 245, 245) if on_kf else DIM, 1, cv2.LINE_AA)
    else:
        cv2.putText(canvas, "no keypose nearby", (x0 + 24, y0 + 34), FONT_BOLD, 0.6,
                    DIM, 1, cv2.LINE_AA)

    if camera_label:
        w = text_size(camera_label, 0.45)[0]
        cv2.putText(canvas, camera_label, (x1 - w - 16, y0 + 34), FONT, 0.45,
                    (225, 225, 230), 1, cv2.LINE_AA)

    if ev is not None:
        cv2.rectangle(canvas, (MARGIN + 1, VIDEO_Y0 + 1),
                      (MARGIN + VIDEO_AREA_W - 2, VIDEO_Y0 + VIDEO_AREA_H - 2),
                      color, 3 if on_kf else 1)
    return ev


def draw_gripper_zoom(
    canvas: np.ndarray,
    state: EpisodeRenderState,
    frame: int,
    span_seconds: float = 6.0,
) -> None:
    """Zoomed gripper trace around the current frame.

    This is the direct answer to "is the detected keypose really where the gripper
    acts?": the trace is drawn at the resolution where the individual ramp frames
    are visible, with the detected interval shaded and the keyposes marked.
    """
    x0 = MARGIN
    y0 = VIDEO_Y0 + VIDEO_AREA_H + GAP
    x1 = MARGIN + VIDEO_AREA_W
    y1 = y0 + PANEL_EXTRA_H - GAP
    fill_rect(canvas, x0, y0, x1, y1, PANEL_BG)
    pad = 14
    fps = state.fps
    half = max(10, int(span_seconds * fps / 2))
    f0 = max(0, frame - half)
    f1 = min(state.horizon, frame + half)
    if f1 - f0 < 2:
        return

    inner_x0, inner_x1 = x0 + pad + 26, x1 - pad
    n_arms = max(1, len(state.streams))
    band_h = (y1 - y0 - 2 * pad - 18) / n_arms

    cv2.putText(canvas, f"GRIPPER ZOOM   +/-{span_seconds / 2:.1f} s",
                (x0 + pad, y0 + 14), FONT_BOLD, 0.45, DIM, 1, cv2.LINE_AA)

    for ai, arm in enumerate(sorted(state.streams.keys())):
        st = state.streams[arm]
        g = st.gripper_state
        lo, hi = float(g.min()), float(g.max())
        rng = hi - lo if hi - lo > 1e-9 else 1.0
        by0 = y0 + pad + 18 + ai * band_h
        by1 = by0 + band_h - 6

        # detected intervals inside the window
        for iv in state.timelines.get(arm, {}).get("gripper", []):
            a, b = max(iv.start, f0), min(iv.end, f1)
            if b <= a:
                continue
            color = EVENT_COLORS["grasp"] if iv.label == "closing" else EVENT_COLORS["release"]
            px0 = inner_x0 + (inner_x1 - inner_x0) * (a - f0) / (f1 - f0)
            px1 = inner_x0 + (inner_x1 - inner_x0) * (b - f0) / (f1 - f0)
            fill_rect(canvas, px0, by0, max(px1, px0 + 1), by1, color, 0.28)

        # the trace itself, one sample per frame
        idx = np.arange(f0, f1)
        vals = (g[idx] - lo) / rng
        xs = inner_x0 + (inner_x1 - inner_x0) * (idx - f0) / (f1 - f0)
        ys = by1 - vals * (by1 - by0)
        pts = np.stack([xs, ys], axis=1).astype(np.int32)
        cv2.polylines(canvas, [pts], False, ARM_COLORS.get(arm, FG), 2, cv2.LINE_AA)

        # keyposes on this arm
        for f, etype, earm in zip(state.kp_frames, state.kp_types, state.kp_arms):
            if earm != arm or not (f0 <= f <= f1):
                continue
            px = inner_x0 + (inner_x1 - inner_x0) * (f - f0) / (f1 - f0)
            color = EVENT_COLORS.get(etype, FG)
            cv2.line(canvas, (int(px), int(by0)), (int(px), int(by1)), color, 2)
            cv2.circle(canvas, (int(px), int(by1)), 3, color, -1)

        cv2.putText(canvas, f"{arm[:1].upper()}", (x0 + pad, int((by0 + by1) / 2) + 5),
                    FONT_BOLD, 0.5, ARM_COLORS.get(arm, FG), 1, cv2.LINE_AA)

    # playhead + time ticks
    px = inner_x0 + (inner_x1 - inner_x0) * (frame - f0) / (f1 - f0)
    cv2.line(canvas, (int(px), y0 + 18), (int(px), y1 - pad + 4), (255, 255, 255), 2)
    for k in range(0, 5):
        f = f0 + int((f1 - f0) * k / 4)
        tx = inner_x0 + (inner_x1 - inner_x0) * (f - f0) / (f1 - f0)
        cv2.putText(canvas, f"{f / fps:.1f}s", (int(tx) - 14, y1 - 2),
                    FONT, 0.36, DIM, 1, cv2.LINE_AA)


def draw_dashboard(
    canvas: np.ndarray,
    state: EpisodeRenderState,
    frame: int,
    hold_radius: int,
) -> None:
    x0 = MARGIN + VIDEO_AREA_W + MARGIN
    y0 = MARGIN
    x1 = x0 + PANEL_W
    y1 = VIDEO_Y0 + VIDEO_AREA_H + PANEL_EXTRA_H - GAP
    fill_rect(canvas, x0, y0, x1, y1, PANEL_BG)
    pad = 14

    # ---- header -------------------------------------------------------
    x = x0 + pad
    y = y0 + pad
    y = draw_text(canvas, f"EPISODE {state.episode}", (x, y), 0.75, FG, 2, FONT_BOLD)
    y += 2
    y = draw_text(canvas, f"{state.fps:.1f} fps   frame {frame} / {state.horizon}"
                          f"   t = {frame / state.fps:6.2f} s",
                  (x, y), 0.5, DIM)
    y += 10
    cv2.line(canvas, (x, y), (x1 - pad, y), (80, 80, 88), 1)
    y += 14

    # ---- active subtask ----------------------------------------------
    y = draw_text(canvas, "ACTIVE SUBTASK", (x, y), 0.45, DIM, 1, FONT_BOLD)
    y += 4
    si = state.subtask_at(frame)
    if si is None:
        y = draw_text(canvas, "(outside every annotated window)", (x, y), 0.5, DIM)
    else:
        sub = state.subtasks[si]
        arm = state.subtask_arm[si]
        chip = f"#{si}  {sub.kind.upper()}   arm={arm}"
        put_label(canvas, chip, x, y, KIND_COLORS.get(sub.kind, KIND_COLORS["other"]), 0.5)
        y += text_size(chip, 0.5)[1] + 10
        y = draw_text(canvas, f"window [{sub.start_frame}, {sub.end_frame})"
                              f"   ({(sub.end_frame - sub.start_frame) / state.fps:.1f} s)",
                      (x, y), 0.45, DIM)
        y += 2
        body = sub.action_english or sub.action or "(no text)"
        for line in wrap_text(body, PANEL_W - 2 * pad, 0.5)[:3]:
            y = draw_text(canvas, line, (x, y), 0.5, FG)
    y += 16

    # ---- keyframe status ---------------------------------------------
    y = draw_text(canvas, "KEYFRAME STATUS", (x, y), 0.45, DIM, 1, FONT_BOLD)
    y += 6
    ev = state.event_at(frame, radius=hold_radius) if hold_radius > 0 else None
    if ev is not None and ev["frame"] == frame:
        msg, col = f"ON KEYFRAME: {ev['type'].upper()} ({ev['arm']})", EVENT_COLORS.get(ev["type"], ACCENT)
    elif ev is not None:
        msg, col = f"{abs(ev['frame'] - frame)} frames from {ev['type']} ({ev['arm']})", DIM
    else:
        msg, col = "between keyposes", DIM
    w, h = text_size(msg, 0.55)
    fill_rect(canvas, x, y, x + w + 16, y + h + 10, col if ev is not None and ev["frame"] == frame else (58, 58, 66))
    cv2.putText(canvas, msg, (x + 8, y + h + 2), FONT_BOLD, 0.55,
                (255, 255, 255) if ev is not None and ev["frame"] == frame else FG, 1, cv2.LINE_AA)
    y += h + 20

    last = state.last_event_before(frame)
    if last is not None:
        y = draw_text(canvas, f"last event: {last['type']} ({last['arm']}) "
                              f"@ frame {last['frame']}  ({frame - last['frame']} frames ago)",
                      (x, y), 0.45, DIM)
    y += 16

    # ---- gripper readouts --------------------------------------------
    y = draw_text(canvas, "GRIPPER", (x, y), 0.45, DIM, 1, FONT_BOLD)
    y += 6
    bar_w = PANEL_W - 2 * pad - 110
    for arm in sorted(state.streams.keys()):
        st = state.streams[arm]
        idx = min(max(frame, 0), st.num_frames - 1)
        g, cmd = float(st.gripper_state[idx]), float(st.gripper_cmd[idx])
        lo, hi = float(st.gripper_state.min()), float(st.gripper_state.max())
        frac = (g - lo) / (hi - lo) if hi - lo > 1e-9 else 0.0
        cv2.putText(canvas, f"{arm[:1].upper()}", (x, y + 14), FONT_BOLD, 0.5,
                    ARM_COLORS.get(arm, FG), 1, cv2.LINE_AA)
        bx = x + 22
        fill_rect(canvas, bx, y + 2, bx + bar_w, y + 18, (70, 70, 78))
        fill_rect(canvas, bx, y + 2, bx + int(bar_w * frac), y + 18,
                  ARM_COLORS.get(arm, FG), 0.85)
        state_txt = "OPEN" if frac > 0.5 else "CLOSED"
        cv2.putText(canvas, f"{g:.4f} {state_txt}", (bx + bar_w + 8, y + 15),
                    FONT, 0.4, FG, 1, cv2.LINE_AA)
        # command marker
        cfrac = (cmd - lo) / (hi - lo) if hi - lo > 1e-9 else 0.0
        cx = bx + int(np.clip(cfrac, 0, 1) * bar_w)
        cv2.line(canvas, (cx, y), (cx, y + 20), (255, 255, 255), 1)
        y += 26
    y += 10

    # ---- keypose list, then the full subtask list in the remaining space ---
    y = draw_text(canvas, f"KEYPOSES ({len(state.kp_frames)})", (x, y), 0.45, DIM, 1, FONT_BOLD)
    y += 4
    row_h = 17
    cur = len([f for f in state.kp_frames if f <= frame])
    max_rows = max(3, (y1 - pad - y - 190) // row_h)
    start = max(0, min(cur - max_rows // 2, len(state.kp_frames) - max_rows))
    shown = state.kp_frames[start:start + max_rows]
    for k, f in enumerate(shown):
        i = start + k
        etype, earm = state.kp_types[i], state.kp_arms[i]
        is_now = f == frame
        is_past = f <= frame
        color = EVENT_COLORS.get(etype, FG) if is_now else (FG if is_past else DIM)
        line = (f"{i:2d}  {etype:8s} {earm:5s}  f={f:5d}  t={f / state.fps:6.2f}")
        if is_now:
            fill_rect(canvas, x - 6, y - 3, x1 - pad, y + row_h - 5, color, 0.35)
        cv2.putText(canvas, line, (x, y + row_h - 6), FONT, 0.43, color, 1, cv2.LINE_AA)
        y += row_h
    y += 12

    # ---- subtask list -------------------------------------------------
    y = draw_text(canvas, f"SUBTASKS ({len(state.subtasks)})", (x, y), 0.45, DIM, 1, FONT_BOLD)
    y += 4
    current = state.subtask_at(frame)
    row_h = 17
    max_rows = max(0, (y1 - pad - y) // row_h)
    first = max(0, min((current if current is not None else 0) - max_rows // 2,
                       max(0, len(state.subtasks) - max_rows)))
    for i in range(first, min(len(state.subtasks), first + max_rows)):
        sub = state.subtasks[i]
        arm = state.subtask_arm[i]
        is_now = i == current
        color = (FG if is_now else DIM) if i <= (current if current is not None else -1) else (120, 120, 128)
        tag = f"{i:2d} {sub.kind[:5]:5s} {str(arm)[:1].upper()}"
        cv2.putText(canvas, tag, (x, y + row_h - 5), FONT_BOLD, 0.42,
                    KIND_COLORS.get(sub.kind, KIND_COLORS["other"]) if is_now else color,
                    1, cv2.LINE_AA)
        tag_w = text_size(tag, 0.42)[0]
        body = truncate_to_px(sub.action_english or sub.action or "",
                              x1 - pad - (x + tag_w + 12), 0.4)
        cv2.putText(canvas, body, (x + tag_w + 12, y + row_h - 5), FONT, 0.4, color, 1, cv2.LINE_AA)
        if is_now:
            fill_rect(canvas, x - 6, y - 2, x1 - pad, y + row_h - 4, KIND_COLORS.get(sub.kind, FG), 0.22)
        y += row_h


def draw_timeline(
    canvas: np.ndarray,
    state: EpisodeRenderState,
    frame: int,
    show_secondary: bool = True,
    header: str = "",
) -> None:
    x0, y0 = MARGIN, TIMELINE_Y0
    x1, y1 = CANVAS_W - MARGIN, y0 + TIMELINE_H
    fill_rect(canvas, x0, y0, x1, y1, PANEL_BG)
    pad = 10
    gutter = 58  # room for the left-hand row labels
    inner_x0, inner_x1 = x0 + gutter, x1 - pad
    span = max(1, state.horizon)

    def fx(f: int) -> int:
        return int(inner_x0 + (inner_x1 - inner_x0) * min(max(f, 0), span) / span)

    lx = x0 + pad
    if header:
        cv2.putText(canvas, header, (lx, y0 + 15), FONT, 0.42, DIM, 1, cv2.LINE_AA)

    # ---- subtask band ------------------------------------------------
    top = y0 + 24
    row_sub_h = 15
    for i, s in enumerate(state.subtasks):
        bx0, bx1 = fx(s.start_frame), fx(s.end_frame)
        is_now = state.subtask_at(frame) == i
        fill_rect(canvas, bx0, top, max(bx1, bx0 + 1), top + row_sub_h,
                  KIND_COLORS.get(s.kind, KIND_COLORS["other"]),
                  alpha=1.0 if is_now else 0.5)
        if bx1 - bx0 > 26:
            label = f"{i}{s.kind[:1].upper()}"
            w = text_size(label, 0.34)[0]
            cv2.putText(canvas, label, ((bx0 + bx1) // 2 - w // 2, top + row_sub_h - 4),
                        FONT, 0.34, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, "subtasks", (lx, top + row_sub_h - 4), FONT, 0.4, DIM, 1, cv2.LINE_AA)

    # ---- gripper intervals, one row per arm --------------------------
    g_y = top + row_sub_h + 6
    arm_keys = sorted(state.timelines.keys())
    per_arm_h = 11
    for ai, arm in enumerate(arm_keys):
        ay = g_y + ai * per_arm_h
        for iv in state.timelines[arm].get("gripper", []):
            if iv.end <= 0 or iv.start >= span:
                continue
            color = EVENT_COLORS["grasp"] if iv.label == "closing" else EVENT_COLORS["release"]
            fill_rect(canvas, fx(iv.start), ay + 1, max(fx(iv.end), fx(iv.start) + 1),
                      ay + per_arm_h - 2, color, 0.92)
        cv2.putText(canvas, f"{arm[:1].upper()} grip", (lx, ay + per_arm_h - 4),
                    FONT, 0.4, ARM_COLORS.get(arm, DIM), 1, cv2.LINE_AA)

    # ---- keypose ticks ------------------------------------------------
    kp_y = g_y + per_arm_h * max(1, len(arm_keys)) + 4
    for f, etype in zip(state.kp_frames, state.kp_types):
        px = fx(f)
        cv2.line(canvas, (px, kp_y), (px, kp_y + 16), EVENT_COLORS.get(etype, FG), 2)
    if show_secondary:
        for e in state.events:
            if e["primary"]:
                continue
            px = fx(e["frame"])
            cv2.line(canvas, (px, kp_y + 12), (px, kp_y + 18),
                     EVENT_COLORS.get(e["type"], DIM), 1)
    cv2.putText(canvas, "keyposes", (lx, kp_y + 13), FONT, 0.4, DIM, 1, cv2.LINE_AA)

    # ---- playhead -----------------------------------------------------
    px = fx(frame)
    cv2.line(canvas, (px, top - 6), (px, y1 - pad), (255, 255, 255), 2)
    cv2.circle(canvas, (px, top - 6), 4, (255, 255, 255), -1)

    # ---- legend (bottom right) ----------------------------------------
    lx2 = inner_x1
    ly = y1 - pad
    items = [("pick", KIND_COLORS["pick"]), ("place", KIND_COLORS["place"]),
             ("closing", EVENT_COLORS["grasp"]), ("opening", EVENT_COLORS["release"]),
             ("grasp kp", EVENT_COLORS["grasp"]), ("release kp", EVENT_COLORS["release"])]
    for name, col in reversed(items):
        w = text_size(name, 0.4)[0]
        lx2 -= w + 14
        cv2.putText(canvas, name, (lx2, ly), FONT, 0.4, DIM, 1, cv2.LINE_AA)
        lx2 -= 14
        fill_rect(canvas, lx2, ly - 9, lx2 + 10, ly, col)
        lx2 -= 14


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# output sink
# ---------------------------------------------------------------------------
class VideoSink:
    """Write BGR frames to an mp4.

    Prefers piping raw frames into ``ffmpeg -c:v libx264`` because OpenCV's
    bundled encoders are poor here (``avc1`` needs a hardware device that does not
    exist on this machine, and ``mp4v`` produces ~1.4 MB per second, i.e. ~1.7 GB
    for a 20-episode batch).  Falls back to ``cv2.VideoWriter`` when ffmpeg is
    unavailable.
    """

    def __init__(self, path: str, width: int, height: int, fps: float,
                 crf: int = 20, preset: str = "veryfast",
                 codec: str = "mp4v") -> None:
        self.path = path
        self.width = width
        self.height = height
        self.fps = fps
        self.process: Optional[subprocess.Popen] = None
        self.writer: Optional[cv2.VideoWriter] = None
        self.backend = ""

        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg and width % 2 == 0 and height % 2 == 0:
            cmd = [
                ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-s", f"{width}x{height}", "-r", f"{fps:.6f}", "-i", "-",
                "-an", "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", path,
            ]
            try:
                self.process = subprocess.Popen(
                    cmd, stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                )
                self.backend = f"ffmpeg/libx264 crf={crf} preset={preset}"
                return
            except OSError:
                self.process = None

        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
        if not self.writer.isOpened():
            raise IOError(f"cannot open a video writer for {path}")
        self.backend = f"cv2/{codec}"

    def write(self, frame: np.ndarray) -> None:
        if self.process is not None and self.process.stdin is not None:
            self.process.stdin.write(np.ascontiguousarray(frame).tobytes())
        elif self.writer is not None:
            self.writer.write(frame)

    def close(self) -> None:
        if self.process is not None:
            if self.process.stdin is not None:
                self.process.stdin.close()
            stderr = self.process.stderr.read().decode("utf-8", "replace") \
                if self.process.stderr is not None else ""
            self.process.wait()
            if self.process.returncode != 0 and stderr:
                raise IOError(f"ffmpeg failed for {self.path}: {stderr.strip()[:400]}")
            self.process = None
        if self.writer is not None:
            self.writer.release()
            self.writer = None

    def __enter__(self) -> "VideoSink":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def camera_path(meta: Dict[str, Any], episode: int, camera: str) -> str:
    chunk = episode // meta["chunks_size"]
    rel = meta["video_path"].format(
        episode_chunk=chunk, episode_index=episode, video_key=camera
    )
    return os.path.join(meta["root"], rel)


def render_episode(
    meta: Dict[str, Any],
    episode: int,
    rec: Dict[str, Any],
    cameras: Sequence[str],
    output_path: str,
    hold_frames: int = 0,
    event_radius: Optional[int] = None,
    speed: float = 1.0,
    max_seconds: Optional[float] = None,
    codec: str = "mp4v",
    crf: int = 22,
    preset: str = "veryfast",
) -> Dict[str, Any]:
    """Write one annotated review video; returns a small stats dict."""
    fps = float(rec["fps"])
    state = EpisodeRenderState(episode, rec)
    horizon = state.horizon
    if event_radius is None:
        event_radius = max(3, int(round(0.35 * fps)))

    caps = []
    for cam in cameras:
        path = camera_path(meta, episode, cam)
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise IOError(f"cannot open camera video: {path}")
        caps.append((cam, cap))

    n_cams = len(caps)
    if max_seconds is not None:
        horizon = min(horizon, max(1, int(max_seconds * fps)))

    configure_layout(n_cams)
    sink = VideoSink(output_path, CANVAS_W, CANVAS_H, fps / max(speed, 1e-6),
                     crf=crf, preset=preset, codec=codec)

    hold_countdown = 0
    frames_written = 0
    keyframes_written = 0

    for frame in range(horizon):
        frames = []
        for cam, cap in caps:
            ok, img = cap.read()
            if not ok:
                # video shorter than the parquet: freeze the last frame
                img = frames[-1] if frames else np.zeros((480, 640, 3), np.uint8)
                cv2.putText(img, f"{cam}: no frame", (10, 24), FONT, 0.6, (255, 255, 255), 1)
            frames.append(img)

        canvas = np.full((CANVAS_H, CANVAS_W, 3), BG, np.uint8)
        for i, img in enumerate(frames):
            paste_videos(canvas, img, i, n_cams)

        ev = draw_keyframe_banner(
            canvas, state, frame, event_radius,
            camera_label="cameras: " + ", ".join(cameras),
        )
        draw_dashboard(canvas, state, frame, event_radius)
        draw_gripper_zoom(canvas, state, frame)
        draw_timeline(
            canvas, state, frame,
            header=("TIMELINE   subtask spans / detected gripper intervals / keyposes.  "
                    "Annotation text is shown in English (OpenCV cannot render CJK)."),
        )

        sink.write(canvas)
        frames_written += 1

        # hold on a keypose so it is easy to see
        if hold_frames > 0 and ev is not None and ev["frame"] == frame:
            for _ in range(hold_frames):
                sink.write(canvas)
                frames_written += 1
            keyframes_written += 1

    for _cam, cap in caps:
        cap.release()
    backend = sink.backend
    sink.close()

    return {
        "episode": episode,
        "output": output_path,
        "frames": frames_written,
        "camera_frames": horizon,
        "keyframes": len(state.kp_frames),
        "held": keyframes_written,
        "resolution": f"{CANVAS_W}x{CANVAS_H}",
        "fps": fps / max(speed, 1e-6),
        "backend": backend,
        "bytes": os.path.getsize(output_path) if os.path.exists(output_path) else 0,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset_root", required=True, help="LeRobot dataset root")
    parser.add_argument("--episode", type=int, default=None, help="Single episode index")
    parser.add_argument("--num_episodes", type=int, default=None,
                        help="Render this many episodes from --start_episode")
    parser.add_argument("--episode_list", type=str, default=None, help='e.g. "0,3,7"')
    parser.add_argument("--start_episode", type=int, default=0)
    parser.add_argument("--output_dir", default="./data/lerobot_videos")
    parser.add_argument("--cameras", default="cam_high",
                        help="Comma separated camera keys, e.g. cam_high,cam_left_wrist")
    parser.add_argument("--profile", default="moz1")
    parser.add_argument("--arms", default="left,right")
    parser.add_argument("--search_radius", type=int, default=None)
    parser.add_argument("--rotation_convention", default="zyx")
    parser.add_argument("--hold_frames", type=int, default=10,
                        help="Extra frames to repeat each keypose (0 = real time)")
    parser.add_argument("--event_radius", type=int, default=None,
                        help="Frames around a keypose that highlight the video (default 0.35 s)")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed factor")
    parser.add_argument("--max_seconds", type=float, default=None, help="Only render the first N seconds")
    parser.add_argument("--codec", default="mp4v",
                        help="Fallback cv2 codec when ffmpeg is unavailable")
    parser.add_argument("--crf", type=int, default=22,
                        help="x264 quality (lower = better/larger; 18..24 is a good range)")
    parser.add_argument("--preset", default="veryfast",
                        help="x264 preset (ultrafast..slow)")
    parser.add_argument("--skip_existing", action="store_true", help="Do not re-render existing mp4s")
    args = parser.parse_args()

    meta = load_dataset_meta(args.dataset_root)
    params = get_profile(args.profile)
    arms = [a.strip() for a in args.arms.split(",") if a.strip() in ARMS]
    cameras = [c.strip() for c in args.cameras.split(",") if c.strip()]

    if args.episode is not None:
        indices = [args.episode]
    elif args.episode_list:
        indices = [int(x) for x in args.episode_list.split(",") if x.strip()]
    else:
        count = args.num_episodes if args.num_episodes is not None else 1
        indices = list(range(args.start_episode, args.start_episode + count))

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 78)
    print("LeRobot keypose review video")
    print("=" * 78)
    print(f"Dataset  : {args.dataset_root}")
    print(f"Episodes : {len(indices)}" + (f"  ({indices[0]}..{indices[-1]})" if indices else ""))
    print(f"Cameras  : {cameras}")
    print(f"Canvas   : {CANVAS_W}x{CANVAS_H} @ {meta['fps'] / max(args.speed, 1e-6):.1f} fps"
          f"   ({len(cameras)} cam{'s' if len(cameras) > 1 else ''} at native size)"
          f"   hold={args.hold_frames} frames   crf={args.crf}")
    print(f"Output   : {args.output_dir}")
    print("=" * 78)

    from tqdm import tqdm

    stats: List[Dict[str, Any]] = []
    for ep in tqdm(indices, desc="Rendering videos"):
        out_path = os.path.join(args.output_dir, f"episode_{ep:06d}_review.mp4")
        if args.skip_existing and os.path.exists(out_path):
            print(f"  episode {ep}: exists, skipped")
            continue
        try:
            rec = process_episode(
                meta, ep, params, arms,
                rotation_convention=args.rotation_convention,
                radius=args.search_radius,
            )
            st = render_episode(
                meta, ep, rec, cameras, out_path,
                hold_frames=args.hold_frames,
                event_radius=args.event_radius,
                speed=args.speed,
                max_seconds=args.max_seconds,
                codec=args.codec,
                crf=args.crf,
                preset=args.preset,
            )
            stats.append(st)
            tqdm.write(
                f"  episode {ep}: {st['keyframes']} keyposes, "
                f"{st['frames']} frames, {st['bytes'] / 1e6:.1f} MB "
                f"-> {os.path.basename(out_path)}"
            )
        except Exception as exc:  # noqa: BLE001 - keep the batch going
            tqdm.write(f"  episode {ep}: FAILED - {exc}")

    print("\n" + "=" * 78)
    print(f"Rendered {len(stats)}/{len(indices)} videos into {args.output_dir}")
    if stats:
        total = sum(s["frames"] for s in stats)
        size = sum(s["bytes"] for s in stats) / 1e6
        print(f"Total output frames: {total}  "
              f"(~{total / max(stats[0]['fps'], 1e-6) / 60:.1f} min of footage)")
        print(f"Total size: {size:.1f} MB   encoder: {stats[0].get('backend', 'n/a')}")
    print("=" * 78)


if __name__ == "__main__":
    main()

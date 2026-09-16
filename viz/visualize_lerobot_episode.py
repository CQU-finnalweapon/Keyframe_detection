#!/usr/bin/env python3
"""
Visualise the annotation-driven keypose extraction for a LeRobot episode.

One figure per episode with four stacked panels:

  1. end-effector position of both arms + annotated subtask windows + keyposes
  2. per-axis velocity of the *active* arm with the segmentation thresholds and
     the moving / stable bands produced by ``trajectory_segmentation_v3``
  3. gripper state of both arms with the detected closing / opening intervals
  4. timeline strip: subtask bands, gripper events, keypose ticks

Usage
-----
    # one episode
    python viz/visualize_lerobot_episode.py \
        --dataset_root /path/to/PickPlaceONLY_Moz1_cjb \
        --episode 0 --output_dir ./data/lerobot_kp/viz

    # a batch: one figure per episode plus a combined overview sheet
    python viz/visualize_lerobot_episode.py \
        --dataset_root /path/to/PickPlaceONLY_Moz1_cjb \
        --num_episodes 20 --output_dir ./data/lerobot_kp/viz
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from src.lerobot_utils import (  # noqa: E402
    ARMS,
    get_episode_length,
    get_profile,
    load_dataset_meta,
    load_episode_streams,
    extract_subtasks,
)
from src.subtask_keypose import (  # noqa: E402
    DEFAULT_KEY_EVENT_TYPES,
    extract_key_events,
    make_keyposes,
    segment_all_arms,
)

ARM_COLORS = {"left": "#1f77b4", "right": "#d62728"}
AXIS_COLORS = ("#4c72b0", "#dd8452", "#55a868")
KIND_COLORS = {"pick": "#2e7d32", "place": "#c62828", "other": "#616161"}
EVENT_COLORS = {
    "grasp": "#2e7d32",
    "release": "#c62828",
    "detach": "#7b1fa2",
    "attach_drop": "#f9a825",
}


def compute_velocity(stream, fps: float, smooth: bool = False, window: int = 41) -> np.ndarray:
    """Velocity of the arm's cartesian position (T, 3).

    With ``smooth=True`` the Savitzky-Golay filter used by the segmentation is
    applied first, so the plotted signal matches what the algorithm sees.
    """
    pos = stream.pos
    if smooth:
        from src.trajectory_segmentation_v3 import smooth_trajectory

        pos = smooth_trajectory(pos.copy(), window_length=min(window, len(pos) - 1))
    vel = np.zeros_like(pos)
    if len(pos) > 1:
        vel[1:] = (pos[1:] - pos[:-1]) * fps
    return vel


def process_episode(
    meta: Dict[str, Any],
    episode_idx: int,
    params: Dict[str, Any],
    arms: List[str],
    rotation_convention: str = "zyx",
    radius: Optional[int] = None,
) -> Dict[str, Any]:
    """Run the annotation-driven extraction for one episode."""
    fps = float(meta["fps"])
    length = get_episode_length(meta, episode_idx)
    subtasks = extract_subtasks(meta, episode_idx, fps=fps, num_frames=length)
    streams = load_episode_streams(
        meta, episode_idx, arms=arms,
        rotation_convention=rotation_convention,
        gripper_cmd_threshold=params["gripper_cmd_threshold"],
    )
    timelines = segment_all_arms(streams, params, fps)
    events, results = extract_key_events(
        subtasks, timelines, streams, fps=fps, radius=radius,
        key_event_types=DEFAULT_KEY_EVENT_TYPES,
    )
    return {
        "fps": fps,
        "subtasks": subtasks,
        "streams": streams,
        "timelines": timelines,
        "events": events,
        "params": params,
        "results": results,
        "resolved_arms": [r.arm for r in results],
        "keyposes": make_keyposes(events, streams),
    }


def plot_overview(
    records: List[Dict[str, Any]],
    output_path: str,
    ncols: int = 2,
) -> str:
    """Compact contact sheet: gripper trace + keyposes for many episodes at once.

    One row per episode, showing the measured gripper opening of both arms with
    the annotated subtask spans, the detected closing/opening intervals and the
    resulting keyposes.  Meant for eyeballing a whole batch quickly, before
    opening individual figures.
    """
    n = len(records)
    ncols = max(1, min(ncols, n))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(9.0 * ncols, 1.9 * nrows + 1.2),
        squeeze=False,
    )

    for k, rec in enumerate(records):
        ax = axes[k // ncols][k % ncols]
        ep = rec["episode"]
        fps = rec["fps"]
        streams = rec["streams"]
        subtasks = rec["subtasks"]
        timelines = rec["timelines"]
        keyposes = rec["keyposes"]
        horizon = next(iter(streams.values())).num_frames

        # annotated subtask windows
        for sub in subtasks:
            ax.axvspan(
                sub.start_frame / fps, sub.end_frame / fps,
                color=KIND_COLORS.get(sub.kind, KIND_COLORS["other"]),
                alpha=0.08, lw=0,
            )

        # measured gripper opening per arm
        for arm, stream in streams.items():
            g = stream.gripper_state
            lo, hi = float(g.min()), float(g.max())
            norm = (g - lo) / (hi - lo) if hi - lo > 1e-9 else np.zeros_like(g)
            t = np.arange(len(norm)) / fps
            ax.plot(t, norm + (0.0 if arm == "left" else 1.15), lw=0.9,
                    color=ARM_COLORS.get(arm, "#333333"),
                    label=f"{arm} gripper")

        # detected gripper intervals
        for arm, tl in timelines.items():
            offset = 0.0 if arm == "left" else 1.15
            for iv in tl.get("gripper", []):
                if iv.end <= 0 or iv.start >= horizon:
                    continue
                color = "#2e7d32" if iv.label == "closing" else "#c62828"
                ax.axvspan(iv.start / fps, min(iv.end, horizon) / fps,
                           ymin=0.0, ymax=0.5, color=color, alpha=0.22, lw=0)

        # keyposes
        for frame, etype in zip(keyposes["frames"], keyposes["event_types"]):
            ax.axvline(frame / fps, color=EVENT_COLORS.get(etype, "#333333"),
                       lw=1.6, alpha=0.95)

        ax.set_xlim(0, horizon / fps)
        ax.set_ylim(-0.15, 2.35)
        ax.set_yticks([0.5, 1.65])
        ax.set_yticklabels(["L", "R"], fontsize=8)
        ax.tick_params(labelsize=8)
        n_kp = len(keyposes["frames"])
        ax.set_title(
            f"episode {ep}  |  {len(subtasks)} subtasks, {n_kp} keyposes: "
            + ",".join(keyposes["event_types"]),
            fontsize=9, loc="left",
        )
        ax.grid(axis="x", alpha=0.25, lw=0.5)
        if k % ncols == 0:
            ax.set_ylabel("gripper", fontsize=8)
        if k // ncols == nrows - 1:
            ax.set_xlabel("time (s)", fontsize=8)

    for k in range(n, nrows * ncols):
        axes[k // ncols][k % ncols].set_visible(False)

    handles = [
        Patch(facecolor="#2e7d32", alpha=0.22, label="detected closing"),
        Patch(facecolor="#c62828", alpha=0.22, label="detected opening"),
        plt.Line2D([0], [0], color=EVENT_COLORS["grasp"], lw=1.6, label="grasp keypose"),
        plt.Line2D([0], [0], color=EVENT_COLORS["release"], lw=1.6, label="release keypose"),
        Patch(facecolor=KIND_COLORS["pick"], alpha=0.08, label="pick subtask"),
        Patch(facecolor=KIND_COLORS["place"], alpha=0.08, label="place subtask"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(
        f"Keypose extraction overview - {n} episodes", fontsize=13, y=0.995
    )
    fig.tight_layout(rect=(0, 0.022, 1, 0.985))
    fig.savefig(output_path, dpi=110)
    plt.close(fig)
    return output_path


def plot_episode(
    meta: Dict[str, Any],
    episode_idx: int,
    result: Dict[str, Any],
    keyposes: Dict[str, Any],
    output_path: str,
    title: Optional[str] = None,
    resolved_arms: Optional[List[Optional[str]]] = None,
) -> str:
    """Render the diagnostic figure and return the written path."""
    fps = result["fps"]
    subtasks = result["subtasks"]
    streams = result["streams"]
    timelines = result["timelines"]
    events = result["events"]
    horizon = next(iter(streams.values())).num_frames
    t = np.arange(horizon) / fps

    # Derive the active arm per timestep from the annotated subtask windows
    active_arm = np.empty(horizon, dtype=object)
    for arm in streams:
        active_arm[:] = arm
    for i, sub in enumerate(subtasks):
        arm = None
        if resolved_arms is not None and i < len(resolved_arms):
            arm = resolved_arms[i]
        arm = arm or sub.arm
        if arm is None or arm not in streams:
            continue
        s, e = max(0, sub.start_frame), min(horizon, sub.end_frame)
        active_arm[s:e] = arm

    fig = plt.figure(figsize=(17, 11))
    gs = fig.add_gridspec(4, 1, height_ratios=[2.1, 2.1, 1.4, 1.0], hspace=0.42)

    def _subtask_spans(ax, alpha=0.10):
        for sub in subtasks:
            ax.axvspan(
                sub.start_frame / fps,
                sub.end_frame / fps,
                color=KIND_COLORS.get(sub.kind, "#616161"),
                alpha=alpha,
                lw=0,
            )
            ax.axvline(sub.start_frame / fps, color="#bdbdbd", lw=0.7, ls="--", zorder=1)

    def _event_lines(ax, only_primary=True):
        for ev in events:
            if only_primary and not ev.is_primary:
                continue
            ax.axvline(
                ev.frame / fps,
                color=EVENT_COLORS.get(ev.type, "k"),
                lw=1.8,
                ls="-",
                alpha=0.9,
                zorder=5,
            )

    # ---------------- panel 1: positions --------------------------------
    ax1 = fig.add_subplot(gs[0])
    _subtask_spans(ax1)
    for arm, stream in streams.items():
        for d, c in zip(range(3), AXIS_COLORS):
            ax1.plot(t, stream.pos[:, d], color=c, lw=1.0, alpha=0.85,
                     ls="-" if arm == "right" else "--",
                     label=f"{arm} {'xyz'[d]}")
    # keypose markers on the position of their own arm
    for i, (frame, arm, etype) in enumerate(
        zip(keyposes["frames"], keyposes["arm"], keyposes["event_types"])
    ):
        if arm not in streams:
            continue
        ax1.plot(
            frame / fps,
            streams[arm].pos[frame, 2],
            marker="v" if etype == "grasp" else "^",
            ms=9,
            color=EVENT_COLORS.get(etype, "k"),
            mec="k",
            mew=0.6,
            zorder=6,
        )
    _event_lines(ax1)
    ax1.set_ylabel("EE position [m]")
    ax1.set_title(
        f"Episode {episode_idx} - keypose extraction "
        f"({len(subtasks)} subtasks, {len(keyposes['frames'])} keyposes)"
        if title is None
        else title,
        fontsize=12,
    )
    ax1.legend(ncol=6, fontsize=7, loc="upper right", framealpha=0.9)
    ax1.grid(alpha=0.25)
    ax1.set_xlim(0, t[-1])

    # ---------------- panel 2: velocity of the active arm ---------------
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    _subtask_spans(ax2)
    params = result["params"]
    thr = params["thresholds"]
    for arm, stream in streams.items():
        vel = compute_velocity(stream, fps, smooth=True, window=params["window_size"])
        for d, c in zip(range(3), AXIS_COLORS):
            ax2.plot(t, vel[:, d], color=c, lw=0.9, alpha=0.85,
                     ls="-" if arm == "right" else "--",
                     label=f"{arm} v{'xyz'[d]}")
        # moving bands only for this arm, dimmed unless active
        for iv in timelines[arm]["movement"]:
            if iv.label != "moving":
                continue
            x0, x1 = (iv.start - 1) / fps, (iv.end - 1) / fps
            ax2.axvspan(x0, x1, color=ARM_COLORS.get(arm, "k"),
                        alpha=0.13 if (active_arm == arm).mean() > 0.5 else 0.05, lw=0)
    for d, c in zip(range(3), AXIS_COLORS):
        ax2.axhline(thr["xyz"[d]], color=c, ls=":", lw=0.9, alpha=0.8)
        ax2.axhline(-thr["xyz"[d]], color=c, ls=":", lw=0.9, alpha=0.8)
    _event_lines(ax2)
    ax2.set_ylabel("velocity [m/s]")
    ax2.set_title(
        f"Smoothed per-axis velocity vs thresholds "
        f"(x={thr['x']}, y={thr['y']}, z={thr['z']} m/s, window={params['window_size']}); "
        f"shaded = 'moving' intervals",
        fontsize=10,
    )
    ax2.legend(ncol=6, fontsize=7, loc="upper right", framealpha=0.9)
    ax2.grid(alpha=0.25)

    # ---------------- panel 3: gripper ----------------------------------
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    _subtask_spans(ax3)
    for arm, stream in streams.items():
        ax3.plot(t, stream.gripper_state, color=ARM_COLORS[arm], lw=1.4, label=f"{arm} state")
        ax3.plot(t, stream.gripper_cmd, color=ARM_COLORS[arm], lw=0.8, ls=":", alpha=0.7,
                 label=f"{arm} cmd")
        for iv in timelines[arm]["gripper"]:
            x0, x1 = (iv.start - 1) / fps, max(iv.end - 1, iv.start) / fps
            ax3.axvspan(x0, x1, color="#2e7d32" if iv.label == "closing" else "#c62828",
                        alpha=0.35, lw=0)
    _event_lines(ax3)
    ax3.set_ylabel("gripper [m]")
    ax3.set_title("Gripper state/command; shaded = detected closing (green) / opening (red)",
                  fontsize=10)
    ax3.legend(ncol=4, fontsize=7, loc="upper right", framealpha=0.9)
    ax3.grid(alpha=0.25)

    # ---------------- panel 4: timeline strip ---------------------------
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    for sub in subtasks:
        y = 0.55 if sub.kind == "pick" else 0.25
        arm_label = ""
        if resolved_arms is not None and sub.index < len(resolved_arms):
            arm_label = resolved_arms[sub.index] or ""
        ax4.add_patch(
            plt.Rectangle(
                (sub.start_frame / fps, y - 0.16),
                (sub.end_frame - sub.start_frame) / fps,
                0.22,
                color=KIND_COLORS.get(sub.kind, "#616161"),
                alpha=0.75,
            )
        )
        ax4.text(
            sub.start_frame / fps,
            y + 0.16,
            f"#{sub.index} {sub.kind[:4]}:{arm_label or sub.arm or '?'}",
            fontsize=6.5,
            va="bottom",
        )
    for ev in events:
        y = 0.92 if ev.is_primary else 0.06
        ax4.plot(ev.frame / fps, y, marker="|", ms=14, mew=2.0,
                 color=EVENT_COLORS.get(ev.type, "k"))
        ax4.text(ev.frame / fps, y, f" {ev.type}", fontsize=6.5, color=EVENT_COLORS.get(ev.type, "k"),
                 va="center" if ev.is_primary else "bottom", rotation=0)
    ax4.set_ylim(0, 1.15)
    ax4.set_yticks([])
    ax4.set_xlabel("time [s]")
    ax4.set_title("Subtask timeline (upper row = pick, lower row = place) and detected events",
                  fontsize=10)
    ax4.set_xlim(0, t[-1])

    handles = [
        Patch(color=KIND_COLORS["pick"], alpha=0.75, label="pick"),
        Patch(color=KIND_COLORS["place"], alpha=0.75, label="place"),
        Patch(color="#2e7d32", alpha=0.35, label="closing"),
        Patch(color="#c62828", alpha=0.35, label="opening"),
    ]
    ax4.legend(handles=handles, ncol=4, fontsize=7, loc="lower right")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset_root", required=True, help="LeRobot dataset root")
    parser.add_argument("--episode", type=int, default=None,
                        help="Single episode index (omit to use --num_episodes/--episode_list)")
    parser.add_argument("--num_episodes", type=int, default=None,
                        help="Visualise this many episodes starting at --start_episode")
    parser.add_argument("--episode_list", type=str, default=None,
                        help='Comma separated indices, e.g. "0,3,7"')
    parser.add_argument("--start_episode", type=int, default=0, help="First episode for --num_episodes")
    parser.add_argument("--output_dir", default="./data/lerobot_kp/viz", help="Where to write the figures")
    parser.add_argument("--profile", default="moz1", help="Segmentation parameter profile")
    parser.add_argument("--arms", default="left,right", help="Arms to analyse")
    parser.add_argument("--threshold_x", type=float, default=None)
    parser.add_argument("--threshold_y", type=float, default=None)
    parser.add_argument("--threshold_z", type=float, default=None)
    parser.add_argument("--window_size", type=int, default=None)
    parser.add_argument("--search_radius", type=int, default=None)
    parser.add_argument("--rotation_convention", default="zyx")
    parser.add_argument("--overview_cols", type=int, default=2,
                        help="Columns in the combined overview sheet")
    parser.add_argument("--no_overview", action="store_true",
                        help="Skip the combined overview sheet (default: write it for batches)")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Only print a one-line summary per episode")
    args = parser.parse_args()

    meta = load_dataset_meta(args.dataset_root)
    params = get_profile(args.profile)
    for axis in ("x", "y", "z"):
        override = getattr(args, f"threshold_{axis}")
        if override is not None:
            params["thresholds"][axis] = override
    if args.window_size is not None:
        params["window_size"] = args.window_size

    arms = [a.strip() for a in args.arms.split(",") if a.strip() in ARMS]

    # ---- which episodes -------------------------------------------------
    if args.episode is not None:
        indices = [args.episode]
        batch = False
    elif args.episode_list:
        indices = [int(x) for x in args.episode_list.split(",") if x.strip() != ""]
        batch = True
    else:
        count = args.num_episodes if args.num_episodes is not None else 1
        indices = list(range(args.start_episode, args.start_episode + count))
        batch = count > 1

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 78)
    print("LeRobot keypose visualisation")
    print("=" * 78)
    print(f"Dataset  : {args.dataset_root}")
    print(f"Episodes : {len(indices)}" + (f"  ({indices[0]}..{indices[-1]})" if indices else ""))
    print(f"Output   : {args.output_dir}")
    print("=" * 78)

    from tqdm import tqdm

    records: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    iterator = tqdm(indices, desc="Rendering") if batch and not args.quiet else indices

    for ep in iterator:
        try:
            rec = process_episode(
                meta, ep, params, arms,
                rotation_convention=args.rotation_convention,
                radius=args.search_radius,
            )
        except Exception as exc:  # noqa: BLE001 - keep the batch going
            failures.append({"episode": ep, "error": str(exc)})
            print(f"  [episode {ep}] FAILED: {exc}")
            continue

        rec["episode"] = ep
        out = os.path.join(args.output_dir, f"episode_{ep:06d}.png")
        plot_episode(
            meta, ep,
            {
                "fps": rec["fps"],
                "subtasks": rec["subtasks"],
                "streams": rec["streams"],
                "timelines": rec["timelines"],
                "events": rec["events"],
                "params": params,
            },
            rec["keyposes"],
            out,
            resolved_arms=rec["resolved_arms"],
        )
        records.append(rec)

        if args.quiet:
            kp = rec["keyposes"]
            print(f"  ep{ep:6d}  subtasks={len(rec['subtasks']):3d}  "
                  f"keyposes={len(kp['frames']):3d}  -> {os.path.basename(out)}")
        else:
            print(f"\nEpisode {ep}: {len(rec['subtasks'])} subtasks, "
                  f"{len(rec['keyposes']['frames'])} keyposes -> {out}")
            for r in rec["results"]:
                iv = r.matched_interval
                print(f"  #{r.subtask.index} {r.subtask.kind:5s} arm={str(r.arm):5s} "
                      f"win=[{r.subtask.start_frame},{r.subtask.end_frame}) "
                      f"iv={str(iv):12s} {str(r.matched_label):8s} {r.message}")
            print("  keypose frames:", rec["keyposes"]["frames"].tolist())
            print("  keypose types :", rec["keyposes"]["event_types"])

    # ---- combined overview ---------------------------------------------
    if records and (batch or len(records) > 1) and not args.no_overview:
        overview_path = os.path.join(
            args.output_dir, f"overview_{len(records)}episodes.png"
        )
        plot_overview(records, overview_path, ncols=args.overview_cols)
        print(f"\nOverview sheet -> {overview_path}")

    total_kp = sum(len(r["keyposes"]["frames"]) for r in records)
    total_sub = sum(len(r["subtasks"]) for r in records)
    print("\n" + "=" * 78)
    print(f"Rendered {len(records)}/{len(indices)} episodes, "
          f"{total_sub} subtasks, {total_kp} keyposes")
    if failures:
        print(f"Failures: {[f['episode'] for f in failures]}")
    print("=" * 78)


if __name__ == "__main__":
    main()

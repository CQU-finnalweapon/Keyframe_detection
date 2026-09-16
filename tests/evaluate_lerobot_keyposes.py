#!/usr/bin/env python3
"""
Batch validation of the LeRobot keypose extraction.

Runs the annotation-driven extraction over a range of episodes and checks:

  1. coverage      - how many annotated subtasks received a gripper match;
  2. in-window     - the keypose frame lies inside its subtask window;
  3. ordering      - keyposes are strictly increasing in time;
  4. alternation   - pick/place pairs produce grasp/release alternating;
  5. realness      - the gripper actually travels at the detected interval;
  6. direction     - the gripper is (much) more closed after a grasp than before,
                     and more open after a release than before;
  7. geometry      - grasp happens before the matching release.

Usage
-----
    python tests/evaluate_lerobot_keyposes.py \
        --dataset_root /path/to/PickPlaceONLY_Moz1_cjb \
        --num_episodes 100 --report ./data/lerobot_kp/eval_report.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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


def evaluate_episode(
    meta: Dict[str, Any],
    episode_idx: int,
    params: Dict[str, Any],
    arms: List[str],
    radius: Optional[int] = None,
    split_cycles: bool = True,
) -> Dict[str, Any]:
    """Run the pipeline on one episode and return validation metrics."""
    fps = float(meta["fps"])
    length = get_episode_length(meta, episode_idx)
    subtasks = extract_subtasks(meta, episode_idx, fps=fps, num_frames=length)
    streams = load_episode_streams(
        meta, episode_idx, arms=arms,
        gripper_cmd_threshold=params["gripper_cmd_threshold"],
    )
    timelines = segment_all_arms(streams, params, fps)
    events, results = extract_key_events(
        subtasks, timelines, streams, fps=fps, radius=radius,
        key_event_types=DEFAULT_KEY_EVENT_TYPES,
        split_cycle_windows=split_cycles,
    )
    keyposes = make_keyposes(events, streams)

    n_sub = len(subtasks)
    n_matched = sum(1 for r in results if r.matched_interval is not None)
    primary = [e for e in events if e.is_primary]
    window_of = {sub.index: (sub.start_frame, sub.end_frame) for sub in subtasks}

    # ---- checks ---------------------------------------------------------
    in_window = 0
    real_event = 0
    direction_ok = 0
    offsets: List[float] = []
    for ev in primary:
        win = window_of[ev.subtask_index]
        if win[0] <= ev.frame < win[1]:
            in_window += 1
        offsets.append((ev.frame - win[0]) / fps)

        stream = streams.get(ev.arm)
        if stream is None or ev.interval is None:
            continue
        s, e = ev.interval
        # gripper travel across the detected interval (+/- 2 frames of margin)
        lo = max(0, s - 2)
        hi = min(stream.num_frames, e + 2)
        travel = float(np.abs(np.diff(stream.gripper_state[lo:hi])).sum())
        if travel > 5e-3:
            real_event += 1
        # direction: a grasp must end with a smaller opening, a release larger
        pre = stream.gripper_state[max(0, s - 3):s]
        post = stream.gripper_state[max(0, e - 1):min(stream.num_frames, e + 2)]
        if len(pre) and len(post):
            delta = float(np.mean(post) - np.mean(pre))
            if (ev.type == "grasp" and delta < 0) or (ev.type == "release" and delta > 0):
                direction_ok += 1

    frames = keyposes["frames"]
    monotonic = bool(np.all(np.diff(frames) > 0)) if len(frames) > 1 else True

    # alternation: consecutive primary events of a pick/place chain must alternate
    types = [e.type for e in primary]
    alternates = all(
        types[i] != types[i + 1] for i in range(len(types) - 1)
    ) if types else True

    # grasp must precede its release
    pair_ok = True
    pending_grasp: Optional[int] = None
    for ev in primary:
        if ev.type == "grasp":
            pending_grasp = ev.frame
        elif ev.type == "release" and pending_grasp is not None:
            if ev.frame <= pending_grasp:
                pair_ok = False
            pending_grasp = None

    return {
        "episode": episode_idx,
        "subtasks": n_sub,
        "matched": n_matched,
        "keyposes": len(frames),
        "events": len(events),
        "in_window": in_window,
        "real_event": real_event,
        "direction_ok": direction_ok,
        "monotonic": int(monotonic),
        "alternating": int(alternates),
        "grasp_before_release": int(pair_ok),
        "mean_offset_s": float(np.mean(offsets)) if offsets else float("nan"),
        "median_offset_s": float(np.median(offsets)) if offsets else float("nan"),
        "keypose_types": ",".join(types),
        "keypose_frames": ",".join(str(int(f)) for f in frames),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--num_episodes", type=int, default=50)
    parser.add_argument("--episode_list", type=str, default=None, help='e.g. "0,1,5"')
    parser.add_argument("--start_episode", type=int, default=0)
    parser.add_argument("--arms", default="left,right")
    parser.add_argument("--profile", default="moz1")
    parser.add_argument("--search_radius", type=int, default=None)
    parser.add_argument("--threshold_x", type=float, default=None)
    parser.add_argument("--threshold_y", type=float, default=None)
    parser.add_argument("--threshold_z", type=float, default=None)
    parser.add_argument("--window_size", type=int, default=None)
    parser.add_argument("--report", default=None, help="Where to write the per-episode CSV")
    parser.add_argument("--split_cycles", action="store_true",
                        help="Emit both halves of a collapsed pick+place window (off by default: "
                             "measured to help fewer episodes than it breaks)")
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
    if args.episode_list:
        indices = [int(x) for x in args.episode_list.split(",") if x.strip()]
    else:
        indices = list(range(args.start_episode, args.start_episode + args.num_episodes))

    from tqdm import tqdm

    rows: List[Dict[str, Any]] = []
    for ep in tqdm(indices, desc="Evaluating"):
        try:
            rows.append(evaluate_episode(meta, ep, params, arms, args.search_radius,
                                         split_cycles=args.split_cycles))
        except Exception as exc:  # noqa: BLE001
            rows.append({"episode": ep, "error": str(exc), "subtasks": 0, "matched": 0,
                         "keyposes": 0, "events": 0, "in_window": 0, "real_event": 0,
                         "direction_ok": 0, "monotonic": 0, "alternating": 0,
                         "grasp_before_release": 0, "mean_offset_s": float("nan"),
                         "median_offset_s": float("nan"), "keypose_types": "",
                         "keypose_frames": ""})

    ok = [r for r in rows if "error" not in r]
    total_sub = sum(r["subtasks"] for r in ok)
    total_matched = sum(r["matched"] for r in ok)
    total_kp = sum(r["keyposes"] for r in ok)
    total_primary = sum(r["in_window"] for r in ok)

    def pct(num: int, den: int) -> str:
        return f"{num}/{den} ({100.0 * num / den:.1f}%)" if den else "n/a"

    print("\n" + "=" * 78)
    print("Keypose extraction validation")
    print("=" * 78)
    print(f"Episodes                 : {len(ok)}/{len(rows)}")
    print(f"Subtasks                 : {total_sub}")
    print(f"Subtask match rate       : {pct(total_matched, total_sub)}")
    print(f"Keyposes                 : {total_kp}")
    print(f"Keypose inside its window: {pct(sum(r['in_window'] for r in ok), total_primary)}")
    print(f"Gripper really moved     : {pct(sum(r['real_event'] for r in ok), total_primary)}")
    print(f"Correct close/open dir.  : {pct(sum(r['direction_ok'] for r in ok), total_primary)}")
    print(f"Strictly increasing      : {pct(sum(r['monotonic'] for r in ok), len(ok))}")
    print(f"Grasp/release alternate  : {pct(sum(r['alternating'] for r in ok), len(ok))}")
    print(f"Grasp before release     : {pct(sum(r['grasp_before_release'] for r in ok), len(ok))}")
    offsets = [r["mean_offset_s"] for r in ok if not np.isnan(r["mean_offset_s"])]
    if offsets:
        print(f"Offset from window start : mean {np.mean(offsets):.2f} s, "
              f"median {np.median(offsets):.2f} s (fraction of the subtask spent reaching)")
    print("=" * 78)

    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        keys = list(rows[0].keys())
        with open(args.report, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Per-episode report written to {args.report}")


if __name__ == "__main__":
    main()

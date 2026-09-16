#!/usr/bin/env python3
"""Diagnose why subtask -> gripper interval matching fails on a LeRobot dataset.

Runs the annotation-driven extraction over a range of episodes and buckets every
subtask by outcome so systematic failure modes become visible:

    ok                 matched an interval with the expected label
    label-mismatch     an interval was assigned but had the wrong label
    empty-window       the assigned arm had no gripper interval in the window
    no-arm             arm resolution failed for the subtask
    skipped-other      subtask intent could not be classified as pick/place
    unmatched          interval exists but was filtered out before matching

Usage:
    python tests/diagnose_lerobot_matching.py \
        --dataset_root /path/to/dataset --num_episodes 200
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.lerobot_utils import (  # noqa: E402
    get_episode_length,
    get_profile,
    has_annotation,
    list_episode_indices,
    load_dataset_meta,
    load_episode_streams,
    extract_subtasks,
)
from src.subtask_keypose import (  # noqa: E402
    DEFAULT_KEY_EVENT_TYPES,
    extract_key_events,
    segment_all_arms,
)


def classify_outcome(res, sub, timelines, streams) -> str:
    """Bucket one SubtaskResult into a coarse outcome label."""
    if res.arm is None:
        return "no-arm"
    if res.matched_interval is not None:
        return "ok"
    if sub.kind == "other":
        return "skipped-other"
    if res.message.startswith("expected"):
        return "label-mismatch"
    if res.message.startswith("no interval"):
        # Distinguish "timeline empty" from "intervals exist but elsewhere"
        arm_ivs = (timelines.get(res.arm) or {}).get("gripper", [])
        return "empty-timeline" if not arm_ivs else "empty-window"
    return "unmatched"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--num_episodes", type=int, default=100)
    ap.add_argument("--episode_list", type=str, default=None)
    ap.add_argument("--profile", type=str, default="moz1")
    ap.add_argument("--top", type=int, default=25, help="How many example failures to print per bucket")
    ap.add_argument("--json_out", type=str, default=None, help="Optional path to dump the full report")
    args = ap.parse_args()

    meta = load_dataset_meta(args.dataset_root)
    params = get_profile(args.profile)
    fps = float(meta["fps"])

    if args.episode_list:
        indices = [int(x) for x in args.episode_list.split(",") if x.strip()]
    else:
        indices = list_episode_indices(meta)[: args.num_episodes]

    buckets: Counter = Counter()
    examples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    kind_stats: Dict[str, Counter] = defaultdict(Counter)
    per_episode: List[Dict[str, Any]] = []
    window_overflow = Counter()  # were unmatched intervals just outside the window?

    for ep in indices:
        if not has_annotation(meta, ep):
            buckets["no_annotation"] += 1
            continue
        length = get_episode_length(meta, ep)
        subtasks = extract_subtasks(meta, ep, fps=fps, num_frames=length)
        streams = load_episode_streams(
            meta, ep, gripper_cmd_threshold=params["gripper_cmd_threshold"]
        )
        timelines = segment_all_arms(streams, params, fps)
        events, results = extract_key_events(
            subtasks, timelines, streams, fps=fps,
            key_event_types=DEFAULT_KEY_EVENT_TYPES,
        )

        n_ok = 0
        for res in results:
            outcome = classify_outcome(res, res.subtask, timelines, streams)
            buckets[outcome] += 1
            kind_stats[res.subtask.kind][outcome] += 1
            if outcome == "ok":
                n_ok += 1
                continue

            # For window failures, check how far the nearest interval is
            nearest = None
            if res.arm and res.arm in timelines:
                arm_ivs = timelines[res.arm].get("gripper", [])
                win = (res.subtask.start_frame, res.subtask.end_frame)
                best = None
                for iv in arm_ivs:
                    d = 0 if win[0] <= iv.start < win[1] else min(
                        abs(iv.start - win[0]), abs(iv.start - win[1])
                    )
                    if best is None or d < best[0]:
                        best = (d, iv)
                if best:
                    nearest = {"dist": int(best[0]), "start": best[1].start,
                               "end": best[1].end, "label": best[1].label}
                    if best[0] <= int(round(fps)):
                        window_overflow["within_1s"] += 1
                    elif best[0] <= int(round(2 * fps)):
                        window_overflow["within_2s"] += 1
                    else:
                        window_overflow["beyond_2s"] += 1

            if len(examples[outcome]) < args.top:
                examples[outcome].append(
                    {
                        "episode": ep,
                        "subtask": res.subtask.index,
                        "kind": res.subtask.kind,
                        "arm": res.arm,
                        "arm_source": res.arm_source,
                        "window": [res.subtask.start_frame, res.subtask.end_frame],
                        "message": res.message,
                        "nearest_interval": nearest,
                        "action": res.subtask.action,
                        "action_english": res.subtask.action_english,
                    }
                )
        per_episode.append({"episode": ep, "subtasks": len(results), "matched": n_ok})

    total = sum(buckets.values())
    print("=" * 80)
    print(f"LeRobot matching diagnostics - {args.dataset_root}")
    print(f"Episodes: {len(indices)}   Subtasks: {total}   profile={args.profile}")
    print("=" * 80)
    for name, count in buckets.most_common():
        pct = 100.0 * count / total if total else 0.0
        print(f"  {name:18s} {count:6d}  ({pct:5.1f}%)")

    print("\nBy subtask kind:")
    for kind, counter in kind_stats.items():
        sub_total = sum(counter.values())
        print(f"  {kind}: {dict(counter)}  (total {sub_total})")

    print("\nHow far is the nearest interval when the window is empty?")
    for name, count in window_overflow.most_common():
        print(f"  {name:12s} {count}")

    for outcome, rows in examples.items():
        if outcome == "ok":
            continue
        print("\n" + "-" * 80)
        print(f"Examples: {outcome} ({buckets[outcome]} total)")
        print("-" * 80)
        for r in rows[: args.top]:
            print(
                f"  ep{r['episode']:5d} sub#{r['subtask']} {r['kind']:5s} arm={str(r['arm']):5s}"
                f"({r['arm_source']:15s}) win=[{r['window'][0]:5d},{r['window'][1]:5d}) "
                f"nearest={r['nearest_interval']}\n"
                f"        msg: {r['message']}\n"
                f"        act: {r['action_english'][:90]}"
            )

    matched = buckets.get("ok", 0)
    if total:
        print("\n" + "=" * 80)
        print(f"MATCH RATE: {matched}/{total} = {100.0 * matched / total:.1f}%")
        print("=" * 80)

    if args.json_out:
        report = {
            "dataset_root": args.dataset_root,
            "num_episodes": len(indices),
            "buckets": dict(buckets),
            "kind_stats": {k: dict(v) for k, v in kind_stats.items()},
            "window_overflow": dict(window_overflow),
            "examples": examples,
            "per_episode": per_episode,
        }
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nReport written to {args.json_out}")


if __name__ == "__main__":
    main()

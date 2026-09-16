#!/usr/bin/env python3
"""
Generate keypose training data from a LeRobot dataset (no VLM required).

Why this script exists
----------------------
``gen_dataset.py`` starts from a robomimic HDF5 file *plus* VL model output.
LeRobot datasets recorded on Moz1 already carry hand-annotated subtasks in
``dataset.json``, so the semantic stage ("what happens, roughly when") is free.
This script skips the VLM entirely and runs only the deterministic part:

    annotated subtask  ->  pick / place + arm
                       ->  V3 trajectory segmentation on that arm
                       ->  grasp / release frame inside the subtask window
                       ->  keypose dataset (hdf5) + diagnostics (json)

Usage
-----
    python gen_dataset_lerobot.py \
        --dataset_root /path/to/PickPlaceONLY_Moz1_cjb \
        --output_dir ./data/lerobot_kp/PickPlaceONLY_Moz1_cjb \
        --num_episodes 10

    # single episode, verbose
    python gen_dataset_lerobot.py --dataset_root ... --episode 3 --verbose
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
import h5py
import numpy as np
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent))

from src.lerobot_utils import (  # noqa: E402
    ARMS,
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
    events_to_json,
    extract_key_events,
    make_keyposes,
    segment_all_arms,
)
from src.trajectory_segmentation_v3 import format_timelines_readable  # noqa: E402

# 0 = none, 1 = grasp, 2 = release, 3 = detach, 4 = attach_drop
EVENT_TYPE_IDS: Dict[str, int] = {
    "grasp": 1,
    "release": 2,
    "detach": 3,
    "attach_drop": 4,
}


# ---------------------------------------------------------------------------
# Per-episode processing
# ---------------------------------------------------------------------------
def process_episode(
    meta: Dict[str, Any],
    episode_idx: int,
    params: Dict[str, Any],
    arms: List[str],
    key_event_types: List[str],
    radius: Optional[int] = None,
    rotation_convention: str = "zyx",
    add_detach: bool = True,
    add_attach_drop: bool = False,
    split_cycles: Optional[bool] = None,
) -> Dict[str, Any]:
    """Run segmentation + keypose extraction for a single episode."""
    fps = float(meta["fps"])
    length = get_episode_length(meta, episode_idx)
    subtasks = extract_subtasks(meta, episode_idx, fps=fps, num_frames=length)
    streams = load_episode_streams(
        meta,
        episode_idx,
        arms=arms,
        rotation_convention=rotation_convention,
        gripper_cmd_threshold=params["gripper_cmd_threshold"],
    )

    timelines = segment_all_arms(streams, params, fps)
    events, results = extract_key_events(
        subtasks,
        timelines,
        streams,
        fps=fps,
        key_event_types=key_event_types,
        radius=radius,
        add_detach=add_detach,
        add_attach_drop=add_attach_drop,
        split_cycle_windows=(
            params.get("split_cycle_windows", False) if split_cycles is None else split_cycles
        ),
    )
    keyposes = make_keyposes(events, streams, key_event_types=key_event_types)

    return {
        "fps": fps,
        "subtasks": subtasks,
        "streams": streams,
        "timelines": timelines,
        "events": events,
        "results": results,
        "keyposes": keyposes,
    }


def build_training_data(
    episode_result: Dict[str, Any],
    episode_idx: int,
    key_event_types: List[str],
) -> Dict[str, Any]:
    """Assemble the training tensors for one episode.

    Layout
    ------
    obs/epos             (T, 9)  end-effector pose of the *active* arm
    obs/left_epos        (T, 9)
    obs/right_epos       (T, 9)
    obs/active_arm       (T,)    0 = left, 1 = right
    obs/subtask_index    (T,)    annotated subtask active at t
    obs/prev_keypose     (T, 9)  last reached keypose (active arm frame of reference)
    target/next_keypose  (T, 9)  next keypose to reach
    target/event_type    (T,)    0 none / 1 grasp / 2 release / 3 detach / 4 attach_drop
    target/stage_index   (T,)    index of the keypose being pursued, in [0, K]
    target/keypose_*     (K, ...) keypose frames / poses / types / arms
    actions/*            raw `<arm>_cmd_*` commands + concatenated vector
    """
    streams = episode_result["streams"]
    subtasks = episode_result["subtasks"]
    keyposes = episode_result["keyposes"]
    fps = episode_result["fps"]

    # Reference stream for `T` (all arms share the episode length)
    first_arm = next(iter(streams))
    horizon = streams[first_arm].num_frames
    arm_ids = {arm: i for i, arm in enumerate(sorted(streams.keys()))}

    # ---- per-timestep active arm / subtask ------------------------------
    active_arm = np.zeros(horizon, dtype=np.int32)
    subtask_index = -np.ones(horizon, dtype=np.int32)
    default_arm = streams[first_arm].arm
    if keyposes["arm"]:
        default_arm = keyposes["arm"][0]
    for sub in subtasks:
        arm = sub.arm or default_arm
        if arm not in streams:
            continue
        s = max(0, min(sub.start_frame, horizon - 1))
        e = max(s + 1, min(sub.end_frame, horizon))
        active_arm[s:e] = arm_ids.get(arm, 0)
        subtask_index[s:e] = sub.index
    # Fill leading/trailing gaps with the first/last known values
    for arr in (active_arm, subtask_index):
        if arr.size and arr[0] < 0:
            arr[0] = 0
        for t in range(1, horizon):
            if arr[t] < 0:
                arr[t] = arr[t - 1]

    left_epos = streams["left"].epos() if "left" in streams else np.zeros((horizon, 9))
    right_epos = streams["right"].epos() if "right" in streams else np.zeros((horizon, 9))
    epos_active = np.where(
        active_arm[:, None] == arm_ids.get("right", 1), right_epos, left_epos
    )

    # ---- keypose bookkeeping -------------------------------------------
    kp_frames = keyposes["frames"]
    kp_epos = keyposes["epos"]
    kp_arms = keyposes["arm"]
    kp_types = keyposes["event_types"]
    num_kp = len(kp_frames)

    obs_prev_keypose = np.zeros((horizon, 9), dtype=float)
    target_next_keypose = np.zeros((horizon, 9), dtype=float)
    target_event_type = np.zeros(horizon, dtype=np.int32)
    target_stage_index = np.zeros(horizon, dtype=np.int32)

    for t in range(horizon):
        prev_idx = np.where(kp_frames <= t)[0]
        if len(prev_idx) > 0:
            i = prev_idx[-1]
            obs_prev_keypose[t] = kp_epos[i]
        else:
            obs_prev_keypose[t] = epos_active[0]

        next_idx = np.where(kp_frames > t)[0]
        if len(next_idx) > 0:
            i = next_idx[0]
            target_next_keypose[t] = kp_epos[i]
            target_event_type[t] = EVENT_TYPE_IDS.get(kp_types[i], 0)
            target_stage_index[t] = i
        else:
            target_next_keypose[t] = epos_active[-1]
            target_event_type[t] = 0
            target_stage_index[t] = num_kp

    # ---- actions (raw commands, concatenated) ---------------------------
    actions: Dict[str, np.ndarray] = {}
    vector_parts: List[np.ndarray] = []
    vector_names: List[str] = []
    for arm in sorted(streams.keys()):
        stream = streams[arm]
        if stream.cart_cmd is not None:
            actions[f"{arm}_cmd_cart_pos"] = stream.cart_cmd
            vector_parts.append(stream.cart_cmd)
            vector_names += [f"{arm}_cmd_cart_pos_{i}" for i in range(stream.cart_cmd.shape[1])]
        if stream.joint_cmd is not None:
            actions[f"{arm}_cmd_joint_pos"] = stream.joint_cmd
            vector_parts.append(stream.joint_cmd)
            vector_names += [f"{arm}_cmd_joint_pos_{i}" for i in range(stream.joint_cmd.shape[1])]
        actions[f"{arm}_gripper_cmd_pos"] = stream.gripper_cmd.reshape(-1, 1)
        vector_parts.append(stream.gripper_cmd.reshape(-1, 1))
        vector_names.append(f"{arm}_gripper_cmd_pos")

    actions["vector"] = (
        np.concatenate(vector_parts, axis=1) if vector_parts else np.zeros((horizon, 0))
    )

    training: Dict[str, Any] = {
        "obs": {
            "epos": epos_active,
            "left_epos": left_epos,
            "right_epos": right_epos,
            "active_arm": active_arm,
            "subtask_index": subtask_index,
            "prev_keypose": obs_prev_keypose,
        },
        "target": {
            "next_keypose": target_next_keypose,
            "event_type": target_event_type,
            "stage_index": target_stage_index,
            "keypose_frames": kp_frames,
            "keypose_epos": kp_epos,
            "keypose_types": kp_types,
            "keypose_arm": kp_arms,
        },
        "actions": actions,
        "action_vector_names": vector_names,
        "meta": {
            "episode_index": episode_idx,
            "fps": fps,
            "num_frames": horizon,
            "num_keyposes": num_kp,
            "arms": sorted(streams.keys()),
            "arm_ids": arm_ids,
            "key_event_types": list(key_event_types),
        },
    }
    if "qpos" in keyposes:
        training["target"]["keypose_qpos"] = keyposes["qpos"]
    return training


def save_keypose_dataset(path: str, training: Dict[str, Any]) -> None:
    """Write the training tensors to HDF5."""
    with h5py.File(path, "w") as f:
        g = f.create_group("obs")
        for key, val in training["obs"].items():
            g.create_dataset(key, data=val)

        g = f.create_group("target")
        for key, val in training["target"].items():
            if key in ("keypose_types", "keypose_arm"):
                g.create_dataset(key, data=np.array(val, dtype=object), dtype=h5py.string_dtype())
            else:
                g.create_dataset(key, data=val)

        g = f.create_group("actions")
        names = training["action_vector_names"]
        g.attrs["vector_names"] = json.dumps(names)
        for key, val in training["actions"].items():
            g.create_dataset(key, data=val)

        m = f.create_group("meta")
        for key, val in training["meta"].items():
            if isinstance(val, (dict, list)):
                m.attrs[key] = json.dumps(val)
            else:
                m.attrs[key] = val


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
@click.command()
@click.option("--dataset_root", required=True, type=str, help="LeRobot dataset root (contains dataset.json)")
@click.option("--output_dir", required=True, type=str, help="Directory for keypose datasets")
@click.option("--episode", type=int, default=None, help="Single episode index to process")
@click.option("--episode_list", type=str, default=None, help='Comma separated indices, e.g. "0,3,7"')
@click.option("--num_episodes", type=int, default=10, help="Process episodes 0..N-1 (ignored if --episode/--episode_list given)")
@click.option("--arms", type=str, default="left,right", help="Arms to analyse (comma separated)")
@click.option("--profile", type=str, default="moz1", help="Segmentation parameter profile (see src/lerobot_utils.py)")
@click.option("--threshold_x", type=float, default=None, help="Override per-axis velocity threshold (m/s)")
@click.option("--threshold_y", type=float, default=None, help="Override per-axis velocity threshold (m/s)")
@click.option("--threshold_z", type=float, default=None, help="Override per-axis velocity threshold (m/s)")
@click.option("--window_size", type=int, default=None, help="Override Savitzky-Golay window (odd, frames)")
@click.option("--search_radius", type=int, default=None, help="Frames to expand the subtask window when matching (default 0.5 s)")
@click.option("--rotation_convention", type=str, default="zyx", help="Euler convention used to derive quaternions from rx,ry,rz")
@click.option("--key_event_types", type=str, default=",".join(DEFAULT_KEY_EVENT_TYPES),
              help="Event types counted as keyposes")
@click.option("--add_detach/--no_add_detach", default=True, help="Also emit 'detach' events for pick subtasks")
@click.option("--add_attach_drop/--no_add_attach_drop", default=False, help="Also emit 'attach_drop' events for place subtasks")
@click.option("--save_timelines/--no_save_timelines", default=True, help="Write readable trajectory timelines per episode")
@click.option("--split_cycles/--no_split_cycles", default=None,
              help="Emit both halves of a window that spans a whole pick+place cycle "
                   "(repairs collapsed annotations; default comes from the profile)")
@click.option("--verbose", "-v", is_flag=True, default=False, help="Print per-subtask matching details")
def main(
    dataset_root: str,
    output_dir: str,
    episode: Optional[int],
    episode_list: Optional[str],
    num_episodes: int,
    arms: str,
    profile: str,
    threshold_x: Optional[float],
    threshold_y: Optional[float],
    threshold_z: Optional[float],
    window_size: Optional[int],
    search_radius: Optional[int],
    rotation_convention: str,
    key_event_types: str,
    add_detach: bool,
    add_attach_drop: bool,
    save_timelines: bool,
    split_cycles: Optional[bool],
    verbose: bool,
) -> None:
    """Generate keypose datasets from a LeRobot dataset with annotated subtasks."""
    dataset_root = os.path.abspath(os.path.expanduser(dataset_root))
    output_dir = os.path.abspath(os.path.expanduser(output_dir))
    os.makedirs(output_dir, exist_ok=True)

    arms_list = [a.strip() for a in arms.split(",") if a.strip()]
    for a in arms_list:
        if a not in ARMS:
            raise click.BadParameter(f"Unknown arm {a!r}; expected one of {ARMS}")
    key_types = [t.strip() for t in key_event_types.split(",") if t.strip()]

    meta = load_dataset_meta(dataset_root)
    params = get_profile(profile)
    if threshold_x is not None:
        params["thresholds"]["x"] = threshold_x
    if threshold_y is not None:
        params["thresholds"]["y"] = threshold_y
    if threshold_z is not None:
        params["thresholds"]["z"] = threshold_z
    if window_size is not None:
        params["window_size"] = window_size

    all_indices = list_episode_indices(meta)
    if episode is not None:
        indices = [episode]
    elif episode_list:
        indices = [int(x) for x in episode_list.split(",") if x.strip() != ""]
    else:
        indices = [i for i in all_indices[:num_episodes]]

    print("=" * 80)
    print("Keypose dataset generation (LeRobot / annotation-driven)")
    print("=" * 80)
    print(f"Dataset root : {dataset_root}")
    print(f"Task name    : {meta['raw'].get('task_name')}")
    print(f"FPS          : {meta['fps']}")
    print(f"Episodes     : {len(indices)} (of {len(all_indices)})")
    print(f"Arms         : {arms_list}")
    print(f"Thresholds   : {params['thresholds']}  window={params['window_size']}")
    print(f"Key events   : {key_types}")
    print(f"Output dir   : {output_dir}")
    print("=" * 80)

    summary_rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    for ep_idx in tqdm(indices, desc="Episodes"):
        try:
            if not has_annotation(meta, ep_idx):
                summary_rows.append(
                    {"episode": ep_idx, "status": "no_annotation", "num_keyposes": 0,
                     "num_events": 0, "subtasks": 0, "matched": 0}
                )
                continue

            result = process_episode(
                meta,
                ep_idx,
                params,
                arms_list,
                key_types,
                radius=search_radius,
                rotation_convention=rotation_convention,
                add_detach=add_detach,
                add_attach_drop=add_attach_drop,
                split_cycles=split_cycles,
            )
            training = build_training_data(result, ep_idx, key_types)
            kp_path = os.path.join(output_dir, f"kp_episode_{ep_idx}.hdf5")
            save_keypose_dataset(kp_path, training)

            payload = events_to_json(
                result["events"],
                result["results"],
                ep_idx,
                result["fps"],
                extra={
                    "dataset_root": dataset_root,
                    "task_name": meta["raw"].get("task_name"),
                    "params": {
                        "profile": profile,
                        "thresholds": params["thresholds"],
                        "window_size": params["window_size"],
                        "gripper_threshold": params["gripper_threshold"],
                        "max_gap": params["max_gap"],
                    },
                    "keyposes": {
                        "frames": result["keyposes"]["frames"].tolist(),
                        "types": result["keyposes"]["event_types"],
                        "arm": result["keyposes"]["arm"],
                    },
                },
            )
            with open(os.path.join(output_dir, f"events_episode_{ep_idx}.json"), "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)

            if save_timelines:
                lines = []
                for arm, timelines in result["timelines"].items():
                    lines.append("#" * 80)
                    lines.append(f"# ARM: {arm}")
                    lines.append("#" * 80)
                    lines.append(format_timelines_readable(timelines, episode=ep_idx))
                with open(os.path.join(output_dir, f"timelines_episode_{ep_idx}.txt"), "w", encoding="utf-8") as f:
                    f.write("\n".join(lines))

            matched = sum(1 for r in result["results"] if r.matched_interval is not None)
            summary_rows.append(
                {
                    "episode": ep_idx,
                    "status": "ok",
                    "num_keyposes": int(len(result["keyposes"]["frames"])),
                    "num_events": len(result["events"]),
                    "subtasks": len(result["subtasks"]),
                    "matched": matched,
                }
            )

            if verbose:
                print(f"\n[Episode {ep_idx}] {len(result['subtasks'])} subtasks, "
                      f"{len(result['keyposes']['frames'])} keyposes")
                for r in result["results"]:
                    iv = r.matched_interval
                    print(f"  #{r.subtask.index} {r.subtask.kind:5s} arm={r.arm:5s} "
                          f"win=[{r.subtask.start_frame},{r.subtask.end_frame}) "
                          f"iv={iv} {r.matched_label or '-':8s} {r.message} "
                          f"| {r.subtask.action_english[:52]}")

        except Exception as exc:  # keep batch runs going
            import traceback

            traceback.print_exc()
            failures.append({"episode": ep_idx, "error": str(exc)})
            summary_rows.append(
                {"episode": ep_idx, "status": f"error: {exc}", "num_keyposes": 0,
                 "num_events": 0, "subtasks": 0, "matched": 0}
            )

    # ---- summary --------------------------------------------------------
    summary_path = os.path.join(output_dir, "summary.csv")
    if summary_rows:
        import csv

        keys = ["episode", "status", "num_keyposes", "num_events", "subtasks", "matched"]
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for row in summary_rows:
                writer.writerow({k: row.get(k, "") for k in keys})

    ok = [r for r in summary_rows if r["status"] == "ok"]
    total_sub = sum(r["subtasks"] for r in ok)
    total_matched = sum(r["matched"] for r in ok)
    total_kp = sum(r["num_keyposes"] for r in ok)
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Episodes processed      : {len(ok)}/{len(indices)}")
    if total_sub:
        rate = 100.0 * total_matched / total_sub
        print(f"Subtask match rate      : {total_matched}/{total_sub} ({rate:.1f}%)")
    else:
        print("Subtask match rate      : n/a (no subtasks processed)")
    print(f"Keyposes extracted      : {total_kp}")
    print(f"Per-episode summary     : {summary_path}")
    if failures:
        print(f"Failed episodes         : {[f['episode'] for f in failures]}")
        with open(os.path.join(output_dir, "failures.json"), "w", encoding="utf-8") as f:
            json.dump(failures, f, indent=2)
    print("=" * 80)


if __name__ == "__main__":
    main()

"""
LeRobot dataset adapter for the keypose-labelling pipeline.

The original pipeline consumes robomimic HDF5 datasets, where the VL model has to
*discover* the interaction events (grasp / release / ...) from video.  LeRobot
datasets collected on the Moz1 platform already ship with **hand-annotated
subtasks** (``dataset.json`` -> ``episodes[i].annotation.annotation_data.data``),
so the "what / roughly when" stage is already solved and the VLM can be skipped.

This module converts the LeRobot layout into the same canonical structures the
rest of the project expects:

    trajectory (T, 9)   [pos(3), quat(4), gripper(1), 0.0]
    action_gripper (T,) -1 = opening command, +1 = closing command, 0 = none

so that ``src.trajectory_segmentation_v3`` / ``src.event_corrector_v3`` keep
working unchanged.

Directory layout assumed (LeRobot v2.1)::

    <root>/
      dataset.json                 # info + episodes[].annotation (inline subtasks)
      annotations.json             # optional, keyed by recording_id
      data/chunk-000/episode_000000.parquet
      videos/chunk-000/cam_high/episode_000000.mp4

Arm features use the ``<arm>arm_*`` prefix:

    leftarm_state_cart_pos      (T, 6)  x, y, z, rx, ry, rz
    rightarm_state_cart_pos     (T, 6)
    <arm>arm_gripper_state_pos  (T, 1)  0.0 = closed, ~0.1 = open
    <arm>arm_gripper_cmd_pos    (T, 1)
    <arm>arm_cmd_cart_pos       (T, 6)  absolute cartesian command
    <arm>arm_cmd_joint_pos      (T, 7)

Author: TASE Project
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:  # pandas / pyarrow are only needed for the parquet reader
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None  # type: ignore

ARM_PREFIXES: Dict[str, str] = {"left": "leftarm", "right": "rightarm"}
ARMS: Tuple[str, ...] = ("left", "right")

# ---------------------------------------------------------------------------
# Per-dataset segmentation parameters
# ---------------------------------------------------------------------------
# Moz1 pick&place data: 30 FPS, arm travel ~0.3 m over ~2 s (=> ~0.15 m/s mean).
# Thresholds are in m/s and are deliberately a bit below the mean travel speed so
# that slow approach / insertion phases are still labelled as motion.
LEROBOT_PROFILES: Dict[str, Dict[str, Any]] = {
    "default": {
        "thresholds": {"x": 0.020, "y": 0.020, "z": 0.015},
        "window_size": 41,
        "gripper_threshold": 1e-3,
        "max_gap": 5,
        "gripper_cmd_threshold": 1e-4,
        "min_movement_between": 8,
        "max_close_open_gap": 30,
        "min_closing_duration": 2,
        "min_opening_duration": 2,
        "merge_hiccup_gaps": 6,
        "interrupted_action_gap": 60,
        "min_short_travel": 5e-3,  # rescue 1-frame snap releases
        "mask_whole_command_run": False,  # commands are continuous setpoints
        # Off by default: measured over 600 episodes it fixed 1 episode with
        # collapsed (all-"place") annotations but broke 3 with ordinary ones.
        "split_cycle_windows": False,
        "sentinel_tail": False,
        "fps": 30.0,
    },
    "moz1": {
        # Calibrated on PickPlaceONLY_Moz1_cjb (30 FPS, 1.15k-2.1k frames/episode):
        #  * thresholds sit well below the median travel speed (~0.07 m/s) so that
        #    slow approach phases still count as motion;
        #  * gripper travel is small for bulky objects (apple: 0.019 m vs plug:
        #    0.090 m), so the closing interval can be as short as 3 frames ->
        #    min_closing_duration must stay permissive;
        #  * the gripper frequently closes in two stages (contact + squeeze) with a
        #    1-5 frame blip in between -> merge_hiccup_gaps;
        #  * a slow squeeze spends ~40 frames below the per-frame detection
        #    threshold, chopping one grasp into three -> interrupted_action_gap;
        #  * min_movement_between=0 disables the "correction motion" heuristic:
        #    it was designed for teleoperated robomimic data, and here it deletes
        #    genuine grasp/release pairs that happen without arm travel.  The
        #    annotated subtask windows already provide the semantic filtering.
        "thresholds": {"x": 0.025, "y": 0.025, "z": 0.018},
        "window_size": 41,
        "gripper_threshold": 1e-3,
        "max_gap": 5,
        "gripper_cmd_threshold": 1e-4,
        "min_movement_between": 0,
        "max_close_open_gap": 30,
        "min_closing_duration": 2,
        "min_opening_duration": 2,
        "merge_hiccup_gaps": 6,
        "interrupted_action_gap": 60,
        "min_short_travel": 5e-3,  # rescue 1-frame snap releases
        "mask_whole_command_run": False,  # commands are continuous setpoints
        "split_cycle_windows": False,
        "sentinel_tail": False,
        "fps": 30.0,
    },
}

# Text cues used to attach a subtask to the arm that performs it.
_ARM_CUES: Dict[str, Tuple[str, ...]] = {
    "left": ("左手", "left hand", "left arm", "left_arm"),
    "right": ("右手", "right hand", "right arm", "right_arm"),
}

# Text cues for the subtask kind.  "Pick/Put" are intentionally checked before
# the generic place keywords because English annotations use "Put ... inside".
_PICK_CUES: Tuple[str, ...] = ("拿起", "pick", "grasp", "grab", "拾取", "抓取")
_PLACE_CUES: Tuple[str, ...] = (
    "放在",
    "放到",
    "放入",
    "place",
    "put",
    "drop",
    "释放",
)


def get_profile(name: Optional[str] = None) -> Dict[str, Any]:
    """Return segmentation parameters for a dataset profile (defaults to 'moz1')."""
    if name and name in LEROBOT_PROFILES:
        return dict(LEROBOT_PROFILES[name])
    return dict(LEROBOT_PROFILES["default"])


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
def load_dataset_meta(root: str) -> Dict[str, Any]:
    """Load ``dataset.json`` and flatten the useful info into a dict.

    Args:
        root: Dataset root directory.

    Returns:
        dict with keys ``root``, ``fps``, ``robot_type``, ``features``,
        ``episodes`` (list, index == episode_index) and ``raw``.
    """
    root = os.path.abspath(os.path.expanduser(root))
    meta_path = os.path.join(root, "dataset.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"dataset.json not found under {root}")

    with open(meta_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    info = raw.get("info", {}) or {}
    episodes = raw.get("episodes") or []
    # Index episodes by episode_index so that lookups are O(1) even if the list
    # ever becomes sparse / reordered.
    by_index: Dict[int, Dict[str, Any]] = {}
    for pos, ep in enumerate(episodes):
        idx = int(ep.get("episode_index", pos))
        by_index[idx] = ep

    return {
        "root": root,
        "fps": float(info.get("fps", 30.0)),
        "robot_type": info.get("robot_type"),
        "features": info.get("features", {}) or {},
        "total_episodes": int(info.get("total_episodes", len(episodes))),
        "data_path": info.get(
            "data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
        ),
        "video_path": info.get(
            "video_path",
            "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        ),
        "chunks_size": int(info.get("chunks_size", 1000) or 1000),
        "episodes": by_index,
        "raw": raw,
    }


def list_episode_indices(meta: Dict[str, Any]) -> List[int]:
    """Sorted list of episode indices present in the metadata."""
    return sorted(meta["episodes"].keys())


def resolve_episode_path(meta: Dict[str, Any], episode_idx: int, kind: str = "data") -> str:
    """Resolve the parquet (``kind='data'``) or video-dir (``kind='video'``) path."""
    chunk = episode_idx // meta["chunks_size"]
    if kind == "data":
        rel = meta["data_path"].format(episode_chunk=chunk, episode_index=episode_idx)
    elif kind == "video":
        rel = os.path.dirname(meta["video_path"]).format(
            episode_chunk=chunk, episode_index=episode_idx, video_key="cam_high"
        )
    else:
        raise ValueError(f"Unknown kind: {kind}")
    return os.path.join(meta["root"], rel)


def get_episode_length(meta: Dict[str, Any], episode_idx: int) -> Optional[int]:
    """Frame count of an episode according to ``dataset.json``."""
    ep = meta["episodes"].get(episode_idx)
    if ep is None:
        return None
    length = ep.get("length")
    return int(length) if length is not None else None


# ---------------------------------------------------------------------------
# Subtask annotations
# ---------------------------------------------------------------------------
def infer_arm_from_text(*texts: Optional[str]) -> Optional[str]:
    """Infer 'left' / 'right' from annotation text; None when ambiguous."""
    joined = " ".join(t.lower() for t in texts if t)
    hits = [arm for arm, cues in _ARM_CUES.items() if any(c in joined for c in cues)]
    if len(hits) == 1:
        return hits[0]
    return None


def classify_subtask(*texts: Optional[str]) -> str:
    """Classify a subtask as 'pick', 'place' or 'other' from its annotation text."""
    joined = " ".join(t.lower() for t in texts if t)
    for cue in _PICK_CUES:
        if cue in joined:
            return "pick"
    for cue in _PLACE_CUES:
        if cue in joined:
            return "place"
    return "other"


@dataclass
class Subtask:
    """One annotated subtask with time, frame window and semantic labels."""

    index: int
    start_time: float
    end_time: float
    start_frame: int
    end_frame: int  # exclusive
    action: str = ""
    action_english: str = ""
    objects: List[str] = field(default_factory=list)
    objects_english: List[str] = field(default_factory=list)
    arm: Optional[str] = None  # from text only; corrected later if needed
    kind: str = "other"

    @property
    def duration_frames(self) -> int:
        return self.end_frame - self.start_frame

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "action": self.action,
            "action_english": self.action_english,
            "objects": self.objects,
            "objects_english": self.objects_english,
            "arm": self.arm,
            "kind": self.kind,
        }


def extract_subtasks(
    meta: Dict[str, Any],
    episode_idx: int,
    fps: Optional[float] = None,
    num_frames: Optional[int] = None,
) -> List[Subtask]:
    """Parse the annotated subtasks of one episode into frame windows.

    Args:
        meta: dataset metadata from :func:`load_dataset_meta`.
        episode_idx: Episode index.
        fps: Override FPS (defaults to ``meta['fps']``).
        num_frames: Clip windows to this many frames (usually the episode length).

    Returns:
        List of :class:`Subtask` in chronological order.
    """
    fps = float(fps or meta["fps"])
    ep = meta["episodes"].get(episode_idx)
    if ep is None:
        raise KeyError(f"Episode {episode_idx} not present in dataset.json")

    ann = ep.get("annotation") or {}
    ann_data = (ann.get("annotation_data") or {}) if isinstance(ann, dict) else {}
    raw_items = ann_data.get("data") or []

    if num_frames is None:
        num_frames = get_episode_length(meta, episode_idx) or 10**9

    subtasks: List[Subtask] = []
    for i, item in enumerate(raw_items):
        start_time = float(item.get("start_time", 0.0))
        end_time = float(item.get("end_time", 0.0))
        start_frame = int(round(start_time * fps))
        end_frame = int(round(end_time * fps))
        start_frame = max(0, min(start_frame, num_frames - 1))
        end_frame = max(start_frame + 1, min(end_frame, num_frames))
        action = (item.get("action") or "").strip()
        action_en = (item.get("action_english") or "").strip()
        subtasks.append(
            Subtask(
                index=i,
                start_time=start_time,
                end_time=end_time,
                start_frame=start_frame,
                end_frame=end_frame,
                action=action,
                action_english=action_en,
                objects=list(item.get("objects") or []),
                objects_english=list(item.get("objects_english") or []),
                arm=infer_arm_from_text(action, action_en),
                kind=classify_subtask(action, action_en),
            )
        )
    return subtasks


def has_annotation(meta: Dict[str, Any], episode_idx: int) -> bool:
    """Whether the episode carries at least one subtask annotation."""
    ep = meta["episodes"].get(episode_idx) or {}
    ann = ep.get("annotation") or {}
    data = (ann.get("annotation_data") or {}).get("data") if isinstance(ann, dict) else None
    return bool(data)


# ---------------------------------------------------------------------------
# Parquet -> arm streams
# ---------------------------------------------------------------------------
def load_episode_dataframe(
    meta: Dict[str, Any],
    episode_idx: int,
    columns: Optional[Sequence[str]] = None,
    root: Optional[str] = None,
) -> "pd.DataFrame":
    """Load one episode parquet file (optionally a column subset)."""
    if pd is None:  # pragma: no cover
        raise ImportError("pandas + pyarrow are required to read LeRobot parquet data")
    path = resolve_episode_path(meta, episode_idx, kind="data")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Episode parquet not found: {path}")
    return pd.read_parquet(path, columns=list(columns) if columns else None)


def derive_action_gripper(
    gripper_cmd: np.ndarray, threshold: float = 1e-4
) -> np.ndarray:
    """Convert a gripper *position command* into an action label stream.

    Matches the robomimic convention used by ``trajectory_segmentation_v3``:

        +1 -> closing command, -1 -> opening command, 0 -> hold
    """
    cmd = np.asarray(gripper_cmd, dtype=float).reshape(-1)
    action = np.zeros_like(cmd)
    if cmd.size < 2:
        return action
    diff = np.diff(cmd)
    opening = diff > threshold
    closing = diff < -threshold
    # Forward-fill: the instruction issued at step t is in effect until the next
    # change, which is what the gripper-label filter expects.
    action[1:] = np.where(opening, -1.0, np.where(closing, 1.0, 0.0))
    # Persist the last non-zero command while the value stays constant.
    last_cmd = 0.0
    for i in range(1, len(cmd)):
        if action[i] == 0.0:
            action[i] = last_cmd
        else:
            last_cmd = action[i]
    return action


def euler_xyz_to_quat(rot: np.ndarray, convention: str = "zyx") -> np.ndarray:
    """Convert raw ``rx, ry, rz`` columns into ``wxyz`` quaternions.

    The exact convention of the Moz1 cartesian states is not documented in the
    dataset; ``zyx`` (yaw-pitch-roll, scipy uppercase-aware ordering) produced the
    smoothest quaternion stream on this data.  The raw angles are always stored
    verbatim elsewhere, so this is a *derived* convenience column only.
    """
    from scipy.spatial.transform import Rotation as R

    rot = np.asarray(rot, dtype=float)
    if rot.ndim != 2 or rot.shape[1] != 3:
        raise ValueError(f"Expected (T, 3) rotation array, got {rot.shape}")
    quat_xyzw = R.from_euler(convention, rot).as_quat()
    return quat_xyzw[:, [3, 0, 1, 2]]  # -> wxyz


@dataclass
class ArmStream:
    """Canonical per-arm signals extracted from an episode."""

    arm: str
    pos: np.ndarray  # (T, 3)
    rot: np.ndarray  # (T, 3) raw rx, ry, rz
    quat: np.ndarray  # (T, 4) derived wxyz
    gripper_state: np.ndarray  # (T,) 0 = closed ... 0.1 = open
    gripper_cmd: np.ndarray  # (T,)
    action_gripper: np.ndarray  # (T,) -1 open cmd / +1 close cmd / 0 hold
    cart_cmd: Optional[np.ndarray] = None  # (T, 6)
    joint_pos: Optional[np.ndarray] = None  # (T, 7)
    joint_cmd: Optional[np.ndarray] = None  # (T, 7)

    @property
    def num_frames(self) -> int:
        return int(self.pos.shape[0])

    def epos(self) -> np.ndarray:
        """(T, 9) end-effector trajectory in the project's canonical layout."""
        t = self.pos.shape[0]
        out = np.zeros((t, 9), dtype=float)
        out[:, 0:3] = self.pos
        out[:, 3:7] = self.quat
        out[:, 7] = self.gripper_state
        out[:, 8] = 0.0  # second gripper column reserved (single-DoF gripper)
        return out


def _column_or_none(df: "pd.DataFrame", name: str) -> Optional[np.ndarray]:
    if name not in df.columns:
        return None
    values = df[name].to_numpy()
    # LeRobot stores fixed-size vectors as lists/objects -> stack into (T, d)
    if values.dtype == object:
        return np.vstack([np.asarray(v, dtype=float).reshape(-1) for v in values])
    return np.asarray(values, dtype=float)


def build_arm_stream(
    df: "pd.DataFrame",
    arm: str,
    rotation_convention: str = "zyx",
    gripper_cmd_threshold: float = 1e-4,
) -> ArmStream:
    """Build an :class:`ArmStream` for ``arm`` in {'left', 'right'}."""
    if arm not in ARM_PREFIXES:
        raise ValueError(f"arm must be one of {list(ARM_PREFIXES)}, got {arm!r}")
    prefix = ARM_PREFIXES[arm]

    cart = _column_or_none(df, f"{prefix}_state_cart_pos")
    if cart is None or cart.shape[1] < 6:
        raise KeyError(f"Column {prefix}_state_cart_pos (T, 6) not found in episode parquet")
    pos = cart[:, 0:3]
    rot = cart[:, 3:6]

    gripper_state = _column_or_none(df, f"{prefix}_gripper_state_pos")
    if gripper_state is None:
        raise KeyError(f"Column {prefix}_gripper_state_pos not found")
    gripper_state = gripper_state.reshape(-1)

    gripper_cmd = _column_or_none(df, f"{prefix}_gripper_cmd_pos")
    if gripper_cmd is None:
        gripper_cmd = gripper_state.copy()
    gripper_cmd = gripper_cmd.reshape(-1)

    return ArmStream(
        arm=arm,
        pos=pos,
        rot=rot,
        quat=euler_xyz_to_quat(rot, convention=rotation_convention),
        gripper_state=gripper_state,
        gripper_cmd=gripper_cmd,
        action_gripper=derive_action_gripper(gripper_cmd, gripper_cmd_threshold),
        cart_cmd=_column_or_none(df, f"{prefix}_cmd_cart_pos"),
        joint_pos=_column_or_none(df, f"{prefix}_state_joint_pos"),
        joint_cmd=_column_or_none(df, f"{prefix}_cmd_joint_pos"),
    )


def load_episode_streams(
    meta: Dict[str, Any],
    episode_idx: int,
    arms: Iterable[str] = ARMS,
    rotation_convention: str = "zyx",
    gripper_cmd_threshold: float = 1e-4,
) -> Dict[str, ArmStream]:
    """Load every requested arm stream for one episode."""
    df = load_episode_dataframe(meta, episode_idx)
    return {
        arm: build_arm_stream(df, arm, rotation_convention, gripper_cmd_threshold)
        for arm in arms
    }


def load_episode_context(
    root: str,
    episode_idx: int,
    arms: Iterable[str] = ARMS,
    profile: Optional[str] = None,
) -> Dict[str, Any]:
    """Convenience helper: metadata + subtasks + arm streams for one episode.

    Returns:
        dict with ``meta``, ``subtasks``, ``streams``, ``fps``, ``params``.
    """
    meta = load_dataset_meta(root)
    params = get_profile(profile)
    length = get_episode_length(meta, episode_idx)
    subtasks = extract_subtasks(meta, episode_idx, num_frames=length)
    streams = load_episode_streams(
        meta,
        episode_idx,
        arms=arms,
        gripper_cmd_threshold=params["gripper_cmd_threshold"],
    )
    return {
        "meta": meta,
        "fps": meta["fps"],
        "subtasks": subtasks,
        "streams": streams,
        "params": params,
        "length": length,
    }

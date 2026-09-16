"""
Subtask-driven keypose extraction.

Motivation
----------
The robomimic pipeline uses a VLM to *discover* interaction events, then corrects
their timestamps with trajectory analysis.  LeRobot datasets collected on Moz1
already contain annotated subtasks (``Pick up X`` / ``Place X on Y``), which play
exactly the same role as the VL output: they say **what** happens and **roughly
when**.  What is still missing is the exact frame, and that is what the
deterministic trajectory/gripper analysis provides.

This module therefore replaces ``divide_video.py`` + ``event_corrector_v3.py``
with a single, annotation-driven stage:

    annotated subtask  ->  arm + intent (pick / place)
                       ->  gripper closing / opening interval inside the window
                       ->  keypose frame + (optional) detach / attach_drop

Everything below is deterministic and needs no API key.

Author: TASE Project
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .lerobot_utils import ArmStream, Subtask
from .trajectory_segmentation_v3 import Interval, segment_trajectory_v3

# Event types produced by this module
EVENT_GRASP = "grasp"
EVENT_RELEASE = "release"
EVENT_DETACH = "detach"
EVENT_ATTACH_DROP = "attach_drop"

DEFAULT_KEY_EVENT_TYPES: Tuple[str, ...] = (EVENT_GRASP, EVENT_RELEASE)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------
@dataclass
class KeyEvent:
    """A single interaction event with its corrected frame index."""

    type: str
    frame: int  # 0-indexed, corrected by trajectory analysis
    subtask_index: int
    arm: str
    source: str  # 'gripper_closing' | 'gripper_opening' | 'gripper_end' | 'movement'
    anchor_frame: Optional[int] = None  # subtask boundary frame used as prior
    interval: Optional[Tuple[int, int]] = None  # (start, end) 1-indexed, exclusive end
    is_primary: bool = True  # grasp / release are primary keyposes
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "type": self.type,
            "frame_idx": self.frame,
            "corrected_frame_idx": self.frame,
            "subtask_index": self.subtask_index,
            "arm": self.arm,
            "source": self.source,
            "is_primary": self.is_primary,
        }
        if self.anchor_frame is not None:
            d["anchor_frame"] = self.anchor_frame
        if self.interval is not None:
            d["interval"] = [int(self.interval[0]), int(self.interval[1])]
        if self.note:
            d["note"] = self.note
        return d


@dataclass
class SubtaskResult:
    """Diagnostics for one annotated subtask."""

    subtask: Subtask
    arm: Optional[str] = None
    arm_source: str = "text"
    kind_used: Optional[str] = None  # 'pick' / 'place' actually used (may differ)
    kind_source: str = "text"  # 'text' or 'gripper-evidence'
    events: List[KeyEvent] = field(default_factory=list)
    matched_interval: Optional[Tuple[int, int]] = None
    matched_label: Optional[str] = None
    window_start_frame: Optional[int] = None  # first moving frame inside the window
    search_radius: int = 0
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subtask_index": self.subtask.index,
            "kind": self.subtask.kind,
            "kind_used": self.kind_used or self.subtask.kind,
            "kind_source": self.kind_source,
            "action": self.subtask.action,
            "action_english": self.subtask.action_english,
            "arm": self.arm,
            "arm_source": self.arm_source,
            "window": [self.subtask.start_frame, self.subtask.end_frame],
            "motion_start_frame": self.window_start_frame,
            "matched_interval": list(self.matched_interval) if self.matched_interval else None,
            "matched_label": self.matched_label,
            "events": [e.to_dict() for e in self.events],
            "message": self.message,
        }


# ---------------------------------------------------------------------------
# Arm inference fallback
# ---------------------------------------------------------------------------
def _window_signal(stream: ArmStream, start: int, end: int) -> float:
    """Score how "active" an arm is inside a frame window.

    Combines gripper travel (dominant cue for pick/place) with cartesian travel.
    """
    start = max(0, int(start))
    end = min(stream.num_frames, int(end))
    if end - start < 2:
        return 0.0
    grip = stream.gripper_state[start:end]
    grip_travel = float(np.abs(np.diff(grip)).sum())
    pos_travel = float(np.linalg.norm(np.diff(stream.pos[start:end], axis=0), axis=1).sum())
    return grip_travel + 0.25 * pos_travel


def infer_arm_from_activity(
    subtask: Subtask, streams: Dict[str, ArmStream]
) -> Tuple[Optional[str], str]:
    """Fallback arm inference when the annotation text does not name a hand."""
    if not streams:
        return None, "unknown"
    scores = {
        arm: _window_signal(s, subtask.start_frame, subtask.end_frame)
        for arm, s in streams.items()
    }
    arm = max(scores, key=lambda k: scores[k])
    if scores[arm] <= 1e-9:
        return None, "none"
    return arm, "activity"


def resolve_arms(
    subtasks: Sequence[Subtask], streams: Dict[str, ArmStream]
) -> List[Tuple[Optional[str], str]]:
    """Resolve the executing arm for every subtask.

    Priority:
      1. explicit hand mentioned in the annotation text (``右手`` / ``right hand``);
      2. inheritance from the previous subtask with a known arm -- these datasets
         are built from pick/place pairs and "Place X on Y" rarely repeats the
         hand that picked X up, while the picking subtask almost always names it;
      3. inheritance from the next known subtask;
      4. motion / gripper activity inside the window (last resort).

    Returns:
        List of ``(arm, source)`` aligned with ``subtasks``.
    """
    resolved: List[Tuple[Optional[str], str]] = [
        (s.arm, "text") if s.arm else (None, "") for s in subtasks
    ]

    last_known: Optional[str] = None
    for i, (arm, _src) in enumerate(resolved):
        if arm is not None:
            last_known = arm
        elif last_known is not None:
            resolved[i] = (last_known, "prev")

    next_known: Optional[str] = None
    for i in range(len(resolved) - 1, -1, -1):
        arm, _src = resolved[i]
        if arm is not None:
            next_known = arm
        elif next_known is not None:
            resolved[i] = (next_known, "next")

    for i, (arm, _src) in enumerate(resolved):
        if arm is None:
            resolved[i] = infer_arm_from_activity(subtasks[i], streams)

    return resolved


# ---------------------------------------------------------------------------
# Interval matching
# ---------------------------------------------------------------------------
def _distance_to_window(frame: int, window: Tuple[int, int]) -> int:
    """Frames between ``frame`` and a window (0 when inside)."""
    if window[0] <= frame < window[1]:
        return 0
    return min(abs(frame - window[0]), abs(frame - window[1]))


def assign_intervals(
    gripper_intervals: Sequence[Interval],
    subtask_windows: Sequence[Tuple[int, int]],
    radius: int,
) -> Dict[int, List[int]]:
    """Assign every gripper interval of one arm to at most one subtask window.

    The annotated windows of a single arm are disjoint and tile the episode, so
    each gripper action belongs to exactly one subtask.  Attributing globally
    (instead of letting each subtask greedily grab the nearest unused interval)
    removes the "stealing" failure mode where a place subtask takes the release
    that belongs to the next subtask.

    Args:
        gripper_intervals: intervals of one arm (1-indexed, exclusive end).
        subtask_windows: ``(start, end)`` windows of that arm's subtasks, in
            chronological order.
        radius: extra frames tolerated outside every window (annotation drift).

    Returns:
        ``{position_in_subtask_windows: [interval indices]}``
    """
    assignment: Dict[int, List[int]] = {i: [] for i in range(len(subtask_windows))}
    if not subtask_windows:
        return assignment

    for iv_idx, iv in enumerate(gripper_intervals):
        owner: Optional[int] = None
        for pos, window in enumerate(subtask_windows):
            if window[0] <= iv.start < window[1]:
                owner = pos
                break
        if owner is None and radius > 0:
            best_dist: Optional[int] = None
            for pos, window in enumerate(subtask_windows):
                dist = _distance_to_window(iv.start, window)
                if dist <= radius and (best_dist is None or dist < best_dist):
                    best_dist = dist
                    owner = pos
        if owner is not None:
            assignment[owner].append(iv_idx)
    return assignment


def _select_interval(
    intervals: Sequence[Interval],
    candidates: Sequence[int],
    labels: Sequence[str],
    allow_label_mismatch: bool = False,
) -> Tuple[Optional[int], str]:
    """Pick the chronologically first candidate with an expected label.

    Returns ``(interval index, message)``.
    """
    if not candidates:
        return None, "no interval assigned to this subtask"

    matching = [i for i in candidates if intervals[i].label in labels]
    if matching:
        idx = min(matching, key=lambda i: intervals[i].start)
        return idx, "matched"

    if allow_label_mismatch:
        idx = min(candidates, key=lambda i: intervals[i].start)
        return idx, f"label-mismatch({intervals[idx].label})"

    labels_str = "/".join(labels)
    found = "/".join(sorted({intervals[i].label for i in candidates}))
    return None, f"expected {labels_str} but window only holds {found}"


def contains_full_cycle(
    candidates: Sequence[int],
    gripper_intervals: Sequence[Interval],
    window: Tuple[int, int],
    min_separation: int = 1,
) -> bool:
    """Whether a subtask window holds a complete close+open cycle.

    Annotators sometimes collapse a whole pick-and-place into a single segment
    (or label every segment in an episode ``place``).  A subtask can physically
    consume only *one* gripper action, so such a window must be split.

    Two guards keep this from firing on ordinary data:

    * only intervals that genuinely *start* inside the window count, so ones
      pulled in by the search radius cannot trigger a split;
    * the closing and the opening must be at least ``min_separation`` frames
      apart.  A real pick-and-place has a transport phase in between, whereas a
      "squeeze and immediately release" shows up as two adjacent intervals and is
      one action, not two.
    """
    inside = [
        gripper_intervals[i]
        for i in candidates
        if window[0] <= gripper_intervals[i].start < window[1]
    ]
    closings = [iv for iv in inside if iv.label == "closing"]
    openings = [iv for iv in inside if iv.label == "opening"]
    if not closings or not openings:
        return False

    # A cycle exists when some closing is followed by an opening after a gap.
    for c in closings:
        for o in openings:
            if o.start >= c.end and (o.start - c.end) >= min_separation:
                return True
    return False


def expected_labels_for_kind(kind: str) -> Tuple[str, ...]:
    """Gripper interval label(s) that realise an annotated intent."""
    if kind == "pick":
        return ("closing",)
    if kind == "place":
        return ("opening",)
    return ("closing", "opening")


def find_other_arm_evidence(
    streams: Dict[str, ArmStream],
    timelines: Dict[str, Dict[str, List[Interval]]],
    exclude_arm: str,
    window: Tuple[int, int],
    expected_labels: Sequence[str],
) -> Optional[Tuple[str, List[Interval], List[int]]]:
    """Find another arm that shows gripper evidence inside a subtask window.

    Used when the annotated arm has nothing in the window -- a sign that the
    annotation named the wrong hand.  An arm holding an interval with an
    *expected* label is strong evidence; an arm holding some other action is
    weaker, so it is only accepted when it is the single candidate (this is what
    repairs annotations like "Pick up the apple ... with left hand" that the
    right arm actually performs as a release).

    Returns:
        ``(arm, intervals, candidate indices)`` or ``None``.
    """
    strong: List[Tuple[str, List[Interval], List[int]]] = []
    weak: List[Tuple[str, List[Interval], List[int]]] = []
    for other in streams:
        if other == exclude_arm:
            continue
        ivs = (timelines.get(other) or {}).get("gripper", [])
        candidates = [
            k for k, iv in enumerate(ivs) if window[0] <= iv.start < window[1]
        ]
        if not candidates:
            continue
        if any(ivs[k].label in expected_labels for k in candidates):
            strong.append((other, ivs, candidates))
        else:
            weak.append((other, ivs, candidates))

    if len(strong) == 1:
        return strong[0]
    if not strong and len(weak) == 1:
        return weak[0]
    return None


def _first_moving_start_in_window(
    movement_intervals: Sequence[Interval], window: Tuple[int, int], after: int = 0
) -> Optional[int]:
    """Frame of the first 'moving' interval that starts inside the window."""
    best: Optional[int] = None
    for iv in movement_intervals:
        if iv.label != "moving":
            continue
        if iv.start < window[0] or iv.start >= window[1]:
            continue
        if iv.start < after:
            continue
        if best is None or iv.start < best:
            best = iv.start
    return best


def resolve_intent(
    kind: str,
    candidates: Sequence[int],
    gripper_intervals: Sequence[Interval],
    allow_kind_fix: bool = True,
) -> Tuple[Tuple[str, ...], str, str, str]:
    """Decide which gripper action realises a subtask.

    The annotation says *what* the demonstrator intended, but the annotation text
    is human input and therefore imperfect.  The gripper signal, on the other
    hand, is physics.  This function reconciles the two:

    * ``pick``  -> expects a ``closing``  action -> event ``grasp``
    * ``place`` -> expects an ``opening``  action -> event ``release``
    * ``other`` -> intent missing, inferred from whichever action is present

    When the annotated intent contradicts the only action available inside the
    window, the signal wins (this repairs mislabelled subtasks such as
    ``"Pick up the pencil sharpener on the canned tea"`` that actually release the
    object).  The correction is reported back through the returned ``kind_used``
    and an explanatory message.

    Args:
        kind: Annotated intent (``pick`` / ``place`` / ``other``).
        candidates: Gripper interval indices assigned to this subtask's window.
        gripper_intervals: The full gripper timeline of the executing arm.
        allow_kind_fix: Whether physical evidence may override the annotation text.

    Returns:
        ``(labels, event_type, kind_used, message)`` where ``labels`` are the
        accepted interval labels (empty means "accept anything").
    """
    if kind == "pick":
        labels: Tuple[str, ...] = ("closing",)
        event_type = EVENT_GRASP
    elif kind == "place":
        labels = ("opening",)
        event_type = EVENT_RELEASE
    else:
        labels = ("closing", "opening")
        event_type = "interaction"

    if not candidates:
        return labels, event_type, kind, ""

    present = {gripper_intervals[i].label for i in candidates}

    # Annotated intent contradicted by the only action available -> trust physics.
    if kind in ("pick", "place") and allow_kind_fix and not (set(labels) & present):
        if kind == "pick" and "opening" in present:
            return (
                ("opening",), EVENT_RELEASE, "place",
                "kind corrected by gripper evidence (annotated pick, found release)",
            )
        if kind == "place" and "closing" in present:
            return (
                ("closing",), EVENT_GRASP, "pick",
                "kind corrected by gripper evidence (annotated place, found grasp)",
            )

    # Intent missing entirely -> infer from the action that is present.
    if kind == "other":
        if "closing" in present and "opening" not in present:
            return (
                ("closing",), EVENT_GRASP, "pick",
                "kind inferred from gripper evidence (closing)",
            )
        if "opening" in present and "closing" not in present:
            return (
                ("opening",), EVENT_RELEASE, "place",
                "kind inferred from gripper evidence (opening)",
            )

    return labels, event_type, kind, ""


def _moving_start_after(
    movement_intervals: Sequence[Interval], frame: int, window: Tuple[int, int]
) -> Optional[int]:
    """Frame of the first moving interval strictly after ``frame`` (within window)."""
    for iv in sorted(movement_intervals, key=lambda x: x.start):
        if iv.label != "moving":
            continue
        if iv.start > frame and iv.start < window[1]:
            return iv.start
    return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def segment_all_arms(
    streams: Dict[str, ArmStream],
    params: Dict[str, Any],
    fps: float,
) -> Dict[str, Dict[str, List[Interval]]]:
    """Run the V3 trajectory segmentation for every arm stream."""
    sampling_interval = 1.0 / float(fps)
    timelines: Dict[str, Dict[str, List[Interval]]] = {}
    for arm, stream in streams.items():
        timelines[arm] = segment_trajectory_v3(
            trajectory=stream.epos(),
            action_gripper=stream.action_gripper,
            task=params.get("task_name", "default"),
            window_size=params.get("window_size"),
            thresholds=params.get("thresholds"),
            gripper_threshold=params.get("gripper_threshold", 1e-3),
            max_gap=params.get("max_gap", 5),
            sampling_interval=sampling_interval,
            min_movement_between=params.get("min_movement_between", 10),
            max_close_open_gap=params.get("max_close_open_gap", 30),
            min_closing_duration=params.get("min_closing_duration", 4),
            min_opening_duration=params.get("min_opening_duration", 1),
            merge_hiccup_gaps=params.get("merge_hiccup_gaps", 0),
            interrupted_action_gap=params.get("interrupted_action_gap", 0),
            min_short_travel=params.get("min_short_travel", 0.0),
            mask_whole_command_run=params.get("mask_whole_command_run", True),
            sentinel_tail=params.get("sentinel_tail", False),
        )
        timelines[arm]["gripper"] = _drop_tail_sentinels(
            timelines[arm]["gripper"], stream.num_frames
        )
    return timelines


def _drop_tail_sentinels(
    intervals: Sequence[Interval], num_frames: int
) -> List[Interval]:
    """Remove the 1-frame placeholder intervals appended at the very end.

    ``segment_gripper_timeline`` can append a dummy closing/opening interval at
    the last frame so that robomimic tasks never miss a trailing event.  For
    annotation-driven datasets that placeholder would be indistinguishable from a
    real action, so it is dropped here.
    """
    kept: List[Interval] = []
    for iv in intervals:
        if iv.start >= num_frames - 1 and (iv.end - iv.start) <= 1:
            continue
        kept.append(iv)
    return kept


def extract_key_events(
    subtasks: Sequence[Subtask],
    timelines: Dict[str, Dict[str, List[Interval]]],
    streams: Dict[str, ArmStream],
    fps: float,
    key_event_types: Iterable[str] = DEFAULT_KEY_EVENT_TYPES,
    radius: Optional[int] = None,
    add_detach: bool = True,
    add_attach_drop: bool = False,
    allow_label_mismatch: bool = False,
    swap_arms_on_evidence: bool = True,
    fix_kind_on_evidence: bool = True,
    split_cycle_windows: bool = False,
    min_cycle_separation: Optional[int] = None,
    include_other_subtasks: bool = False,
) -> Tuple[List[KeyEvent], List[SubtaskResult]]:
    """Extract keypose events for every annotated subtask.

    Pipeline per episode:
      1. resolve which arm executes each subtask (:func:`resolve_arms`);
      2. attribute each gripper closing/opening interval to the subtask whose
         annotated window contains it (:func:`assign_intervals`);
      3. inside a pick subtask the first ``closing`` interval is the *grasp*,
         inside a place subtask the first ``opening`` interval is the *release*;
      4. optionally attach secondary events (``detach`` / ``attach_drop``).

    Args:
        subtasks: Annotated subtasks (frame windows + pick/place intent).
        timelines: Output of :func:`segment_all_arms`.
        streams: Per-arm signals (used for arm inference fallback).
        fps: Dataset frame rate (used for the default search radius).
        key_event_types: Event types that count as keyposes (primary events).
        radius: Frames of tolerated drift outside a window (default 0.5 s).
        add_detach: Emit a ``detach`` event (lift-off after grasp) for pick subtasks.
        add_attach_drop: Emit an ``attach_drop`` event (object lands) for place subtasks.
        allow_label_mismatch: Accept an interval with an unexpected label instead
            of reporting the subtask as unmatched.
        swap_arms_on_evidence: When the annotated arm has nothing in the window but
            the other arm holds exactly the expected action, switch arms.
        fix_kind_on_evidence: When the annotated pick/place intent contradicts the
            only gripper action inside the window, trust the gripper signal and
            flip the intent (repairs mislabelled subtasks).
        split_cycle_windows: When an annotated window contains a complete
            close+open cycle, emit both actions instead of only the one matching
            the annotated intent.  This repairs datasets whose annotations
            collapse a pick+place into one segment (or label every segment
            ``place``), but on this dataset it costs more than it gains -- it
            fires on ordinary windows that merely extend past the next pick's
            closure -- so it is **off by default**.  Measure before enabling.
        min_cycle_separation: Frames that must separate the closing and the
            opening of a window for it to count as a collapsed cycle (default
            0.5 s).  Stops a "squeeze then release" from being split into two
            events.
        include_other_subtasks: Also process subtasks whose intent is unknown.

    Returns:
        (events sorted by frame, per-subtask diagnostics)
    """
    if radius is None:
        radius = max(2, int(round(0.5 * fps)))
    if min_cycle_separation is None:
        min_cycle_separation = max(2, int(round(0.5 * fps)))

    key_set = set(key_event_types)
    events: List[KeyEvent] = []
    results: List[SubtaskResult] = []

    # ---- 1. resolve arms ------------------------------------------------
    arm_resolution = resolve_arms(subtasks, streams)

    # ---- 2. attribute gripper intervals to subtask windows --------------
    # positions per arm -> list of subtask indices (chronological)
    positions_by_arm: Dict[str, List[int]] = {}
    for i, sub in enumerate(subtasks):
        arm = arm_resolution[i][0]
        if arm is None:
            continue
        positions_by_arm.setdefault(arm, []).append(i)

    assignment: Dict[str, Dict[int, List[int]]] = {}
    for arm, positions in positions_by_arm.items():
        gripper_ivs = (timelines.get(arm) or {}).get("gripper", [])
        windows = [
            (subtasks[p].start_frame, subtasks[p].end_frame) for p in positions
        ]
        arm_assign = assign_intervals(gripper_ivs, windows, radius)
        # re-key by absolute subtask position
        assignment[arm] = {
            positions[local_pos]: iv_idxs for local_pos, iv_idxs in arm_assign.items()
        }

    # ---- 3. build events -------------------------------------------------
    for i, sub in enumerate(subtasks):
        arm, arm_source = arm_resolution[i]
        res = SubtaskResult(subtask=sub, arm=arm, arm_source=arm_source)
        results.append(res)

        if arm is None or arm not in timelines:
            res.message = "no arm stream available"
            continue

        timeline = timelines[arm]
        gripper_ivs = timeline.get("gripper", [])
        movement_ivs = timeline.get("movement", [])
        window = (sub.start_frame, sub.end_frame)
        res.window_start_frame = _first_moving_start_in_window(movement_ivs, window)

        candidates = assignment.get(arm, {}).get(i, [])
        labels, event_type, kind_used, kind_msg = resolve_intent(
            sub.kind, candidates, gripper_ivs, allow_kind_fix=fix_kind_on_evidence
        )
        if kind_used != sub.kind:
            res.kind_used = kind_used
            res.kind_source = "gripper-evidence"

        # ---- decide which actions belong to this subtask ------------------
        # `plans` holds (interval index, event type) pairs, so an unannotated
        # span can contribute several events.
        plans: List[Tuple[int, str]] = []
        msg = kind_msg

        if sub.kind == "other" and not include_other_subtasks and kind_used == "other":
            # The annotator could not describe this span ("无法标注" / "unknown"),
            # so there is no semantic constraint to enforce: keep every gripper
            # action inside the window instead of guessing one.
            if not candidates:
                res.message = "skipped subtask kind='other' (no gripper action in window)"
                continue
            plans = [
                (k, EVENT_GRASP if gripper_ivs[k].label == "closing" else EVENT_RELEASE)
                for k in sorted(candidates, key=lambda k: gripper_ivs[k].start)
            ]
            msg = (
                f"{msg}; " if msg else ""
            ) + f"unannotated span: kept all {len(plans)} gripper action(s)"
        elif split_cycle_windows and contains_full_cycle(
            candidates, gripper_ivs, window, min_separation=min_cycle_separation
        ):
            # The annotated window spans a whole pick+place; emit both halves.
            plans = [
                (k, EVENT_GRASP if gripper_ivs[k].label == "closing" else EVENT_RELEASE)
                for k in sorted(candidates, key=lambda k: gripper_ivs[k].start)
            ]
            msg = (
                f"{msg}; " if msg else ""
            ) + (
                f"window spans a full close+open cycle: emitted {len(plans)} "
                "action(s) instead of one"
            )
        else:
            idx, sel_msg = _select_interval(
                gripper_ivs, candidates, labels, allow_label_mismatch=allow_label_mismatch
            )
            msg = "; ".join(p for p in (kind_msg, sel_msg) if p)

            # ---- fallback: trust the measured gripper over the annotation text --
            # Some episodes are annotated with the wrong hand ("right hand" while
            # the left arm performs the motion).  The physical gripper signal is
            # the ground truth here, so when the assigned arm shows nothing in the
            # window, look at the other arm and re-resolve the intent against its
            # evidence (the switch may also flip pick <-> place).
            if idx is None and swap_arms_on_evidence:
                evidence = find_other_arm_evidence(
                    streams, timelines, arm, window, expected_labels_for_kind(sub.kind)
                )
                if evidence is not None:
                    other_arm, other_ivs, other_candidates = evidence
                    o_labels, o_type, o_kind, o_msg = resolve_intent(
                        sub.kind, other_candidates, other_ivs,
                        allow_kind_fix=fix_kind_on_evidence,
                    )
                    o_idx, o_sel = _select_interval(
                        other_ivs, other_candidates, o_labels,
                        allow_label_mismatch=allow_label_mismatch,
                    )
                    if o_idx is not None:
                        arm, gripper_ivs, idx = other_arm, other_ivs, o_idx
                        labels, event_type, kind_used = o_labels, o_type, o_kind
                        res.arm = other_arm
                        res.arm_source = "gripper-evidence"
                        if kind_used != sub.kind:
                            res.kind_used = kind_used
                            res.kind_source = "gripper-evidence"
                        msg = "; ".join(
                            p for p in (
                                o_msg, o_sel,
                                f"arm corrected by gripper evidence -> {other_arm}",
                            ) if p
                        )

            if idx is not None:
                plans = [(idx, event_type)]

        res.message = msg
        if not plans:
            continue

        primary_iv = gripper_ivs[plans[0][0]]
        res.matched_interval = (primary_iv.start, primary_iv.end)
        res.matched_label = primary_iv.label

        for plan_idx, plan_type in plans:
            iv = gripper_ivs[plan_idx]
            event = KeyEvent(
                type=plan_type,
                frame=iv.start - 1,  # 1-indexed interval -> 0-indexed time step
                subtask_index=sub.index,
                arm=arm,
                source=f"gripper_{iv.label}",
                anchor_frame=sub.start_frame,
                interval=(iv.start, iv.end),
                is_primary=plan_type in key_set,
                note=msg,
            )
            events.append(event)
            res.events.append(event)

            # ---- secondary events ------------------------------------------
            # Use the *resolved* kind: a subtask annotated as "pick" that actually
            # releases the object must not produce a detach event.
            if add_detach and kind_used == "pick" and plan_type == EVENT_GRASP:
                detach_frame = _moving_start_after(movement_ivs, iv.start, window)
                if detach_frame is None:
                    detach_frame = iv.end  # fall back to the closing action's end
                detach = KeyEvent(
                    type=EVENT_DETACH,
                    frame=max(iv.start, detach_frame) - 1,
                    subtask_index=sub.index,
                    arm=arm,
                    source="movement" if detach_frame != iv.end else "gripper_end",
                    anchor_frame=sub.start_frame,
                    interval=None,
                    is_primary=False,
                )
                events.append(detach)
                res.events.append(detach)

            if add_attach_drop and kind_used == "place" and plan_type == EVENT_RELEASE:
                drop = KeyEvent(
                    type=EVENT_ATTACH_DROP,
                    frame=max(iv.start, iv.end - 1) - 1,
                    subtask_index=sub.index,
                    arm=arm,
                    source="gripper_end",
                    anchor_frame=sub.start_frame,
                    interval=(iv.start, iv.end),
                    is_primary=False,
                )
                events.append(drop)
                res.events.append(drop)

    events.sort(key=lambda e: (e.frame, 0 if e.is_primary else 1))
    return events, results


def make_keyposes(
    events: Sequence[KeyEvent],
    streams: Dict[str, ArmStream],
    key_event_types: Iterable[str] = DEFAULT_KEY_EVENT_TYPES,
    monotonic: bool = True,
) -> Dict[str, Any]:
    """Turn primary events into per-episode keypose arrays.

    Returns a dict with ``frames`` (K,), ``arm`` (list), ``event_types`` (list),
    ``epos`` (K, 9) and ``qpos`` (K, 7) -- the first two mirroring the robomimic
    keypose dictionary, the rest being LeRobot specific.
    """
    key_set = set(key_event_types)
    selected = [e for e in events if e.type in key_set]

    if monotonic:
        fixed: List[KeyEvent] = []
        last = -1
        for e in sorted(selected, key=lambda x: x.frame):
            if e.frame <= last:
                e.frame = last + 1
            last = e.frame
            fixed.append(e)
        selected = fixed

    frames: List[int] = []
    epos: List[np.ndarray] = []
    qpos: List[np.ndarray] = []
    arms: List[str] = []
    types: List[str] = []
    for e in selected:
        stream = streams.get(e.arm)
        if stream is None:
            continue
        if not (0 <= e.frame < stream.num_frames):
            continue
        frames.append(int(e.frame))
        epos.append(stream.epos()[e.frame])
        if stream.joint_pos is not None:
            qpos.append(stream.joint_pos[e.frame])
        arms.append(e.arm)
        types.append(e.type)

    out: Dict[str, Any] = {
        "frames": np.asarray(frames, dtype=int),
        "epos": np.asarray(epos, dtype=float) if epos else np.zeros((0, 9)),
        "event_types": types,
        "arm": arms,
    }
    if qpos:
        out["qpos"] = np.asarray(qpos, dtype=float)
    return out


def events_to_json(
    events: Sequence[KeyEvent],
    results: Sequence[SubtaskResult],
    episode_idx: int,
    fps: float,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Serialise events + diagnostics into an inspectable JSON structure."""
    payload: Dict[str, Any] = {
        "episode": episode_idx,
        "fps": fps,
        "source": "lerobot_subtask_annotation",
        "num_events": len(events),
        "events": [e.to_dict() for e in events],
        "subtasks": [r.to_dict() for r in results],
    }
    if extra:
        payload.update(extra)
    return payload

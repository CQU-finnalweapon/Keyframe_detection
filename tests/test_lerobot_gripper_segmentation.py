"""Unit tests for the gripper-segmentation changes made for LeRobot support.

These tests use synthetic signals only, so they run without any dataset on disk
(``tests/test_v3_segmentation.py`` and ``tests/test_correction_rules.py`` need
the robomimic HDF5 files).

Covered behaviour
-----------------
``merge_gripper_hiccups``
    * merges a genuine two-stage closure (close / brief open / close);
    * refuses to merge same-label intervals that are far apart -- the original
      implementation welded the closing action of one subtask to the closing
      action of the next one, producing a single interval spanning most of the
      episode and starving every subtask in between of its keypose;
    * respects ``max_hiccup`` and can be disabled.

``_filter_gripper_labels``
    * masks only the frames where measurement and command actually conflict, so
      a one-frame command/state lag no longer destroys the following closing
      action.

``segment_gripper_timeline``
    * ``sentinel_tail`` toggles the robomimic trailing placeholder.

``resolve_intent``
    * flips a mislabelled ``pick``/``place`` when the gripper says otherwise;
    * infers the intent of an unannotated subtask from the action present.

Run with::

    pytest tests/test_lerobot_gripper_segmentation.py -v
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.lerobot_utils import derive_action_gripper  # noqa: E402
from src.trajectory_segmentation_v3 import (  # noqa: E402
    Interval,
    _filter_gripper_labels,
    filter_gripper_intervals,
    merge_gripper_hiccups,
    merge_interrupted_actions,
    segment_gripper_timeline,
)
from src.subtask_keypose import (  # noqa: E402
    EVENT_GRASP,
    EVENT_RELEASE,
    assign_intervals,
    contains_full_cycle,
    find_other_arm_evidence,
    resolve_intent,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def make_gripper(signal: np.ndarray) -> np.ndarray:
    """Pack a 1-D opening signal into the (N, 2) layout the segmenter expects."""
    signal = np.asarray(signal, dtype=float)
    return np.stack([signal, np.zeros_like(signal)], axis=1)


def ramp(start: float, end: float, n: int) -> np.ndarray:
    return np.linspace(start, end, n, endpoint=False)


def build_two_stage_closure() -> np.ndarray:
    """Gripper that closes, blips open for 2 frames, then closes again.

    Steps are ~4e-3 per frame so they clear the 1e-3 detection threshold.

    Layout (frame ranges are half-open):
        0-19   open hold
        20-25  closing ramp
        26-27  hold
        28-29  brief opening (hiccup)
        30-31  hold
        32-37  closing ramp
        38-49  closed hold
    """
    open_val, mid_val, closed_val = 0.098, 0.070, 0.030
    parts = [
        np.full(20, open_val),
        ramp(open_val, mid_val, 6),
        np.full(2, mid_val),
        ramp(mid_val, 0.078, 2),
        np.full(2, 0.078),
        ramp(0.078, closed_val, 6),
        np.full(12, closed_val),
    ]
    return np.concatenate(parts)


# ---------------------------------------------------------------------------
# merge_gripper_hiccups
# ---------------------------------------------------------------------------
def test_merge_hiccups_merges_contiguous_two_stage_closure():
    intervals = [
        Interval(20, 24, "closing"),
        Interval(26, 28, "opening"),
        Interval(30, 34, "closing"),
    ]
    merged = merge_gripper_hiccups(intervals, max_hiccup=5)
    assert len(merged) == 1, f"expected a single merged interval, got {merged}"
    assert (merged[0].start, merged[0].end) == (20, 34)
    assert merged[0].label == "closing"


def test_merge_hiccups_does_not_merge_distant_intervals():
    """Regression: the closing of subtask N must not swallow subtask N+1."""
    intervals = [
        Interval(345, 354, "closing"),
        Interval(513, 519, "opening"),
        Interval(1111, 1127, "closing"),
    ]
    merged = merge_gripper_hiccups(intervals, max_hiccup=6)
    assert merged == intervals, (
        "distant intervals were merged; this reintroduces the bug where a single "
        f"interval spanned the whole episode: {merged}"
    )


def test_merge_hiccups_rejects_long_blip():
    intervals = [
        Interval(20, 24, "closing"),
        Interval(26, 40, "opening"),  # 14 frames, far too long to be a hiccup
        Interval(42, 46, "closing"),
    ]
    merged = merge_gripper_hiccups(intervals, max_hiccup=5)
    assert merged == intervals


def test_merge_hiccups_rejects_far_apart_but_short_blip():
    """The blip is short, but the parts are not contiguous -> keep them apart."""
    intervals = [
        Interval(20, 24, "closing"),
        Interval(100, 102, "opening"),
        Interval(200, 204, "closing"),
    ]
    merged = merge_gripper_hiccups(intervals, max_hiccup=5)
    assert merged == intervals


def test_merge_hiccups_disabled_and_short_input():
    intervals = [
        Interval(20, 24, "closing"),
        Interval(26, 28, "opening"),
        Interval(30, 34, "closing"),
    ]
    assert merge_gripper_hiccups(intervals, max_hiccup=0) == intervals
    assert merge_gripper_hiccups(intervals[:2], max_hiccup=5) == intervals[:2]
    assert merge_gripper_hiccups([], max_hiccup=5) == []


def test_merge_hiccups_never_bridges_a_large_gap():
    """Invariant: merging only happens across gaps <= max_hiccup.

    The merged span may therefore never exceed the sum of the three parts plus
    ``2 * max_hiccup``.
    """
    max_hiccup = 8
    intervals = [Interval(i, i + 4, "closing") for i in range(0, 60, 10)]
    intervals.insert(1, Interval(4, 6, "opening"))

    merged = merge_gripper_hiccups(intervals, max_hiccup=max_hiccup)

    for iv in merged:
        # reconstruct the parts that could have produced this interval
        assert (iv.end - iv.start) <= 4 + 6 + 4 + 2 * max_hiccup, (
            f"interval {iv} is longer than any legal merge could produce"
        )
    # The first merge is legal: (0,4)+(4,6)+(10,14) with gaps 0 and 4.
    assert (merged[0].start, merged[0].end) == (0, 14)
    # (20,24) and (30,34) are 6 frames apart but the triple needs a blip in
    # between, so they must stay separate.
    assert Interval(20, 24, "closing") in merged


# ---------------------------------------------------------------------------
# _filter_gripper_labels
# ---------------------------------------------------------------------------
def test_filter_gripper_labels_masks_only_the_conflict():
    """A 1-frame state/command lag must not wipe out the whole closing action.

    Command is "close" throughout; the measured opening still rises for a single
    frame (state lags the setpoint) before it starts falling.
    """
    action = np.array([1, 1, 1, 1], dtype=float)
    raw = ["none", "opening", "closing", "closing"]

    filtered = _filter_gripper_labels(raw, action, mask_whole_command_run=False)

    assert filtered[1] == "none", "the conflicting frame should be masked"
    assert filtered[2] == "closing" and filtered[3] == "closing", (
        "valid closing frames after the conflict were discarded "
        f"(the old implementation skipped the whole command run): {filtered}"
    )


def test_filter_gripper_labels_legacy_mode_masks_whole_run():
    """The robomimic default must keep the historical behaviour (zero regression)."""
    action = np.array([1, 1, 1, 1], dtype=float)
    raw = ["none", "opening", "closing", "closing"]

    assert _filter_gripper_labels(raw, action) == ["none"] * 4
    assert _filter_gripper_labels(raw, action, mask_whole_command_run=True) == ["none"] * 4


def test_filter_gripper_labels_masks_repeated_conflict():
    action = np.array([-1, -1, -1, -1], dtype=float)
    raw = ["closing", "closing", "opening", "opening"]

    filtered = _filter_gripper_labels(raw, action, mask_whole_command_run=False)

    assert filtered[:2] == ["none", "none"]
    assert filtered[2:] == ["opening", "opening"], (
        "an opening measured under an opening command is not a conflict"
    )


def test_filter_gripper_labels_passes_through_when_no_command():
    action = np.zeros(3)
    raw = ["closing", "none", "opening"]
    assert _filter_gripper_labels(raw, action) == raw


# ---------------------------------------------------------------------------
# segment_gripper_timeline
# ---------------------------------------------------------------------------
def test_segment_timeline_detects_two_stage_closure():
    signal = build_two_stage_closure()
    gripper = make_gripper(signal)
    # Command tracks the measurement (the usual case on real hardware), so no
    # frame is treated as a correction period.
    action = derive_action_gripper(signal, 1e-4)

    raw = segment_gripper_timeline(
        gripper, action, 1e-3, 5, sentinel_tail=False, mask_whole_command_run=False
    )
    labels = [(iv.start, iv.end, iv.label) for iv in raw]
    # Interval starts are 1-indexed: a run of diffs at frames 20..26 becomes
    # the interval [21, 27).
    assert labels == [(21, 27, "closing"), (30, 31, "opening"), (34, 39, "closing")], labels

    merged = merge_gripper_hiccups(raw, max_hiccup=5)
    assert [(iv.start, iv.end, iv.label) for iv in merged] == [(21, 39, "closing")]


def test_segment_timeline_masked_blip_is_bridged_by_gap_fill():
    """A hiccup that the command filter masks is still absorbed by gap filling.

    With a constant "close" command the brief opening looks like a correction
    period and gets masked; ``_fill_gripper_gaps`` then bridges the hole so the
    result is one closing interval either way.
    """
    signal = build_two_stage_closure()
    gripper = make_gripper(signal)
    action = np.ones(len(signal))  # keep commanding "close"

    raw = segment_gripper_timeline(
        gripper, action, 1e-3, 5, sentinel_tail=False, mask_whole_command_run=False
    )
    labels = [(iv.start, iv.end, iv.label) for iv in raw]

    assert labels == [(21, 39, "closing")], labels


def test_segment_timeline_legacy_masking_swallows_the_rest_of_the_run():
    """Documents the historical robomimic behaviour that we must not regress.

    A conflict masks the remainder of the command run, so a single spurious
    opening under a constant "close" command removes everything after it.
    """
    signal = build_two_stage_closure()
    gripper = make_gripper(signal)
    action = np.ones(len(signal))

    raw = segment_gripper_timeline(
        gripper, action, 1e-3, 5, sentinel_tail=False, mask_whole_command_run=True
    )
    labels = [(iv.start, iv.end, iv.label) for iv in raw]

    assert labels == [(21, 27, "closing")], labels


def test_segment_timeline_sentinel_tail_toggle():
    """The robomimic trailing placeholder must be opt-out for annotated data."""
    signal = np.concatenate([np.full(10, 0.098), ramp(0.098, 0.03, 8), np.full(12, 0.03)])
    gripper = make_gripper(signal)
    action = np.ones(len(signal))

    with_sentinel = segment_gripper_timeline(gripper, action, 1e-3, 5, sentinel_tail=True)
    without = segment_gripper_timeline(gripper, action, 1e-3, 5, sentinel_tail=False)

    assert len(with_sentinel) == len(without) + 1
    last = with_sentinel[-1]
    assert (last.end - last.start) == 1, "the sentinel should be a 1-frame placeholder"


# ---------------------------------------------------------------------------
# resolve_intent
# ---------------------------------------------------------------------------
def test_resolve_intent_trusts_gripper_over_annotation():
    """`Pick up X on Y` that only releases inside its window is really a place."""
    gripper = [Interval(659, 667, "opening")]

    labels, event_type, kind_used, msg = resolve_intent("pick", [0], gripper)

    assert kind_used == "place"
    assert event_type == EVENT_RELEASE
    assert "gripper evidence" in msg
    labels, event_type, kind_used, msg = resolve_intent("place", [0], [Interval(10, 20, "closing")])
    assert (kind_used, event_type) == ("pick", EVENT_GRASP)


def test_resolve_intent_keeps_annotation_when_consistent():
    labels, event_type, kind_used, msg = resolve_intent(
        "pick", [0], [Interval(68, 85, "closing")]
    )
    assert (labels, event_type, kind_used) == (("closing",), EVENT_GRASP, "pick")
    assert msg == ""


def test_resolve_intent_infers_other_from_single_action():
    _labels, event_type, kind_used, msg = resolve_intent(
        "other", [0], [Interval(80, 94, "closing")]
    )
    assert (kind_used, event_type) == ("pick", EVENT_GRASP)
    assert "inferred" in msg

    _labels, event_type, kind_used, _msg = resolve_intent(
        "other", [0], [Interval(80, 94, "opening")]
    )
    assert (kind_used, event_type) == ("place", EVENT_RELEASE)


def test_resolve_intent_stays_ambiguous_with_both_actions():
    gripper = [Interval(1227, 1242, "opening"), Interval(1382, 1403, "closing")]
    _labels, _event_type, kind_used, _msg = resolve_intent("other", [0, 1], gripper)
    assert kind_used == "other", "an unannotated span with both actions stays ambiguous"


def test_resolve_intent_no_candidates_is_inert():
    labels, event_type, kind_used, msg = resolve_intent("pick", [], [])
    assert (labels, event_type, kind_used, msg) == (("closing",), EVENT_GRASP, "pick", "")


# ---------------------------------------------------------------------------
# assign_intervals
# ---------------------------------------------------------------------------
def test_assign_intervals_owner_is_where_the_action_starts():
    """Each interval belongs to the window containing its *start*."""
    gripper = [Interval(80, 94, "closing"), Interval(263, 272, "opening")]
    windows = [(0, 75), (75, 152), (152, 348)]

    assignment = assign_intervals(gripper, windows, radius=15)

    assert assignment[0] == []
    assert assignment[1] == [0], "the closing at frame 80 belongs to window #1"
    assert assignment[2] == [1]


def test_assign_intervals_respects_radius():
    """An interval outside every window is only claimed within the radius."""
    gripper = [Interval(760, 780, "closing")]
    windows = [(517, 704), (900, 1100)]

    without_radius = assign_intervals(gripper, windows, radius=0)
    assert without_radius[0] == [] and without_radius[1] == []

    # frame 760 sits 56 frames after window #0 ends and 140 before #1 starts
    with_radius = assign_intervals(gripper, windows, radius=80)
    assert with_radius[0] == [0], "a small drift outside the window may be tolerated"
    assert with_radius[1] == [], "the far window must not steal the interval"


# ---------------------------------------------------------------------------
# merge_interrupted_actions
# ---------------------------------------------------------------------------
def test_merge_interrupted_actions_bridges_a_slow_plateau():
    """One slow closure split by a sub-threshold plateau becomes one action."""
    # slow descent: fast, then a long plateau, then fast again
    signal = np.concatenate([
        np.full(10, 0.098),
        ramp(0.098, 0.076, 6),      # fast initial closure
        np.full(40, 0.0734),        # plateau below the per-frame threshold
        ramp(0.0734, 0.001, 6),     # final squeeze
        np.full(10, 0.001),
    ])
    intervals = [
        Interval(11, 17, "closing"),
        Interval(57, 63, "closing"),
    ]
    merged = merge_interrupted_actions(intervals, signal, max_gap=60)
    assert len(merged) == 1
    assert (merged[0].start, merged[0].end) == (11, 63)


def test_merge_interrupted_actions_refuses_a_reversal():
    """A same-label pair separated by an upward move must stay apart."""
    signal = np.concatenate([
        np.full(10, 0.098),
        ramp(0.098, 0.060, 6),      # closing
        ramp(0.060, 0.098, 6),      # the gripper opens again
        np.full(10, 0.098),
        ramp(0.098, 0.060, 6),      # a second, independent closing
    ])
    intervals = [Interval(11, 17, "closing"), Interval(33, 39, "closing")]
    merged = merge_interrupted_actions(intervals, signal, max_gap=60)
    assert merged == intervals


def test_merge_interrupted_actions_respects_label_and_gap():
    signal = np.linspace(0.098, 0.001, 200)
    intervals = [
        Interval(20, 30, "closing"),
        Interval(35, 45, "opening"),   # different label -> never merged
        Interval(60, 70, "opening"),
    ]
    assert merge_interrupted_actions(intervals, signal, max_gap=60) == intervals

    far = [Interval(20, 30, "closing"), Interval(200, 210, "closing")]
    assert merge_interrupted_actions(far, signal, max_gap=60) == far


def test_merge_interrupted_actions_disabled():
    signal = np.linspace(0.098, 0.001, 200)
    intervals = [Interval(20, 30, "closing"), Interval(35, 45, "closing")]
    assert merge_interrupted_actions(intervals, signal, max_gap=0) == intervals


# ---------------------------------------------------------------------------
# filter_gripper_intervals: amplitude-aware short-interval handling
# ---------------------------------------------------------------------------
def test_short_interval_with_large_travel_is_kept():
    """A release that snaps fully open in one frame is a real event."""
    signal = np.concatenate([np.full(20, 0.004), np.full(20, 0.098)])
    intervals = [Interval(20, 21, "opening")]  # one frame, 0.094 m of travel

    dropped = filter_gripper_intervals([Interval(20, 21, "opening")], [], min_opening_duration=2)
    assert dropped == [], "without the signal there is no way to tell, so it is dropped"

    kept = filter_gripper_intervals(
        intervals, [], min_opening_duration=2, gripper_signal=signal
    )
    assert len(kept) == 1, "a 0.094 m single-frame release must survive"


def test_short_interval_with_small_travel_is_dropped():
    signal = np.concatenate([np.full(20, 0.0700), np.full(20, 0.0705)])
    intervals = [Interval(20, 21, "opening")]  # 0.5 mm of travel = jitter
    kept = filter_gripper_intervals(
        intervals, [], min_opening_duration=2, gripper_signal=signal
    )
    assert kept == []


# ---------------------------------------------------------------------------
# contains_full_cycle
# ---------------------------------------------------------------------------
def test_contains_full_cycle_detects_a_collapsed_pick_and_place():
    """The window spans grasp -> transport -> release."""
    ivs = [Interval(49, 54, "closing"), Interval(177, 182, "opening")]
    assert contains_full_cycle([0, 1], ivs, (0, 231), min_separation=15)


def test_contains_full_cycle_rejects_a_squeeze_then_release():
    """Closing followed immediately by opening is one action, not a cycle."""
    ivs = [Interval(909, 914, "closing"), Interval(915, 926, "opening")]
    assert not contains_full_cycle([0, 1], ivs, (826, 995), min_separation=15)


def test_contains_full_cycle_needs_both_labels():
    assert not contains_full_cycle([0], [Interval(49, 54, "closing")], (0, 231))
    assert not contains_full_cycle([0], [Interval(49, 54, "opening")], (0, 231))


def test_contains_full_cycle_ignores_intervals_outside_the_window():
    """Intervals claimed by the search radius must not trigger a split."""
    ivs = [Interval(49, 54, "closing"), Interval(240, 245, "opening")]
    assert not contains_full_cycle([0, 1], ivs, (0, 231), min_separation=15)


# ---------------------------------------------------------------------------
# find_other_arm_evidence
# ---------------------------------------------------------------------------
def test_find_other_arm_evidence_prefers_the_expected_label():
    timelines = {
        "left": {"gripper": [Interval(900, 910, "opening")]},
        "right": {"gripper": [Interval(905, 912, "closing")]},
    }
    found = find_other_arm_evidence({"left": None, "right": None}, timelines, "left", (800, 950), ("closing",))
    assert found is not None
    assert found[0] == "right"


def test_find_other_arm_evidence_accepts_a_unique_weak_candidate():
    """Repairs "pick ... with left hand" that the right arm performs as a release."""
    timelines = {
        "left": {"gripper": []},
        "right": {"gripper": [Interval(905, 909, "opening")]},
    }
    found = find_other_arm_evidence({"left": None, "right": None}, timelines, "left", (782, 991), ("closing",))
    assert found is not None and found[0] == "right"


def test_find_other_arm_evidence_stays_silent_when_both_arms_are_active():
    timelines = {
        "left": {"gripper": []},
        "right": {"gripper": [Interval(905, 909, "opening")]},
        "extra": {"gripper": [Interval(910, 914, "opening")]},
    }
    found = find_other_arm_evidence(
        {"left": None, "right": None, "extra": None}, timelines, "left", (782, 991), ("closing",)
    )
    assert found is None


if __name__ == "__main__":
    # Standalone runner: `pytest` on this cluster walks up to /mnt and trips over
    # unreadable mounts, so the tests are also runnable directly.
    failures = []
    tests = [
        (name, obj)
        for name, obj in sorted(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    for name, fn in tests:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failures.append((name, exc))
            print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"ok    {name}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} tests passed")
    sys.exit(1 if failures else 0)

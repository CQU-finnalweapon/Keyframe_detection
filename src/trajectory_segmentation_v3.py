"""
Trajectory Segmentation Module - V3

This module integrates the V3 visualization segmentation logic with gripper analysis.

Architecture:
1. Movement Timeline: Uses visualize_trajectory_v3 logic for per-axis motion analysis
   - Per-axis fast motion detection
   - Raw merged phases (all boundaries)
   - Final combined phases (merge unless reversal)
   - Slow motion detection in stable phases
2. Gripper Timeline: Reuses V2 gripper segmentation logic

Key Features:
- Per-axis motion labels with slow motion detection
- Direction reversal detection for phase boundaries
- Separate movement and gripper timelines
- High temporal resolution with slow motion analysis

Based on: visualize_trajectory_v3.py, trajectory_segmentation_v2.py

Author: TASE Project
Date: Dec 24, 2025
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from scipy.signal import savgol_filter
from dataclasses import dataclass
import sys
import os

# Import visualization v3 functions
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class Interval:
    """Represents a single interval in a timeline."""
    start: int  # Start frame (1-indexed, inclusive)
    end: int    # End frame (1-indexed, exclusive)
    label: str  # 'moving', 'stable', 'opening', 'closing'
    direction: Optional[np.ndarray] = None  # Mean direction vector for moving intervals
    speed: Optional[float] = None  # Mean speed for moving intervals
    motion_signature: Optional[Dict[str, int]] = None  # Per-axis motion: {'x': 1, 'y': 0, 'z': -2}
    
    def __post_init__(self):
        if self.direction is not None and not isinstance(self.direction, np.ndarray):
            self.direction = np.array(self.direction)
    
    def to_string(self) -> str:
        """Generate string representation."""
        if self.motion_signature:
            # Convert motion signature to readable format
            parts = []
            for axis, val in self.motion_signature.items():
                if val == 2:
                    parts.append(f"{axis}⇑")
                elif val == -2:
                    parts.append(f"{axis}⇓")
                elif val == 1:
                    parts.append(f"{axis}↑")
                elif val == -1:
                    parts.append(f"{axis}↓")
            motion_str = ", ".join(parts) if parts else "stable"
            return f"[{self.start}, {self.end})({motion_str})"
        else:
            return f"[{self.start}, {self.end})({self.label})"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        d = {
            'start_step': self.start,
            'end_step': self.end,
            'label': self.label,
        }
        if self.direction is not None:
            d['direction'] = self.direction.tolist()
        if self.speed is not None:
            d['speed'] = float(self.speed)
        if self.motion_signature is not None:
            d['motion_signature'] = self.motion_signature
        return d


# =============================================================================
# Movement Segmentation (from visualize_trajectory_v3.py)
# =============================================================================

# Import constants and functions from visualize_trajectory_v3
TASK_THRESHOLDS = {
    'tool_hang': {'x': 0.022, 'y': 0.046, 'z': 0.049},  # mm/s -> m/s
    'square': {'x': 0.035, 'y': 0.015, 'z': 0.030},
    'can': {'x': 0.030, 'y': 0.070, 'z': 0.040},
}

# Window sizes for smoothing trajectory before motion detection
# Smaller window = can detect shorter motions (but more noise)
# Larger window = smoother but may merge short motions
TASK_WINDOW_SIZES = {
    'tool_hang': 81,  # Longer trajectories, can use larger window
    'square': 41,     # Short insertion motion needs smaller window (was 81)
    'can': 61,        # Medium length trajectories
}

SLOW_MOTION_PARAMS = {
    'displacement_window': 100,
    'displacement_threshold': 0.10,
    'min_slow_segment': 20,
}

def smooth_trajectory(positions: np.ndarray, window_length: int = 15, polyorder: int = 2) -> np.ndarray:
    """Smooth trajectory using Savitzky-Golay filter."""
    n = len(positions)
    if n < window_length:
        window_length = n if n % 2 == 1 else n - 1
        if window_length < 3:
            return positions.copy()
    
    if window_length % 2 == 0:
        window_length += 1
    
    smoothed = np.zeros_like(positions)
    for dim in range(3):
        smoothed[:, dim] = savgol_filter(positions[:, dim], window_length, polyorder)
    
    return smoothed


def compute_motion_labels_per_axis(
    velocities: np.ndarray,
    thresholds: dict,
    stable_window: int = 5
) -> np.ndarray:
    """
    Compute per-axis motion labels using vectorized convolution (from visualization).
    
    Args:
        velocities: (N, 3) array of velocities
        thresholds: Dict with 'x', 'y', 'z' velocity thresholds
        stable_window: Window for smoothing velocity before labeling
        
    Returns:
        (N, 3) array of labels encoded as: -1=down, 0=stable, 1=up
    """
    n = len(velocities)
    labels = np.zeros((n, 3), dtype=int)
    
    axis_thresholds = [thresholds['x'], thresholds['y'], thresholds['z']]
    
    for axis in range(3):
        vel = velocities[:, axis]
        thresh = axis_thresholds[axis]
        
        # Apply moving average smoothing using convolution
        if n >= stable_window:
            kernel = np.ones(stable_window) / stable_window
            vel_smooth = np.convolve(vel, kernel, mode='same')
        else:
            vel_smooth = vel
        
        # Classify each frame
        labels[:, axis] = np.where(vel_smooth > thresh, 1, 
                                   np.where(vel_smooth < -thresh, -1, 0))
    
    return labels


def compute_motion_labels(
    positions: np.ndarray,
    velocities: np.ndarray,
    threshold: float,
    window_size: int = 5
) -> np.ndarray:
    """
    Compute motion labels for a single axis (legacy per-axis version).
    
    Note: This is kept for backward compatibility. Prefer compute_motion_labels_per_axis.
    """
    n = len(positions)
    labels = np.zeros(n, dtype=int)
    
    for i in range(n):
        start_idx = max(0, i - window_size)
        end_idx = min(n, i + window_size + 1)
        
        window_vel = velocities[start_idx:end_idx]
        avg_vel = np.mean(window_vel)
        
        if abs(avg_vel) > threshold:
            labels[i] = 1 if avg_vel > 0 else -1
    
    return labels


def segment_by_labels(labels: np.ndarray, min_segment_length: int = 5) -> List[Tuple[int, int, int]]:
    """Segment an axis based on motion labels."""
    n = len(labels)
    if n == 0:
        return []
    
    segments = []
    current_label = labels[0]
    start = 0
    
    for i in range(1, n):
        if labels[i] != current_label:
            if i - start >= min_segment_length:
                segments.append((start, i, current_label))
            start = i
            current_label = labels[i]
    
    if n - start >= min_segment_length:
        segments.append((start, n, current_label))
    
    return segments


def merge_axis_segments(
    segments_x: List[Tuple[int, int, int]],
    segments_y: List[Tuple[int, int, int]],
    segments_z: List[Tuple[int, int, int]],
    n_frames: int,
    min_segment_length: int = 20
) -> List[dict]:
    """
    Merge per-axis segments into unified motion phases.
    
    Uses frame-wise labels approach from visualization for consistency.
    A merged segment represents a consistent motion pattern across all axes.
    
    Args:
        segments_x, segments_y, segments_z: Per-axis segments (start, end, label)
        n_frames: Total number of frames
        min_segment_length: Minimum frames between boundaries to keep separate
        
    Returns:
        List of dicts with 'start', 'end', 'x', 'y', 'z' (motion labels per axis)
    """
    # Convert segments to frame-wise labels
    labels_x = np.zeros(n_frames, dtype=int)
    labels_y = np.zeros(n_frames, dtype=int)
    labels_z = np.zeros(n_frames, dtype=int)
    
    for start, end, label in segments_x:
        labels_x[start:end] = label
    for start, end, label in segments_y:
        labels_y[start:end] = label
    for start, end, label in segments_z:
        labels_z[start:end] = label
    
    # Find all segment boundaries from all axes
    boundaries = set([0, n_frames])
    for segments in [segments_x, segments_y, segments_z]:
        for start, end, _ in segments:
            boundaries.add(start)
            boundaries.add(end)
    
    boundaries = sorted(boundaries)
    
    # Filter out boundaries that are too close together (merge very short segments)
    filtered_boundaries = [boundaries[0]]
    for b in boundaries[1:]:
        if b - filtered_boundaries[-1] >= min_segment_length // 2:
            filtered_boundaries.append(b)
        # else: skip this boundary (too close to previous)
    
    # Ensure last boundary is included
    if filtered_boundaries[-1] != n_frames:
        filtered_boundaries.append(n_frames)
    
    boundaries = filtered_boundaries
    
    # Create merged segments from filtered boundaries
    merged = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        
        x_label = _dominant_label(labels_x[start:end])
        y_label = _dominant_label(labels_y[start:end])
        z_label = _dominant_label(labels_z[start:end])
        
        merged.append({
            'start': start,
            'end': end,
            'x': x_label,
            'y': y_label,
            'z': z_label
        })
    
    # Post-process: merge adjacent segments with same pattern
    final_merged = []
    for seg in merged:
        if final_merged and _same_pattern(final_merged[-1], seg):
            final_merged[-1]['end'] = seg['end']
        else:
            final_merged.append(seg)
    
    return final_merged


def _dominant_label(labels: np.ndarray) -> int:
    """
    Get the most common non-zero label, or 0 if mostly stable.
    
    Labels: -2=slow down, -1=fast down, 0=stable, 1=fast up, 2=slow up
    Returns the dominant label (keeping slow/fast distinction).
    """
    if len(labels) == 0:
        return 0
    
    # Count all motion types
    up_count = np.sum((labels == 1) | (labels == 2))      # fast or slow up
    down_count = np.sum((labels == -1) | (labels == -2))  # fast or slow down
    stable_count = np.sum(labels == 0)
    
    # Also count slow motion specifically for the label type
    slow_up_count = np.sum(labels == 2)
    slow_down_count = np.sum(labels == -2)
    
    # If stable is dominant (> 60%), return 0
    if stable_count >= len(labels) * 0.6:
        return 0
    
    # Otherwise return the most common direction
    # Use slow label (±2) if slow motion is dominant, else use fast label (±1)
    if up_count >= down_count:
        if up_count > 0:
            return 2 if slow_up_count > np.sum(labels == 1) else 1
        return 0
    else:
        if down_count > 0:
            return -2 if slow_down_count > np.sum(labels == -1) else -1
        return 0


def _same_pattern(seg1: dict, seg2: dict) -> bool:
    """Check if two segments have the same motion pattern."""
    return seg1['x'] == seg2['x'] and seg1['y'] == seg2['y'] and seg1['z'] == seg2['z']


def combine_motion_phases(merged_segments: List[dict], merge_through_stable: bool = False) -> List[dict]:
    """
    Combine merged segments into final motion phases.
    
    Logic: Merge adjacent phases UNLESS an axis REVERSES direction.
    - Reversal = ↑ to ↓ OR ↓ to ↑ (sign change from positive to negative or vice versa)
    - Axis starting/stopping motion (0↔±1) is NOT a reversal
    - Stable phases break continuity (unless merge_through_stable=True)
    
    This captures continuous motion where axes may start/stop at different times,
    but treats direction changes as new phases.
    
    Args:
        merged_segments: List of merged segments with 'start', 'end', 'x', 'y', 'z'
        merge_through_stable: If True, merge motion phases separated by stable phases
                              If False (default), stable phases break the continuity
        
    Returns:
        List of combined phases with same structure
    """
    if not merged_segments:
        return []
    
    combined = [merged_segments[0].copy()]
    
    for seg in merged_segments[1:]:
        prev = combined[-1]
        
        # Check for reversal on any axis
        # Labels: -2=slow down, -1=fast down, 0=stable, 1=fast up, 2=slow up
        # Reversal: direction change (positive to negative or vice versa)
        has_reversal = False
        for axis in ['x', 'y', 'z']:
            prev_dir = prev[axis]
            curr_dir = seg[axis]
            
            # Get direction sign (ignoring slow/fast distinction)
            prev_sign = 1 if prev_dir > 0 else (-1 if prev_dir < 0 else 0)
            curr_sign = 1 if curr_dir > 0 else (-1 if curr_dir < 0 else 0)
            
            # Reversal: both non-zero and opposite signs
            if prev_sign != 0 and curr_sign != 0 and prev_sign * curr_sign < 0:
                has_reversal = True
                break
        
        # Check if transitioning between stable and motion
        prev_is_stable = prev['x'] == 0 and prev['y'] == 0 and prev['z'] == 0
        curr_is_stable = seg['x'] == 0 and seg['y'] == 0 and seg['z'] == 0
        
        # Decide whether to merge
        should_merge = False
        if has_reversal:
            # Never merge on reversal
            should_merge = False
        elif merge_through_stable:
            # Liberal merging: merge unless reversal
            should_merge = True
        else:
            # Conservative merging: stable phases break continuity
            if prev_is_stable and curr_is_stable:
                # Both stable - merge them
                should_merge = True
            elif prev_is_stable or curr_is_stable:
                # Transition between stable and motion - don't merge
                should_merge = False
            else:
                # Both are motion phases - merge if no reversal
                should_merge = True
        
        if should_merge:
            # Merge with previous phase
            combined[-1]['end'] = seg['end']
            # Update labels: take non-zero if available
            for axis in ['x', 'y', 'z']:
                if seg[axis] != 0:
                    if combined[-1][axis] == 0:
                        combined[-1][axis] = seg[axis]
        else:
            # Start new phase
            combined.append(seg.copy())
    
    return combined


def detect_slow_motion_in_phases(
    positions: np.ndarray,
    combined_phases: List[dict],
    displacement_window: int = 100,
    displacement_threshold: float = 0.10,
    min_segment_length: int = 30
) -> List[dict]:
    """Detect slow motion in stable phases and split them."""
    working_range = np.zeros(3)
    for axis in range(3):
        working_range[axis] = positions[:, axis].max() - positions[:, axis].min()
    
    result_phases = []
    
    for phase in combined_phases:
        start, end = phase['start'], phase['end']
        x_label, y_label, z_label = phase['x'], phase['y'], phase['z']
        
        is_stable = x_label == 0 and y_label == 0 and z_label == 0
        
        if not is_stable:
            result_phases.append(phase.copy())
            continue
        
        phase_len = end - start
        if phase_len < min_segment_length:
            result_phases.append(phase.copy())
            continue
        
        frame_labels = np.zeros((phase_len, 3), dtype=int)
        half_window = displacement_window // 2
        
        for axis in range(3):
            if working_range[axis] < 1e-6:
                continue
            
            thresh = displacement_threshold * working_range[axis]
            
            for i in range(phase_len):
                frame_idx = start + i
                win_start = max(start, frame_idx - half_window)
                win_end = min(end - 1, frame_idx + half_window)
                disp = positions[win_end, axis] - positions[win_start, axis]
                
                if abs(disp) >= thresh:
                    frame_labels[i, axis] = 2 if disp > 0 else -2
        
        sub_phases = _segment_slow_motion(frame_labels, start, end, min_segment_length)
        result_phases.extend(sub_phases)
    
    return result_phases


def _segment_slow_motion(
    frame_labels: np.ndarray,
    phase_start: int,
    phase_end: int,
    min_segment_length: int
) -> List[dict]:
    """Segment a stable phase into sub-phases based on slow motion."""
    n = len(frame_labels)
    if n == 0:
        return []
    
    signatures = [tuple(frame_labels[i]) for i in range(n)]
    
    segments = []
    seg_start = 0
    current_sig = signatures[0]
    
    for i in range(1, n):
        if signatures[i] != current_sig:
            segments.append((seg_start, i, current_sig))
            seg_start = i
            current_sig = signatures[i]
    segments.append((seg_start, n, current_sig))
    
    filtered_segments = []
    for seg_start_rel, seg_end_rel, sig in segments:
        seg_len = seg_end_rel - seg_start_rel
        if seg_len >= min_segment_length:
            filtered_segments.append({
                'start': phase_start + seg_start_rel,
                'end': phase_start + seg_end_rel,
                'x': sig[0],
                'y': sig[1],
                'z': sig[2]
            })
        elif filtered_segments:
            filtered_segments[-1]['end'] = phase_start + seg_end_rel
        else:
            filtered_segments.append({
                'start': phase_start + seg_start_rel,
                'end': phase_start + seg_end_rel,
                'x': sig[0],
                'y': sig[1],
                'z': sig[2]
            })
    
    if not filtered_segments:
        return [{'start': phase_start, 'end': phase_end, 'x': 0, 'y': 0, 'z': 0}]
    
    merged = [filtered_segments[0]]
    for seg in filtered_segments[1:]:
        if (merged[-1]['x'] == seg['x'] and
            merged[-1]['y'] == seg['y'] and
            merged[-1]['z'] == seg['z']):
            merged[-1]['end'] = seg['end']
        else:
            merged.append(seg)
    
    # Apply combination logic within sub-phases (skip stable phases)
    combined = [merged[0]]
    for seg in merged[1:]:
        prev_is_stable = combined[-1]['x'] == 0 and combined[-1]['y'] == 0 and combined[-1]['z'] == 0
        curr_is_stable = seg['x'] == 0 and seg['y'] == 0 and seg['z'] == 0
        
        if prev_is_stable or curr_is_stable:
            combined.append(seg.copy())
            continue
        
        can_merge = True
        for axis in ['x', 'y', 'z']:
            prev_label = combined[-1][axis]
            curr_label = seg[axis]
            
            if prev_label * curr_label < 0:
                can_merge = False
                break
        
        if can_merge:
            combined[-1]['end'] = seg['end']
            for axis in ['x', 'y', 'z']:
                if seg[axis] != 0:
                    combined[-1][axis] = seg[axis]
        else:
            combined.append(seg.copy())
    
    return combined


def segment_movement_timeline_v3(
    positions: np.ndarray,
    task: str = 'tool_hang',
    window_size: int = None,
    thresholds: dict = None,
    sampling_interval: float = 0.04
) -> List[Interval]:
    """
    Segment trajectory into movement intervals using V3 logic.
    
    Uses the same segmentation approach as visualize_trajectory_v3.py:
    1. Smooth trajectory using Savitzky-Golay filter
    2. Compute per-axis velocities
    3. Apply convolution-based motion label detection (vectorized)
    4. Segment per axis and merge
    5. Combine phases (merge unless direction reversal)
    6. Detect slow motion in stable phases
    
    Args:
        positions: (N, 3) array of positions
        task: Task name for default thresholds
        window_size: Smoothing window size
        thresholds: Custom thresholds {'x', 'y', 'z'} in m/s
        sampling_interval: Time between samples
    
    Returns:
        List of Interval objects with motion signatures
    """
    n = len(positions)
    if n < 2:
        return [Interval(1, n + 1, "stable")]
    
    # Get task-specific parameters
    if thresholds is None:
        thresholds = TASK_THRESHOLDS.get(task, {'x': 0.030, 'y': 0.030, 'z': 0.030})
    if window_size is None:
        window_size = TASK_WINDOW_SIZES.get(task, 81)
    
    # Smooth trajectory (same as visualization)
    smoothed = smooth_trajectory(positions, window_length=window_size)
    
    # Compute velocities from smoothed trajectory
    velocities = np.zeros_like(positions)
    velocities[1:] = (smoothed[1:] - smoothed[:-1]) / sampling_interval
    
    # Per-axis motion detection using vectorized convolution (same as visualization)
    labels = compute_motion_labels_per_axis(velocities, thresholds, stable_window=window_size // 2)
    
    # Segment per axis
    segments_x = segment_by_labels(labels[:, 0], min_segment_length=20)
    segments_y = segment_by_labels(labels[:, 1], min_segment_length=20)
    segments_z = segment_by_labels(labels[:, 2], min_segment_length=20)
    
    # Merge axes (use min_segment_length for boundary filtering)
    merged_segments = merge_axis_segments(segments_x, segments_y, segments_z, n, min_segment_length=20)
    
    # Combine phases (merge unless direction reversal)
    combined_phases = combine_motion_phases(merged_segments)
    
    # Detect slow motion
    slow_params = SLOW_MOTION_PARAMS
    combined_phases_with_slow = detect_slow_motion_in_phases(
        smoothed, combined_phases,
        displacement_window=slow_params['displacement_window'],
        displacement_threshold=slow_params['displacement_threshold'],
        min_segment_length=slow_params['min_slow_segment']
    )
    
    # Convert to Interval objects (0-based to 1-based)
    intervals = []
    for phase in combined_phases_with_slow:
        is_stable = phase['x'] == 0 and phase['y'] == 0 and phase['z'] == 0
        label = 'stable' if is_stable else 'moving'
        
        # Calculate direction and speed
        phase_pos = positions[phase['start']:phase['end']]
        if len(phase_pos) > 1:
            direction = phase_pos[-1] - phase_pos[0]
            disp = np.linalg.norm(direction)
            norm = np.linalg.norm(direction)
            if norm > 1e-6:
                direction = direction / norm
            else:
                direction = np.zeros(3)
            
            dur = (phase['end'] - phase['start']) * sampling_interval
            speed = disp / dur if dur > 0 else 0
        else:
            direction = np.zeros(3)
            speed = 0.0
        
        interval = Interval(
            start=phase['start'] + 1,  # Convert to 1-based
            end=phase['end'] + 1,
            label=label,
            direction=direction,
            speed=speed,
            motion_signature={'x': phase['x'], 'y': phase['y'], 'z': phase['z']}
        )
        intervals.append(interval)
    
    return intervals


# =============================================================================
# Gripper Segmentation (from V2)
# =============================================================================

def segment_gripper_timeline(
    gripper: np.ndarray,
    action_gripper: np.ndarray,
    gripper_threshold: float = 1e-4,
    max_gap: int = 5,
    sentinel_tail: bool = True,
    mask_whole_command_run: bool = True
) -> List[Interval]:
    """
    Segment gripper actions into opening/closing intervals.
    
    Args:
        gripper: (N, 2) array of gripper positions [left, right]
        action_gripper: (N,) array of gripper commands (-1=open, 1=close, 0=none)
        gripper_threshold: Threshold for gripper change detection
        max_gap: Maximum gap to fill in gripper labels
        sentinel_tail: Append a 1-frame dummy interval at the end when the last
            action is a close (or open). Robomimic needs it to guarantee a
            trailing grasp/release; set False for datasets where the annotation
            already defines the final event.
        mask_whole_command_run: How to treat a measurement that contradicts the
            issued command. True (robomimic default) masks the whole command run,
            which is right for discrete commands. False masks only the frames that
            actually conflict, which is required when the command is a continuous
            setpoint that legitimately lags the measured state by a frame or two.

    Returns:
        List of Interval objects with 'opening' or 'closing' labels
    """
    n = len(gripper)
    if n < 2:
        return []
    
    # Compute gripper changes
    gripper_distance = gripper[:, 0] - gripper[:, 1]
    gripper_diffs = gripper_distance[1:] - gripper_distance[:-1]
    
    # Raw labels
    raw_labels = []
    for i in range(n - 1):
        if gripper_diffs[i] > gripper_threshold:
            raw_labels.append("opening")
        elif gripper_diffs[i] < -gripper_threshold:
            raw_labels.append("closing")
        else:
            raw_labels.append("none")
    
    # Filter using action_gripper (exclude correction periods)
    filtered_labels = _filter_gripper_labels(
        raw_labels, action_gripper, mask_whole_command_run=mask_whole_command_run
    )
    
    # Fill small gaps
    filled_labels = _fill_gripper_gaps(filtered_labels, max_gap)
    
    # Generate intervals from non-none labels
    intervals = []
    i = 0
    while i < len(filled_labels):
        if filled_labels[i] in ["opening", "closing"]:
            start = i
            label = filled_labels[i]
            while i < len(filled_labels) and filled_labels[i] == label:
                i += 1
            intervals.append(Interval(
                start=start + 1,  # 1-indexed
                end=i + 1,
                label=label
            ))
        else:
            i += 1

    # Ensure end release/grasp if missing (Hack for square task etc.)
    # If the last interval is closing, we might be missing a release at the very end.
    # If the last interval is opening, we might be missing a grasp.
    # We add a 1-frame interval at the very end to ensure the logic works.
    if sentinel_tail and n > 0:
        last_state = "none"
        if intervals:
            last_state = intervals[-1].label
        else:
            # Infer from gripper width if no intervals found
            width = gripper[-1, 0] - gripper[-1, 1]
            if width < 0.02: # Closed
                last_state = "closing"
            else:
                last_state = "opening"
        
        if last_state == "closing":
            # Add dummy opening at the end
            intervals.append(Interval(n, n + 1, "opening"))
        elif last_state == "opening":
            # Add dummy closing at the end
            intervals.append(Interval(n, n + 1, "closing"))
    
    return intervals


def merge_gripper_hiccups(
    intervals: List[Interval],
    max_hiccup: int = 5
) -> List[Interval]:
    """Merge ``close -> brief open -> close`` hiccups into a single interval.

    Real grippers often close in two stages (contact, then squeeze) with a
    one-to-five frame blip of opposite sign in between.  Treating that blip as a
    real direction change splits one grasp into two events, and the resulting
    close-open pair then looks like a correction motion -- so it would be
    dropped entirely.

    The three intervals must form a *local* pattern: the opposing blip has to be
    short **and** all three have to be contiguous (no long idle gap in between).
    Without the contiguity requirement this function happily welds the closing
    action of one subtask to the closing action of the next one, producing a
    single interval that spans most of the episode.

    Args:
        intervals: chronological gripper intervals.
        max_hiccup: longest opposing interval (frames) that may be absorbed, and
            also the largest idle gap tolerated between the merged parts.

    Returns:
        Merged interval list.  A no-op when ``max_hiccup <= 0``.
    """
    if max_hiccup <= 0 or len(intervals) < 3:
        return list(intervals)

    ordered = sorted(intervals, key=lambda iv: iv.start)
    merged: List[Interval] = []
    k = 0
    while k < len(ordered):
        if k + 2 < len(ordered):
            a, b, c = ordered[k], ordered[k + 1], ordered[k + 2]
            gap_ab = b.start - a.end
            gap_bc = c.start - b.end
            if (
                a.label == c.label
                and a.label != b.label
                and (b.end - b.start) <= max_hiccup
                and 0 <= gap_ab <= max_hiccup
                and 0 <= gap_bc <= max_hiccup
            ):
                merged.append(Interval(start=a.start, end=c.end, label=a.label))
                k += 3
                continue
        merged.append(ordered[k])
        k += 1
    return merged


def merge_interrupted_actions(
    intervals: List[Interval],
    gripper_signal: np.ndarray,
    max_gap: int = 60,
    reversal_tolerance: float = 1e-3
) -> List[Interval]:
    """Merge consecutive same-label intervals interrupted by a slow plateau.

    A physical gripper action runs until the gripper *reverses*, not until it
    merely slows down.  On real hardware a slow closure easily spends tens of
    frames moving less than the per-frame detection threshold, so the naive diff
    segmentation chops one grasp into several pieces -- which then look like
    repeated grasps.

    Two intervals are merged when they carry the same label and the gripper
    signal across the idle gap keeps moving the same way (within
    ``reversal_tolerance``).  Because the merge requires identical labels that are
    *adjacent* in the timeline, a real reversal (which produces an opposite-label
    interval) always blocks it, so distant actions can never be welded together.

    Args:
        intervals: chronological gripper intervals.
        gripper_signal: (N,) or (N, 2) measured gripper opening.
        max_gap: Largest idle gap (frames) to look across.
        reversal_tolerance: Metres of opposing drift tolerated inside the gap.

    Returns:
        Merged interval list.
    """
    if len(intervals) < 2 or max_gap <= 0:
        return list(intervals)

    signal = np.asarray(gripper_signal, dtype=float)
    if signal.ndim == 2:
        signal = signal[:, 0] - signal[:, 1]
    signal = signal.reshape(-1)

    ordered = sorted(intervals, key=lambda iv: iv.start)
    merged: List[Interval] = [ordered[0]]

    for iv in ordered[1:]:
        prev = merged[-1]
        gap = iv.start - prev.end
        if iv.label != prev.label or not (0 <= gap <= max_gap):
            merged.append(iv)
            continue

        # Signal value entering the gap vs leaving it.
        before = float(signal[min(max(prev.end - 1, 0), len(signal) - 1)])
        after = float(signal[min(max(iv.start - 1, 0), len(signal) - 1)])
        delta = after - before

        if prev.label == "closing":  # a closure drives the opening downwards
            continues = delta <= reversal_tolerance
        else:  # an opening drives it upwards
            continues = delta >= -reversal_tolerance

        if continues:
            merged[-1] = Interval(start=prev.start, end=iv.end, label=prev.label)
        else:
            merged.append(iv)

    return merged


def filter_gripper_intervals(
    gripper_intervals: List[Interval],
    movement_intervals: List[Interval],
    min_movement_between: int = 10,
    max_close_open_gap: int = 30,
    min_closing_duration: int = 4,  # Minimum frames for a valid closing interval
    min_opening_duration: int = 1,  # Minimum frames for a valid opening interval
    gripper_signal: Optional[np.ndarray] = None,
    min_short_travel: float = 5e-3
) -> List[Interval]:
    """Filter gripper intervals (reused from V2).
    
    Filters out:
    1. Short intervals (likely noise / sensor blips)
    2. Close-open pairs with minimal movement between (correction periods)

    A short interval is only rejected when it also moves the gripper very little:
    a release that snaps fully open in a single frame carries ~0.1 m of travel and
    is a real event, while a one-frame jitter of a millimetre is not.  Supply
    ``gripper_signal`` to enable that amplitude test.
    """
    if not gripper_intervals:
        return []

    signal: Optional[np.ndarray] = None
    if gripper_signal is not None:
        signal = np.asarray(gripper_signal, dtype=float)
        if signal.ndim == 2:
            signal = signal[:, 0] - signal[:, 1]
        signal = signal.reshape(-1)

    def _short_but_real(iv: Interval) -> bool:
        """A short interval that nevertheless moves the gripper a long way."""
        if signal is None or min_short_travel <= 0 or signal.size == 0:
            return False
        lo = max(0, min(iv.start - 1, signal.size - 1))
        hi = max(0, min(iv.end - 1, signal.size - 1))
        return abs(float(signal[hi]) - float(signal[lo])) >= min_short_travel

    sorted_intervals = sorted(gripper_intervals, key=lambda x: x.start)
    
    # First pass: filter out short intervals (noise)
    duration_filtered = []
    for iv in sorted_intervals:
        duration = iv.end - iv.start
        min_duration = min_closing_duration if iv.label == 'closing' else min_opening_duration
        if duration < min_duration and not _short_but_real(iv):
            continue
        duration_filtered.append(iv)
    
    filtered = []
    
    first_closing_idx = -1
    for i, iv in enumerate(duration_filtered):
        if iv.label == 'closing':
            first_closing_idx = i
            break
    
    start_idx = first_closing_idx if first_closing_idx >= 0 else 0
    
    i = start_idx
    while i < len(duration_filtered):
        iv = duration_filtered[i]
        
        if iv.label == 'closing' and i + 1 < len(duration_filtered):
            next_iv = duration_filtered[i + 1]
            
            if next_iv.label == 'opening':
                gap = next_iv.start - iv.end
                
                if gap <= max_close_open_gap:
                    moving_frames = 0
                    for mv in movement_intervals:
                        if mv.label == 'moving' and mv.speed is not None and mv.speed > 0.05:
                            overlap_start = max(mv.start, iv.end)
                            overlap_end = min(mv.end, next_iv.start)
                            if overlap_end > overlap_start:
                                moving_frames += overlap_end - overlap_start
                    
                    if moving_frames < min_movement_between:
                        i += 2
                        continue
        
        filtered.append(iv)
        i += 1
    
    return filtered


def _filter_gripper_labels(
    raw_labels: List[str],
    action_gripper: np.ndarray,
    mask_whole_command_run: bool = True
) -> List[str]:
    """Filter gripper labels to exclude periods where the measurement conflicts
    with the issued gripper command.

    ``mask_whole_command_run=True`` reproduces the historical robomimic behaviour:
    the moment a conflict appears, the rest of that command run is masked.  That
    is appropriate for discrete commands, but destroys valid motion whenever the
    state lags a continuous setpoint by a frame or two -- hence the opt-out, which
    masks only the frames that actually contradict the command and therefore keeps
    a superset of the intervals.
    """
    n = len(raw_labels)
    filtered = ["none"] * n
    
    i = 0
    while i < n:
        action = action_gripper[i]
        raw_label = raw_labels[i]
        
        if action == -1 and raw_label == "closing":
            # Command opens but the gripper keeps closing: mask the conflict.
            filtered[i] = "none"
            j = i + 1
            while j < n and action_gripper[j] == -1 and (
                mask_whole_command_run or raw_labels[j] == "closing"
            ):
                filtered[j] = "none"
                j += 1
            i = j
            continue
        elif action == 1 and raw_label == "opening":
            # Command closes but the gripper keeps opening: mask the conflict.
            filtered[i] = "none"
            j = i + 1
            while j < n and action_gripper[j] == 1 and (
                mask_whole_command_run or raw_labels[j] == "opening"
            ):
                filtered[j] = "none"
                j += 1
            i = j
            continue
        else:
            filtered[i] = raw_label
        
        i += 1
    
    return filtered


def _fill_gripper_gaps(labels: List[str], max_gap: int) -> List[str]:
    """Fill small gaps in gripper labels."""
    filled = labels.copy()
    n = len(filled)
    
    for label_type in ["opening", "closing"]:
        i = 0
        while i < n:
            if filled[i] == label_type:
                # Find end of this segment
                j = i + 1
                while j < n and filled[j] == label_type:
                    j += 1
                
                # Look for small gap followed by same type
                if j + max_gap < n:
                    for k in range(j + 1, min(j + max_gap + 2, n)):
                        if filled[k] == label_type:
                            # Fill the gap
                            for m in range(j, k):
                                filled[m] = label_type
                            break
                        elif filled[k] not in ["none", label_type]:
                            break
                
                i = j
            else:
                i += 1
    
    return filled


# =============================================================================
# Main API: Segment Trajectory V3
# =============================================================================

def segment_trajectory_v3(
    trajectory: np.ndarray,
    action_gripper: np.ndarray,
    task: str = 'tool_hang',
    window_size: int = None,
    thresholds: dict = None,
    gripper_threshold: float = 1e-4,
    max_gap: int = 5,
    sampling_interval: float = 0.04,
    min_movement_between: int = 10,
    max_close_open_gap: int = 30,
    min_closing_duration: int = 4,
    min_opening_duration: int = 1,
    merge_hiccup_gaps: int = 0,
    interrupted_action_gap: int = 0,
    min_short_travel: float = 0.0,
    mask_whole_command_run: bool = True,
    sentinel_tail: bool = True
) -> Dict[str, List[Interval]]:
    """
    Segment trajectory using V3 logic.
    
    Args:
        trajectory: (N, 9) array [pos(3), quat(4), gripper(2)]
        action_gripper: (N,) array of gripper commands
        task: Task name for default parameters
        window_size: Smoothing window size
        thresholds: Custom thresholds {'x', 'y', 'z'} in m/s
        gripper_threshold: Gripper change threshold
        max_gap: Maximum gap to fill
        sampling_interval: Time between samples
        min_movement_between: Minimum moving frames between a close/open pair for
            the pair to count as a real grasp-release cycle (otherwise it is a
            correction motion and gets dropped)
        max_close_open_gap: Max frame gap between a close and the following open
            for the pair to be considered a correction motion
        min_closing_duration: Closing intervals shorter than this are dropped as noise
        min_opening_duration: Opening intervals shorter than this are dropped as noise
        merge_hiccup_gaps: Merge `close -> brief open -> close` blips up to this
            many frames into a single closing interval (0 = disable)
        interrupted_action_gap: Merge same-label intervals separated by an idle
            plateau of up to this many frames, provided the gripper signal does
            not reverse inside the gap (0 = disable).  Needed for real grippers,
            whose slow closures dip below the per-frame threshold and would
            otherwise be split into several fake actions.
        min_short_travel: Metres of travel that rescue a too-short interval.
            A release that snaps fully open in one frame is real; 0 (the default,
            matching the robomimic pipeline) disables the rescue.
        mask_whole_command_run: True (robomimic default) masks the whole command
            run from the first conflict; False masks only the conflicting frames,
            which is needed for continuous setpoint commands.  See
            :func:`_filter_gripper_labels`.
        sentinel_tail: Append a dummy trailing gripper interval (robomimic hack);
            keep True for robomimic, False when annotations already delimit events
    
    Returns:
        Dict with 'movement' and 'gripper' timeline lists
    """
    n = len(trajectory)
    if n < 2:
        return {'movement': [], 'gripper': []}
    
    positions = trajectory[:, :3]
    gripper = trajectory[:, 7:]
    
    # Movement timeline (V3 logic)
    movement_intervals = segment_movement_timeline_v3(
        positions, task, window_size, thresholds, sampling_interval
    )
    
    # Gripper timeline (V2 logic)
    gripper_intervals = segment_gripper_timeline(
        gripper, action_gripper, gripper_threshold, max_gap,
        sentinel_tail=sentinel_tail,
        mask_whole_command_run=mask_whole_command_run
    )
    
    # Merge two-stage closures into single intervals (real grippers squeeze twice)
    gripper_intervals = merge_gripper_hiccups(gripper_intervals, max_hiccup=merge_hiccup_gaps)

    # Merge same-direction pieces separated by a slow plateau (a physical action
    # runs until the gripper reverses, not until it slows down)
    if interrupted_action_gap > 0:
        gripper_intervals = merge_interrupted_actions(
            gripper_intervals, gripper, max_gap=interrupted_action_gap
        )
    gripper_intervals = filter_gripper_intervals(
        gripper_intervals, movement_intervals,
        min_movement_between=min_movement_between,
        max_close_open_gap=max_close_open_gap,
        min_closing_duration=min_closing_duration,
        min_opening_duration=min_opening_duration,
        gripper_signal=gripper,
        min_short_travel=min_short_travel
    )
    
    return {
        'movement': movement_intervals,
        'gripper': gripper_intervals
    }


def format_timelines_for_output(timelines: Dict[str, List[Interval]]) -> Dict:
    """Format timeline output for JSON serialization."""
    result = {
        'movement_timeline': [],
        'gripper_timeline': [],
        'movement_intervals': {
            'moving': [],
            'stable': []
        },
        'gripper_intervals': {
            'opening': [],
            'closing': []
        }
    }
    
    for interval in timelines['movement']:
        result['movement_timeline'].append(interval.to_string())
        d = interval.to_dict()
        d['movement'] = interval.label
        d['used'] = False
        if interval.label == 'moving':
            result['movement_intervals']['moving'].append(d)
        else:
            result['movement_intervals']['stable'].append(d)
    
    for interval in timelines['gripper']:
        result['gripper_timeline'].append(interval.to_string())
        d = interval.to_dict()
        d['gripper'] = interval.label
        d['used'] = False
        if interval.label == 'opening':
            result['gripper_intervals']['opening'].append(d)
        else:
            result['gripper_intervals']['closing'].append(d)
    
    return result


def format_timelines_readable(timelines: Dict[str, List[Interval]], episode: int = 0) -> str:
    """Format timelines as human-readable text."""
    lines = []
    lines.append(f"Trajectory Segmentation V3 - Episode {episode}")
    lines.append("=" * 80)
    lines.append("")
    
    # Get max frame
    max_frame = max(
        max(iv.end for iv in timelines['movement']) if timelines['movement'] else 0,
        max(iv.end for iv in timelines['gripper']) if timelines['gripper'] else 0
    )
    
    # Movement timeline
    lines.append("MOVEMENT TIMELINE")
    lines.append("-" * 60)
    # Add segment summary
    moving_count = sum(1 for iv in timelines['movement'] if iv.label == 'moving')
    stable_count = sum(1 for iv in timelines['movement'] if iv.label == 'stable')
    lines.append(f"  Segments: {moving_count} moving, {stable_count} stable")
    lines.append("")
    for iv in timelines['movement']:
        # Format interval with motion signature like visualization
        # Convert dict {x: 1, y: -1, z: 0} to "x↑, y↓" format
        motion_sig = ""
        if hasattr(iv, 'motion_signature') and iv.motion_signature:
            if isinstance(iv.motion_signature, dict):
                parts = []
                for axis in ['x', 'y', 'z']:
                    val = iv.motion_signature.get(axis, 0)
                    if val == 2:
                        parts.append(f"{axis}⇑")
                    elif val == -2:
                        parts.append(f"{axis}⇓")
                    elif val == 1:
                        parts.append(f"{axis}↑")
                    elif val == -1:
                        parts.append(f"{axis}↓")
                motion_sig = ", ".join(parts) if parts else "stable"
            else:
                motion_sig = str(iv.motion_signature)
        
        # Build formatted output with aligned columns:
        # [   1,   48) moving    y↓, z↓       dir=[0.09, -0.72, -0.69] speed=0.125m/s
        # [  48,   73) stable    stable       dir=[-0.23, -0.69, -0.69]
        # Fixed widths: motion_sig=12 chars to align "dir=" column
        dir_str = ""
        if hasattr(iv, 'direction') and iv.direction is not None:
            dir_str = f"dir=[{iv.direction[0]:5.2f}, {iv.direction[1]:5.2f}, {iv.direction[2]:5.2f}]"
        
        speed_str = ""
        if hasattr(iv, 'speed') and iv.speed is not None and iv.label == 'moving':
            speed_str = f" speed={iv.speed:.3f}m/s"
        
        # Align columns with fixed widths
        interval_part = f"[{iv.start:4d}, {iv.end:4d})"
        label_part = f"{iv.label:7s}"  # "moving " or "stable "
        motion_part = f"{motion_sig:12s}"  # Fixed width for motion signature
        
        lines.append(f"  {interval_part} {label_part} {motion_part} {dir_str}{speed_str}")
    
    lines.append("")
    
    # Gripper timeline
    lines.append("GRIPPER TIMELINE")
    lines.append("-" * 80)
    for iv in timelines['gripper']:
        lines.append(f"  {iv.to_string()}")
    
    lines.append("")
    
    # Vertical timeline
    lines.append("VERTICAL TIMELINE")
    lines.append("-" * 80)
    lines.append(f"{'Frame':>6} │ {'Movement':^25} │ {'Gripper':^20} │ {'Key':^5}")
    lines.append("─" * 6 + "─┼─" + "─" * 25 + "─┼─" + "─" * 20 + "─┼─" + "─" * 5)
    
    key_frames = set()
    for iv in timelines['movement']:
        key_frames.add(iv.start)
        key_frames.add(iv.end)
    for iv in timelines['gripper']:
        key_frames.add(iv.start)
        key_frames.add(iv.end)
    
    if max_frame > 100:
        for f in range(0, max_frame + 1, 20):
            key_frames.add(f)
    elif max_frame > 50:
        for f in range(0, max_frame + 1, 10):
            key_frames.add(f)
    
    key_frames = sorted(key_frames)
    
    for frame in key_frames:
        movement_label = ""
        gripper_label = ""
        
        for iv in timelines['movement']:
            if iv.start == frame:
                sig_parts = []
                if iv.motion_signature:
                    for axis, val in iv.motion_signature.items():
                        if val == 2:
                            sig_parts.append(f"{axis}⇑")
                        elif val == -2:
                            sig_parts.append(f"{axis}⇓")
                        elif val == 1:
                            sig_parts.append(f"{axis}↑")
                        elif val == -1:
                            sig_parts.append(f"{axis}↓")
                sig_str = ", ".join(sig_parts) if sig_parts else "stable"
                if iv.label == 'stable':
                    movement_label = f"▶ stable"
                else:
                    movement_label = f"▶ {sig_str}"
            elif iv.end == frame:
                movement_label = f"  └─ end"
        
        for iv in timelines['gripper']:
            if iv.start == frame:
                gripper_label = f"▶ {iv.label}"
            elif iv.end == frame:
                gripper_label = f"  └─ end"
        
        if movement_label or gripper_label:
            lines.append(f"{frame:6d} │ {movement_label:25s} │ {gripper_label:20s} │     ")
    
    lines.append("─" * 6 + "─┴─" + "─" * 25 + "─┴─" + "─" * 20 + "─┴─" + "─" * 5)
    lines.append("")
    
    # Summary
    lines.append("SUMMARY")
    lines.append("-" * 40)
    moving_count = sum(1 for iv in timelines['movement'] if iv.label == 'moving')
    stable_count = sum(1 for iv in timelines['movement'] if iv.label == 'stable')
    opening_count = sum(1 for iv in timelines['gripper'] if iv.label == 'opening')
    closing_count = sum(1 for iv in timelines['gripper'] if iv.label == 'closing')
    lines.append(f"  Movement: {moving_count} moving, {stable_count} stable")
    lines.append(f"  Gripper:  {opening_count} opening, {closing_count} closing")
    lines.append(f"  Total frames: {max_frame}")
    lines.append("")
    lines.append("=" * 80)
    
    return "\n".join(lines)


def parse_trajectory_intervals_v3(formatted_output: Dict) -> Dict[str, List[Dict]]:
    """Convert V3 output to format expected by event_corrector."""
    return {
        'closing': formatted_output['gripper_intervals']['closing'],
        'opening': formatted_output['gripper_intervals']['opening'],
        'moving': formatted_output['movement_intervals']['moving'],
        'stable': formatted_output['movement_intervals']['stable']
    }


if __name__ == "__main__":
    print("Trajectory Segmentation V3 - Test")
    print("=" * 80)
    
    # Simple test
    n = 100
    trajectory = np.zeros((n, 9))
    action_gripper = np.zeros(n)
    
    # Simulate motion
    trajectory[:30, 0] = np.linspace(0, 0.3, 30)
    trajectory[30:50, 0] = 0.3
    trajectory[50:80, 2] = np.linspace(0, 0.2, 30)
    trajectory[80:, 2] = 0.2
    
    # Gripper
    trajectory[:, 7] = 0.04
    trajectory[:, 8] = -0.04
    trajectory[40:80, 7] = 0.0
    trajectory[40:80, 8] = 0.0
    
    action_gripper[35:45] = 1
    action_gripper[75:85] = -1
    
    trajectory[:, 6] = 1.0
    
    # Run V3 segmentation
    timelines = segment_trajectory_v3(trajectory, action_gripper, task='tool_hang')
    
    print("\n" + format_timelines_readable(timelines, episode=0))

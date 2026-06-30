#!/usr/bin/env python3
"""
Trajectory Visualization V3 - Per-Axis Motion Analysis

This version focuses on:
1. Smoothed trajectory for better motion pattern recognition
2. Per-axis (x, y, z) motion labels and velocities
3. Configurable thresholds per axis and per task
4. Clear visualization of motion segments for segmentation debugging

Imports segmentation logic from trajectory_segmentation_v3.py to maintain consistency.

Author: TASE Project
Date: Dec 22, 2025
"""

import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import os
import sys

# Import segmentation functions and constants
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.trajectory_segmentation_v3 import (
    smooth_trajectory,
    compute_motion_labels_per_axis,
    segment_by_labels,
    merge_axis_segments,
    combine_motion_phases,
    detect_slow_motion_in_phases,
    TASK_THRESHOLDS as SEG_TASK_THRESHOLDS,
    TASK_WINDOW_SIZES,
    SLOW_MOTION_PARAMS
)

# Font settings
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
# wordsize = 16
# ticksize = 20
# plt.rc('font', size=wordsize)
# plt.rc('axes', titlesize=wordsize)
# plt.rc('axes', labelsize=wordsize)
# plt.rc('xtick', labelsize=ticksize)
# plt.rc('ytick', labelsize=ticksize)
# plt.rc('legend', fontsize=wordsize)
# plt.rc('figure', titlesize=wordsize)

@dataclass
class AxisThresholds:
    """Thresholds for per-axis motion detection."""
    x: float = 5e-3  # m/s
    y: float = 5e-3  # m/s
    z: float = 5e-3  # m/s


# Re-export for backward compatibility (convert dict format to AxisThresholds)
TASK_THRESHOLDS = {
    task: AxisThresholds(**thresholds) 
    for task, thresholds in SEG_TASK_THRESHOLDS.items()
}


def load_trajectory(hdf5_path: str, episode: int) -> tuple:
    """
    Load trajectory and gripper data from HDF5 file.
    
    Uses obs/robot0_eef_pos instead of actions because:
    - obs/eef_pos = actual executed trajectory (measured positions)
    - actions = commanded trajectory (desired positions with ~16mm tracking error)
    
    For motion analysis and keypose detection, we want the actual physical motion.
    
    Returns:
        positions: (N, 3) array of positions
        gripper: (N, 2) array of gripper positions [left, right]
        action_gripper: (N,) array of gripper commands (-1=open, 1=close, 0=none)
    """
    with h5py.File(hdf5_path, 'r') as f:
        demo_key = f'data/demo_{episode}'
        # Use actual positions instead of commanded positions
        positions = f[demo_key]['obs']['robot0_eef_pos'][()]
        gripper = f[demo_key]['obs']['robot0_gripper_qpos'][()]
        action_gripper = f[demo_key]['actions'][()][:, -1]  # Last column is gripper action
    
    return positions, gripper, action_gripper


def compute_per_axis_velocities(positions: np.ndarray, sampling_interval: float = 0.04) -> np.ndarray:
    """
    Compute velocity for each axis separately.
    
    Args:
        positions: (N, 3) array of positions
        sampling_interval: Time between frames (seconds)
        
    Returns:
        (N, 3) array of velocities (first row is zero)
    """
    velocities = np.zeros_like(positions)
    velocities[1:] = (positions[1:] - positions[:-1]) / sampling_interval
    return velocities


def compute_motion_labels(
    velocities: np.ndarray,
    thresholds: AxisThresholds,
    stable_window: int = 5
) -> np.ndarray:
    """
    Compute per-axis motion labels using the imported segmentation function.
    
    Args:
        velocities: (N, 3) array of velocities
        thresholds: Per-axis velocity thresholds
        stable_window: Window for smoothing velocity before labeling
        
    Returns:
        (N, 3) array of labels encoded as: -1=down, 0=stable, 1=up
    """
    # Convert AxisThresholds to dict format for segmentation module
    threshold_dict = {'x': thresholds.x, 'y': thresholds.y, 'z': thresholds.z}
    return compute_motion_labels_per_axis(velocities, threshold_dict, stable_window)


def detect_slow_motion(
    positions: np.ndarray,
    labels: np.ndarray,
    displacement_window: int = 100,
    displacement_threshold: float = 0.15,
    min_segment_length: int = 30
) -> np.ndarray:
    """
    Detect slow motion regions where velocity is below threshold but 
    cumulative displacement is significant.
    
    Uses bidirectional sliding window to compute displacement and marks 
    stable regions (label=0) as slow motion if displacement exceeds threshold.
    
    Slow motion labels: -2=slow down, 2=slow up (to distinguish from fast motion)
    
    Args:
        positions: (N, 3) smoothed positions
        labels: (N, 3) existing motion labels (-1, 0, 1)
        displacement_window: Frames to look back/forward for displacement
        displacement_threshold: Fraction of working range to trigger slow motion
        min_segment_length: Minimum consecutive frames for slow motion
        
    Returns:
        (N, 3) updated labels with slow motion markers
    """
    n = len(positions)
    labels = labels.copy()  # Don't modify original
    
    # Calculate working range for each axis
    working_range = np.zeros(3)
    for axis in range(3):
        working_range[axis] = positions[:, axis].max() - positions[:, axis].min()
    
    # Compute displacement over window for each axis
    for axis in range(3):
        if working_range[axis] < 1e-6:  # Avoid division by zero
            continue
            
        thresh = displacement_threshold * working_range[axis]
        half_window = displacement_window // 2
        
        # For each stable frame, check displacement in both directions
        for i in range(n):
            if labels[i, axis] != 0:
                continue
            
            # Look backward
            back_start = max(0, i - displacement_window)
            disp_back = positions[i, axis] - positions[back_start, axis]
            
            # Look forward
            fwd_end = min(n - 1, i + displacement_window)
            disp_fwd = positions[fwd_end, axis] - positions[i, axis]
            
            # Use centered window for smoother detection
            center_start = max(0, i - half_window)
            center_end = min(n - 1, i + half_window)
            disp_center = positions[center_end, axis] - positions[center_start, axis]
            
            # Check if any window shows significant displacement
            # And ensure the displacement direction is consistent (same sign)
            disps = [disp_back, disp_fwd, disp_center]
            
            for disp in disps:
                if abs(disp) >= thresh:
                    labels[i, axis] = 2 if disp > 0 else -2
                    break
        
        # Post-process: fill gaps in slow motion regions
        # If we have slow up -> stable -> slow up, fill the stable gap
        slow_labels = labels[:, axis].copy()
        for direction, slow_label in [(1, 2), (-1, -2)]:
            in_slow = False
            seg_start = 0
            gap_start = -1
            
            for i in range(n):
                if slow_labels[i] == slow_label:
                    if not in_slow:
                        # Starting a slow segment
                        if gap_start >= 0 and (i - gap_start) < min_segment_length:
                            # Fill the gap
                            labels[gap_start:i, axis] = slow_label
                        in_slow = True
                        seg_start = i
                elif slow_labels[i] == 0:
                    if in_slow:
                        # Ending a slow segment, mark start of potential gap
                        gap_start = i
                        in_slow = False
                else:
                    # Different label, reset
                    in_slow = False
                    gap_start = -1
        
        # Clean up short slow motion segments
        for slow_label in [2, -2]:
            slow_mask = labels[:, axis] == slow_label
            in_segment = False
            seg_start = 0
            
            for i in range(n):
                if slow_mask[i] and not in_segment:
                    in_segment = True
                    seg_start = i
                elif not slow_mask[i] and in_segment:
                    in_segment = False
                    seg_len = i - seg_start
                    if seg_len < min_segment_length:
                        labels[seg_start:i, axis] = 0
            
            if in_segment:
                seg_len = n - seg_start
                if seg_len < min_segment_length:
                    labels[seg_start:, axis] = 0
    
    return labels


def detect_slow_motion_in_phases(
    positions: np.ndarray,
    combined_phases: List[dict],
    n_frames: int,
    displacement_window: int = 100,
    displacement_threshold: float = 0.10,
    min_segment_length: int = 30
) -> List[dict]:
    """
    Detect slow motion in stable phases and SPLIT them into sub-phases.
    
    This is applied AFTER the raw->final phase combination. Stable phases
    are analyzed frame-by-frame and split into:
    - Slow motion segments (where displacement over window exceeds threshold)
    - Truly stable segments (where displacement is below threshold)
    
    Args:
        positions: (N, 3) smoothed positions
        combined_phases: List of combined phases from combine_motion_phases()
        n_frames: Total number of frames
        displacement_window: Frames to look back/forward for displacement
        displacement_threshold: Fraction of working range to trigger slow motion
        min_segment_length: Minimum frames for a segment to be kept
        
    Returns:
        List of phases with stable phases potentially split into slow motion sub-phases
    """
    # Calculate working range for each axis
    working_range = np.zeros(3)
    for axis in range(3):
        working_range[axis] = positions[:, axis].max() - positions[:, axis].min()
    
    result_phases = []
    
    for phase in combined_phases:
        start, end = phase['start'], phase['end']
        x_label, y_label, z_label = phase['x'], phase['y'], phase['z']
        
        # Check if this is a stable phase (all axes = 0)
        is_stable = x_label == 0 and y_label == 0 and z_label == 0
        
        if not is_stable:
            # Keep motion phases unchanged
            result_phases.append(phase.copy())
            continue
        
        # For stable phases, analyze frame-by-frame for slow motion
        phase_len = end - start
        if phase_len < min_segment_length:
            result_phases.append(phase.copy())
            continue
        
        # Compute frame-wise slow motion labels for each axis
        frame_labels = np.zeros((phase_len, 3), dtype=int)
        half_window = displacement_window // 2
        
        for axis in range(3):
            if working_range[axis] < 1e-6:
                continue
            
            thresh = displacement_threshold * working_range[axis]
            
            for i in range(phase_len):
                frame_idx = start + i
                
                # Use centered window
                win_start = max(start, frame_idx - half_window)
                win_end = min(end - 1, frame_idx + half_window)
                
                disp = positions[win_end, axis] - positions[win_start, axis]
                
                if abs(disp) >= thresh:
                    frame_labels[i, axis] = 2 if disp > 0 else -2
        
        # Segment the stable phase based on slow motion labels
        sub_phases = _segment_slow_motion(frame_labels, start, end, min_segment_length)
        result_phases.extend(sub_phases)
    
    return result_phases


def _segment_slow_motion(
    frame_labels: np.ndarray, 
    phase_start: int, 
    phase_end: int, 
    min_segment_length: int
) -> List[dict]:
    """
    Segment a stable phase into sub-phases based on slow motion labels.
    
    Args:
        frame_labels: (N, 3) array of slow motion labels for each frame
        phase_start: Start frame of the phase
        phase_end: End frame of the phase
        min_segment_length: Minimum frames per segment
        
    Returns:
        List of sub-phase dicts
    """
    n = len(frame_labels)
    if n == 0:
        return []
    
    # Create a signature for each frame: tuple of (x_label, y_label, z_label)
    signatures = [tuple(frame_labels[i]) for i in range(n)]
    
    # Segment by signature changes
    segments = []
    seg_start = 0
    current_sig = signatures[0]
    
    for i in range(1, n):
        if signatures[i] != current_sig:
            segments.append((seg_start, i, current_sig))
            seg_start = i
            current_sig = signatures[i]
    segments.append((seg_start, n, current_sig))
    
    # Filter short segments and merge with neighbors
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
            # Merge short segment with previous
            filtered_segments[-1]['end'] = phase_start + seg_end_rel
        else:
            # First segment is short, will be merged with next
            filtered_segments.append({
                'start': phase_start + seg_start_rel,
                'end': phase_start + seg_end_rel,
                'x': sig[0],
                'y': sig[1],
                'z': sig[2]
            })
    
    # If no segments, return original as stable
    if not filtered_segments:
        return [{'start': phase_start, 'end': phase_end, 'x': 0, 'y': 0, 'z': 0}]
    
    # Merge adjacent segments with same pattern
    merged = [filtered_segments[0]]
    for seg in filtered_segments[1:]:
        if (merged[-1]['x'] == seg['x'] and 
            merged[-1]['y'] == seg['y'] and 
            merged[-1]['z'] == seg['z']):
            merged[-1]['end'] = seg['end']
        else:
            merged.append(seg)
    
    # Apply combination logic: merge unless direction reversal OR stable phase
    # Stable phases (recognized as stable in two rounds) should remain separate
    combined = [merged[0]]
    for seg in merged[1:]:
        # Check if either segment is stable (all axes = 0)
        prev_is_stable = combined[-1]['x'] == 0 and combined[-1]['y'] == 0 and combined[-1]['z'] == 0
        curr_is_stable = seg['x'] == 0 and seg['y'] == 0 and seg['z'] == 0
        
        # Don't merge stable phases
        if prev_is_stable or curr_is_stable:
            combined.append(seg.copy())
            continue
        
        # Check for direction reversal
        can_merge = True
        for axis in ['x', 'y', 'z']:
            prev_label = combined[-1][axis]
            curr_label = seg[axis]
            
            # Check for direction reversal (e.g., up -> down or down -> up)
            if prev_label * curr_label < 0:  # Opposite signs (reversal)
                can_merge = False
                break
        
        if can_merge:
            # Merge: extend end frame and update labels (keep non-zero if either has it)
            combined[-1]['end'] = seg['end']
            for axis in ['x', 'y', 'z']:
                if seg[axis] != 0:
                    combined[-1][axis] = seg[axis]
        else:
            # Start new phase
            combined.append(seg.copy())
    
    return combined


def segment_by_labels(labels: np.ndarray, min_segment_length: int = 5) -> List[Tuple[int, int, int]]:
    """
    Segment an axis based on motion labels.
    
    Args:
        labels: 1D array of labels (-1, 0, 1)
        min_segment_length: Minimum frames per segment
        
    Returns:
        List of (start, end, label) tuples
    """
    n = len(labels)
    if n == 0:
        return []
    
    segments = []
    start = 0
    current_label = labels[0]
    
    for i in range(1, n):
        if labels[i] != current_label:
            # End of segment
            if i - start >= min_segment_length:
                segments.append((start, i, current_label))
            start = i
            current_label = labels[i]
    
    # Last segment
    if n - start >= min_segment_length:
        segments.append((start, n, current_label))
    
    return segments


def label_to_text(label: int) -> str:
    """Convert label to human-readable text."""
    if label == 2:
        return "⇑ (slow up)"
    elif label == -2:
        return "⇓ (slow down)"
    elif label == 1:
        return "↑ (up)"
    elif label == -1:
        return "↓ (down)"
    else:
        return "— (stable)"


def label_to_short(label: int) -> str:
    """Convert label to short symbol."""
    if label == 2:
        return "⇑"
    elif label == -2:
        return "⇓"
    elif label == 1:
        return "↑"
    elif label == -1:
        return "↓"
    else:
        return "—"


def multi_scale_motion_labels(
    positions: np.ndarray,
    thresholds: AxisThresholds,
    windows: List[int],
    sampling_interval: float = 0.04
) -> np.ndarray:
    """
    Compute motion labels using multiple window sizes to capture both fast and slow motions.
    
    Strategy: Use OR logic - if any window detects motion, label as moving.
    Small windows catch fast motions, large windows catch slow sustained motions.
    
    Args:
        positions: (N, 3) array of positions
        thresholds: Per-axis velocity thresholds
        windows: List of window sizes to use (e.g., [21, 51, 81])
        sampling_interval: Time between frames
        
    Returns:
        (N, 3) array of labels: -1=down, 0=stable, 1=up
    """
    n = len(positions)
    combined_labels = np.zeros((n, 3), dtype=int)
    axis_thresholds = [thresholds.x, thresholds.y, thresholds.z]
    
    for window_size in windows:
        # Smooth with this window
        smoothed = smooth_trajectory(positions, window_length=window_size)
        velocities = compute_per_axis_velocities(smoothed, sampling_interval)
        
        for axis in range(3):
            vel = velocities[:, axis]
            thresh = axis_thresholds[axis]
            
            # Further smooth velocity
            stable_window = max(3, window_size // 4)
            if n >= stable_window:
                kernel = np.ones(stable_window) / stable_window
                vel_smooth = np.convolve(vel, kernel, mode='same')
            else:
                vel_smooth = vel
            
            # Detect motion
            up_mask = vel_smooth > thresh
            down_mask = vel_smooth < -thresh
            
            # OR logic: if this window detects motion, update combined labels
            # But don't override existing non-zero labels with opposite direction
            for i in range(n):
                if up_mask[i] and combined_labels[i, axis] != -1:
                    combined_labels[i, axis] = 1
                elif down_mask[i] and combined_labels[i, axis] != 1:
                    combined_labels[i, axis] = -1
    
    return combined_labels


def merge_axis_segments(
    segments_x: List[Tuple[int, int, int]],
    segments_y: List[Tuple[int, int, int]],
    segments_z: List[Tuple[int, int, int]],
    n_frames: int,
    min_segment_length: int = 20
) -> List[dict]:
    """
    Merge per-axis segments into unified motion phases.
    
    A merged segment represents a consistent motion pattern across all axes.
    All segment boundaries are preserved (not just significant ones).
    
    Args:
        segments_x, segments_y, segments_z: Per-axis segments
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
    """Get the most common non-zero label, or 0 if mostly stable.
    
    Labels: -2=slow down, -1=fast down, 0=stable, 1=fast up, 2=slow up
    Returns simplified label: -1=down, 0=stable, 1=up (merging slow/fast)
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
    
    # If stable is dominant, return 0
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
    - Reversal = ↑ to ↓ OR ↓ to ↑ (sign change from +1 to -1 or vice versa)
    - Axis starting/stopping motion (0↔±1) is NOT a reversal
    
    This captures continuous motion where axes may start/stop at different times,
    but treats direction changes as new phases.
    
    Args:
        merged_segments: List of merged segments with 'start', 'end', 'x', 'y', 'z'
        merge_through_stable: If True, merge motion phases separated by stable phases
                              If False, stable phases break the continuity
        
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


def combined_phase_to_text(seg: dict) -> str:
    """Convert combined phase to human-readable description showing all directions."""
    parts = []
    labels = {'x': seg['x'], 'y': seg['y'], 'z': seg['z']}
    
    for axis, label in labels.items():
        if label == 2:
            parts.append(f"{axis}⇑")  # slow up
        elif label == -2:
            parts.append(f"{axis}⇓")  # slow down
        elif label == 1:
            parts.append(f"{axis}↑")
        elif label == -1:
            parts.append(f"{axis}↓")
    
    if not parts:
        return "stable"
    return ", ".join(parts)


def merged_segment_to_text(seg: dict) -> str:
    """Convert merged segment to human-readable description."""
    parts = []
    labels = {'x': seg['x'], 'y': seg['y'], 'z': seg['z']}
    
    for axis, label in labels.items():
        if label == 2:
            parts.append(f"{axis}⇑")  # slow up
        elif label == -2:
            parts.append(f"{axis}⇓")  # slow down
        elif label == 1:
            parts.append(f"{axis}↑")
        elif label == -1:
            parts.append(f"{axis}↓")
    
    if not parts:
        return "stable"
    return ", ".join(parts)


def detect_gripper_events(gripper: np.ndarray, action_gripper: np.ndarray) -> dict:
    """
    Detect gripper opening and closing events using V3 segmentation logic.
    Returns start/end frame pairs for opening and closing actions.
    
    Args:
        gripper: (N, 2) array of gripper positions [left, right]
        action_gripper: (N,) array of gripper commands (-1=open, 1=close, 0=none)
        
    Returns:
        dict with 'opening' and 'closing' keys, each containing list of (start, end) tuples
    """
    n = len(gripper)
    if n < 2:
        return {'opening': [], 'closing': []}
    
    # Compute gripper changes
    gripper_distance = gripper[:, 0] - gripper[:, 1]
    gripper_diffs = gripper_distance[1:] - gripper_distance[:-1]
    
    # Raw labels
    raw_labels = []
    for i in range(n - 1):
        if gripper_diffs[i] > 1e-4:  # gripper_threshold
            raw_labels.append("opening")
        elif gripper_diffs[i] < -1e-4:
            raw_labels.append("closing")
        else:
            raw_labels.append("none")
    
    # Filter using action_gripper (exclude correction periods)
    filtered_labels = ["none"] * (n - 1)
    i = 0
    while i < n - 1:
        action = action_gripper[i]
        raw_label = raw_labels[i]
        
        if action == -1:  # Opening command
            if raw_label == "closing":
                # Skip correction period
                filtered_labels[i] = "none"
                i += 1
                while i < n - 1 and action_gripper[i] == -1:
                    filtered_labels[i] = "none"
                    i += 1
                continue
            else:
                filtered_labels[i] = raw_label
        elif action == 1:  # Closing command
            if raw_label == "opening":
                # Skip correction period
                filtered_labels[i] = "none"
                i += 1
                while i < n - 1 and action_gripper[i] == 1:
                    filtered_labels[i] = "none"
                    i += 1
                continue
            else:
                filtered_labels[i] = raw_label
        else:
            filtered_labels[i] = raw_label
        
        i += 1
    
    # Fill small gaps (max_gap=5)
    max_gap = 5
    filled_labels = filtered_labels.copy()
    
    for label_type in ["opening", "closing"]:
        i = 0
        while i < len(filled_labels):
            if filled_labels[i] == label_type:
                # Find end of this segment
                j = i + 1
                while j < len(filled_labels) and filled_labels[j] == label_type:
                    j += 1
                
                # Look for small gap followed by same type
                if j + max_gap < len(filled_labels):
                    for k in range(j + 1, min(j + max_gap + 2, len(filled_labels))):
                        if filled_labels[k] == label_type:
                            # Fill the gap
                            for m in range(j, k):
                                filled_labels[m] = label_type
                            break
                        elif filled_labels[k] not in ["none", label_type]:
                            break
                
                i = j
            else:
                i += 1
    
    # Generate intervals from non-none labels
    events = {'opening': [], 'closing': []}
    i = 0
    while i < len(filled_labels):
        if filled_labels[i] in ["opening", "closing"]:
            start = i
            label = filled_labels[i]
            while i < len(filled_labels) and filled_labels[i] == label:
                i += 1
            # Convert to 0-based (start, end) tuples for visualization
            events[label].append((start, i))
        else:
            i += 1
    
    return events


def visualize_trajectory_v3(
    hdf5_path: str,
    episode: int,
    task: str = 'tool_hang',
    frame_range: tuple = None,
    custom_thresholds: AxisThresholds = None,
    custom_window: int = None,
    sampling_interval: float = 0.04,
    output_path: str = None,
    use_multi_scale: bool = False,  # Disabled by default - single window is more robust
    show_expected_segments: bool = False,
    show_slow_motion: bool = True,  # Whether to show the 'With Slow Motion Detection' row
    events_json_path: str = None,   # Path to corrected events.json (for Corrected row)
    intervals_json_path: str = None, # Path to intervals.json (for Corrected row movement bars)
    vl_imglist_path: str = None,    # Path to raw VL imglist_*.json (for VL Raw row)
    vl_fps: float = 5.0,            # FPS at which the VLM was shown frames
    trajectory_fps: float = 25.0,   # Trajectory recording FPS (for VL→traj frame mapping)
    show_correction: bool = False,  # Whether to show VL Raw + Corrected rows
):
    """
    Create V3 visualization with per-axis motion analysis.
    
    Args:
        hdf5_path: Path to HDF5 dataset
        episode: Episode number
        task: Task name for default thresholds
        frame_range: Optional (start, end) frame range
        custom_thresholds: Override default thresholds
        custom_window: Override default window size
        sampling_interval: Time between frames
        output_path: Path to save figure
        use_multi_scale: Use multiple window sizes for better motion detection
        show_expected_segments: Show expected segment boundaries for debugging
        show_slow_motion: Whether to show the 'With Slow Motion Detection' bottom row
        events_json_path: Path to corrected events.json produced by the VL correction pipeline
        intervals_json_path: Path to intervals.json with corrected movement/gripper timelines
        vl_fps: FPS at which the VLM received frames (used for VL→trajectory frame mapping)
        trajectory_fps: FPS of the recorded trajectory (used for VL→trajectory frame mapping)
        show_correction: Whether to add 'VL Raw' + 'Corrected' rows below the segmentation rows
    """
    delta = 2
    fontsize_label = 14 + delta
    fontsize_title = 16 + delta
    fontsize_legend = 12 + delta
    fontsize_text = 16 + delta
    fontsize_tick = 12 + delta
    import matplotlib as mpl
    mpl.rc('xtick', labelsize=fontsize_tick)
    mpl.rc('ytick', labelsize=fontsize_tick)

    # Load trajectory and gripper data
    positions, gripper, action_gripper = load_trajectory(hdf5_path, episode)
    n_frames = len(positions)
    
    # Apply frame range if specified
    frame_offset = 0
    if frame_range:
        start, end = frame_range
        positions = positions[start:end]
        gripper = gripper[start:end]
        action_gripper = action_gripper[start:end]
        frame_offset = start
    
    n_frames_view = len(positions)
    
    # Get thresholds and window size
    thresholds = custom_thresholds or TASK_THRESHOLDS.get(task, AxisThresholds())
    window_size = custom_window or TASK_WINDOW_SIZES.get(task, 51)
    
    # Define multi-scale windows (small for fast motion, large for slow motion)
    if use_multi_scale:
        # Use 3 scales: fast (21), medium (51), slow (81)
        multi_windows = [21, 51, 81]
        # Adjust if custom window provided
        if custom_window:
            multi_windows = [max(11, custom_window // 2), custom_window, min(101, custom_window * 2)]
        labels = multi_scale_motion_labels(positions, thresholds, multi_windows, sampling_interval)
    else:
        # Single scale
        smoothed_single = smooth_trajectory(positions, window_length=window_size)
        velocities_single = compute_per_axis_velocities(smoothed_single, sampling_interval)
        labels = compute_motion_labels(velocities_single, thresholds, stable_window=window_size // 2)
    
    # Smooth trajectory for display (using primary window)
    smoothed = smooth_trajectory(positions, window_length=window_size)
    velocities_raw = compute_per_axis_velocities(positions, sampling_interval)
    velocities_smoothed = compute_per_axis_velocities(smoothed, sampling_interval)
    
    # === STEP 1: Fast motion segmentation (no slow motion yet) ===
    # Segment each axis based on velocity thresholds only
    segments_x = segment_by_labels(labels[:, 0])
    segments_y = segment_by_labels(labels[:, 1])
    segments_z = segment_by_labels(labels[:, 2])
    
    # Merge segments across axes
    merged_segments = merge_axis_segments(segments_x, segments_y, segments_z, 
                                          n_frames_view, min_segment_length=20)
    
    # === STEP 2: Final combined phases (merge unless direction reversal) ===
    combined_phases = combine_motion_phases(merged_segments)
    
    # === STEP 3: Detect slow motion in stable phases of final combined ===
    # This will split stable phases and re-combine sub-phases within each original phase
    slow_params = SLOW_MOTION_PARAMS
    combined_phases_with_slow = detect_slow_motion_in_phases(
        smoothed, combined_phases, n_frames_view,
        displacement_window=slow_params['displacement_window'],
        displacement_threshold=slow_params['displacement_threshold'],
        min_segment_length=slow_params['min_slow_segment']
    )
    
    # Detect gripper events using V3 segmentation logic
    gripper_events = detect_gripper_events(gripper, action_gripper)

    # === Load optional VL correction data ===
    import json as _json
    import glob as _glob
    vl_raw_data  = None  # parsed raw VL imglist_*.json (VL-fps frame coords)
    vl_events    = None  # parsed corrected events.json (trajectory-frame coords)
    vl_intervals = None  # parsed intervals.json (trajectory-frame movement timeline)
    # VL frames → trajectory frames: VLM saw the video at vl_fps; traj recorded at trajectory_fps
    vl_to_traj   = trajectory_fps / vl_fps  # e.g. 20/5 = 4
    if show_correction:
        import config as _cfg
        keypose_data_dir = _cfg.KEYPOSE_DATA_ROOT
        batch = f"{(episode // 20) * 20:03d}_{(episode // 20) * 20 + 19:03d}"
        corr_ep_dir = os.path.join(keypose_data_dir, 'corrected_events_organized',
                                   task, batch, f'episode_{episode}')
        # Auto-discover raw VL imglist
        if vl_imglist_path is None:
            raw_dir = os.path.join(keypose_data_dir, 'divided_events', task, 'json_sg')
            matches = _glob.glob(os.path.join(raw_dir, f'imglist_episode_{episode}_*.json'))
            if matches:
                vl_imglist_path = matches[0]
        # Auto-discover corrected events.json
        if events_json_path is None:
            candidate = os.path.join(corr_ep_dir, 'events.json')
            if os.path.exists(candidate):
                events_json_path = candidate
        # Auto-discover intervals.json
        if intervals_json_path is None:
            candidate = os.path.join(corr_ep_dir, 'intervals.json')
            if os.path.exists(candidate):
                intervals_json_path = candidate

        # Load raw VL JSON
        if vl_imglist_path and os.path.exists(vl_imglist_path):
            with open(vl_imglist_path) as f:
                _raw = _json.load(f)
            vl_raw_data = _raw
            # Read vl_fps from args if present
            _args = _raw.get('args', {})
            if 'target_fps' in _args:
                vl_fps = float(_args['target_fps'])
                vl_to_traj = trajectory_fps / vl_fps
        # Load corrected events.json
        if events_json_path and os.path.exists(events_json_path):
            with open(events_json_path) as f:
                vl_events = _json.load(f)
            ci = vl_events.get('_correction_info', {})
            if 'trajectory_fps' in ci:
                trajectory_fps = ci['trajectory_fps']
                vl_to_traj = trajectory_fps / vl_fps
        # Load intervals.json
        if intervals_json_path and os.path.exists(intervals_json_path):
            with open(intervals_json_path) as f:
                vl_intervals = _json.load(f)

    # === Figure layout ===
    # Create figure with rectangular layout: 2 rows (pos/vel) x 3 cols (x/y/z), then segmentation rows
    n_seg_rows = 4 if show_slow_motion else 3  # per-axis, raw, combined, [slow]
    n_corr_rows = 2 if (show_correction and (vl_raw_data or vl_events or vl_intervals)) else 0
    n_total_rows = 2 + n_seg_rows + n_corr_rows
    seg_height = 0.6
    height_ratios = [1.0, 1.0] + [0.7] + [seg_height] * (n_seg_rows - 1) + [seg_height] * n_corr_rows
    fig_height = 8 + n_seg_rows * 1 + n_corr_rows * 1.0
    fig = plt.figure(figsize=(28, fig_height))
    gs = GridSpec(n_total_rows, 3, figure=fig, height_ratios=height_ratios,
                  hspace=0.9, wspace=0.2,
                  top=0.92, bottom=0.05)
    
    frames = np.arange(len(positions)) + frame_offset
    axis_names = ['X', 'Y', 'Z']
    axis_colors  = ['#E74C3C', '#27AE60', '#3498DB']  # X=red, Y=green, Z=blue
    # Gripper: dark purple (close) and orange (open) — unique across all rows
    color_close    = '#6C3483'   # Dark purple  - gripper closing
    color_open     = '#E67E22'   # Orange       - gripper opening
    # Motion background shading (pos/vel rows): teal=up, amber=down
    color_bg_up    = '#A8DADC'   # Teal   - upward motion
    color_bg_down  = '#F4A261'   # Amber  - downward motion
    # Raw merged heat scale: blue gradient by number of active axes
    color_raw_0ax  = '#BDC3C7'   # Light gray  - stable (0 axes)
    color_raw_1ax  = '#74B9FF'   # Light blue  - 1 axis moving
    color_raw_2ax  = '#0984E3'   # Medium blue - 2 axes moving
    color_raw_3ax  = '#2D3436'   # Dark        - all 3 axes moving
    # Combined/Slow phase cycle: 10 distinct colors, no overlap with axis or gripper colors
    combined_colors = ['#8E44AD','#16A085','#D35400','#2471A3','#CB4335',
                       '#117A65','#A04000','#1A5276','#922B21','#0E6655']
    
    # 1-3. Per-axis position plots (top row)
    for i, (axis_name, color) in enumerate(zip(axis_names, axis_colors)):
        # Position plot (row 0, column i)
        ax_pos = fig.add_subplot(gs[0, i])
        ax_pos.plot(frames, positions[:, i], color='0.55', alpha=0.9,
                    linewidth=1.4, zorder=2)
        ax_pos.plot(frames, smoothed[:, i], color=color, linewidth=2.4, zorder=3)
        ax_pos.set_ylabel(f'{axis_name} Position (m)', fontsize=fontsize_label)
        ax_pos.grid(True, alpha=0.3)
        # if i == 0: # Only show legend on first position plot to avoid clutter
        #     ax_pos.legend(loc='upper right', fontsize=fontsize_legend)  # Original / Smoothed
        # else:
        #     ax_pos.get_legend().remove() if ax_pos.get_legend() else None
        
        # Add motion label background
        segments = [segments_x, segments_y, segments_z][i]
        for start, end, label in segments:
            if label == 1:      # fast up
                facecolor = color_bg_up
                alpha = 0.30
            elif label == -1:   # fast down
                facecolor = color_bg_down
                alpha = 0.30
            elif label == 2:    # slow up
                facecolor = color_bg_up
                alpha = 0.15
            elif label == -2:   # slow down
                facecolor = color_bg_down
                alpha = 0.15
            else:
                continue
            ax_pos.axvspan(start + frame_offset, end + frame_offset,
                          facecolor=facecolor, alpha=alpha)
        
        # Add gripper event markers (both start and end)
        for start, end in gripper_events['closing']:
            ax_pos.axvline(start + frame_offset, color=color_close, linestyle='--', 
                          linewidth=1.5, alpha=0.8, zorder=10)
            ax_pos.axvline(end + frame_offset, color=color_close, linestyle=':', 
                          linewidth=1.5, alpha=0.5, zorder=10)
        
        for start, end in gripper_events['opening']:
            ax_pos.axvline(start + frame_offset, color=color_open, linestyle='--', 
                          linewidth=1.5, alpha=0.8, zorder=10)
            ax_pos.axvline(end + frame_offset, color=color_open, linestyle=':', 
                          linewidth=1.5, alpha=0.5, zorder=10)
        
        if i == 0:
            # Add legend for gripper markers on first position plot only
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color='lightgray', linewidth=1.5, alpha=0.7, label='Raw'),
                Line2D([0], [0], color=axis_colors[0], linewidth=2, label='Smoothed'),
                Line2D([0], [0], color=color_close, linestyle='--', linewidth=1.5, label='Close (start)'),
                Line2D([0], [0], color=color_close, linestyle=':', linewidth=1.5, alpha=0.5, label='Close (end)'),
                Line2D([0], [0], color=color_open,  linestyle='--', linewidth=1.5, label='Open (start)'),
                Line2D([0], [0], color=color_open,  linestyle=':', linewidth=1.5, alpha=0.5, label='Open (end)')
            ]
            ax_pos.legend(handles=legend_elements, loc='upper left', fontsize=fontsize_legend, ncol=3)
        if i == 1:
            ax_pos.set_title('Per-Axis Position and Motion Labels', fontsize=fontsize_title, fontweight='bold')
        ax_pos.set_xticklabels([])  # No x labels on top row
        
        
        # Velocity plot (row 1, column i)
        ax_vel = fig.add_subplot(gs[1, i])
        ax_vel.plot(frames, velocities_raw[:, i] * 1000, color='0.55',
                    alpha=0.85, linewidth=1.3, zorder=2)
        ax_vel.plot(frames, velocities_smoothed[:, i] * 1000, color=color,
                    linewidth=2.4, zorder=3)
        
        # Threshold lines
        thresh = [thresholds.x, thresholds.y, thresholds.z][i]
        ax_vel.axhline(thresh * 1000, color='gray', linestyle='--', 
                      linewidth=1, alpha=0.7, label=f'±{thresh*1000:.1f} mm/s')
        ax_vel.axhline(-thresh * 1000, color='gray', linestyle='--', 
                      linewidth=1, alpha=0.7)
        ax_vel.axhline(0, color='black', linestyle='-', linewidth=0.5)
        
        ax_vel.set_ylabel(f'{axis_name} Velocity (mm/s)', fontsize=fontsize_label)
        ax_vel.grid(True, alpha=0.3)
        # ax_vel.legend(loc='upper right', fontsize=fontsize_legend)
        # if i == 0:
        #     ax_vel.legend(loc='upper right', fontsize=fontsize_legend)
        # else:
        #     ax_vel.get_legend().remove() if ax_vel.get_legend() else None
        
        # Add same motion label background
        for start, end, label in segments:
            if label == 1:      # fast up
                facecolor = color_bg_up
                alpha = 0.30
            elif label == -1:   # fast down
                facecolor = color_bg_down
                alpha = 0.30
            elif label == 2:    # slow up
                facecolor = color_bg_up
                alpha = 0.15
            elif label == -2:   # slow down
                facecolor = color_bg_down
                alpha = 0.15
            else:
                continue
            ax_vel.axvspan(start + frame_offset, end + frame_offset,
                          facecolor=facecolor, alpha=alpha)
        
        # Add gripper event markers to velocity plots (both start and end)
        for start, end in gripper_events['closing']:
            ax_vel.axvline(start + frame_offset, color=color_close, linestyle='--', 
                          linewidth=1.5, alpha=0.8, zorder=10)
            ax_vel.axvline(end + frame_offset, color=color_close, linestyle=':', 
                          linewidth=1.5, alpha=0.5, zorder=10)
        
        for start, end in gripper_events['opening']:
            ax_vel.axvline(start + frame_offset, color=color_open, linestyle='--', 
                          linewidth=1.5, alpha=0.8, zorder=10)
            ax_vel.axvline(end + frame_offset, color=color_open, linestyle=':', 
                          linewidth=1.5, alpha=0.5, zorder=10)
        
        if i == 1:
            ax_vel.set_title('Per-Axis Velocity and Motion Labels', fontsize=fontsize_title, fontweight='bold')
        ax_vel.set_xlabel('Frame', fontsize=fontsize_label)
    
    # 4. Per-Axis Segments (from velocity thresholds) - row 2, full width
    ax_per_axis = fig.add_subplot(gs[2, :])  # always row 2
    ax_per_axis.set_xlim(frames[0], frames[-1])
    ax_per_axis.set_ylim(-0.5, 3.5)
    
    all_segments = [segments_z, segments_y, segments_x]  # Bottom to top: Z, Y, X
    axis_labels = ['Z', 'Y', 'X']
    axis_segment_colors = ['#E74C3C', '#27AE60', '#3498DB']  # Red, Green, Blue
    
    for axis_idx, (seg_list, ax_label, color) in enumerate(zip(all_segments, axis_labels, axis_segment_colors)):
        y_pos = axis_idx
        
        for start, end, label in seg_list:
            if label == 1:
                bar_color = color  # Full color for up
                alpha = 0.85
                hatch = None
                label_char = '↑'
            elif label == -1:
                bar_color = color  # Same color, hatched for down
                alpha = 0.55
                hatch = '\\\\'
                label_char = '↓'
            else:
                bar_color = '#BDC3C7'  # Light gray for stable
                alpha = 0.35
                hatch = None
                label_char = '—'
            
            ax_per_axis.barh(y_pos, end - start, left=start + frame_offset,
                            height=0.7, color=bar_color, alpha=alpha, hatch=hatch,
                            edgecolor='white', linewidth=1)
            
            # Label for longer segments
            if end - start > 20:
                ax_per_axis.text((start + end) / 2 + frame_offset, y_pos, label_char,
                                ha='center', va='center', fontsize=fontsize_label, fontweight='bold',
                                color='white' if label != 0 else 'gray')
    
    ax_per_axis.set_yticks([0, 1, 2])
    ax_per_axis.set_yticklabels(['Z', 'Y', 'X'], fontsize=fontsize_label)
    ax_per_axis.set_xlabel('Frame', fontsize=fontsize_label)
    ax_per_axis.set_title(f'Per-Axis Motion Segments (↑=up, ↓=down, —=stable)', fontsize=fontsize_title)
    ax_per_axis.grid(True, alpha=0.3, axis='x')
    
    per_axis_legend = [
        Patch(facecolor='#E74C3C', alpha=0.85, label='X Up (↑)'),
        Patch(facecolor='#E74C3C', alpha=0.55, hatch='\\\\', label='X Down (↓)'),
        Patch(facecolor='#27AE60', alpha=0.85, label='Y Up (↑)'),
        Patch(facecolor='#27AE60', alpha=0.55, hatch='\\\\', label='Y Down (↓)'),
        Patch(facecolor='#3498DB', alpha=0.85, label='Z Up (↑)'),
        Patch(facecolor='#3498DB', alpha=0.55, hatch='\\\\', label='Z Down (↓)'),
    ]
    # ax_per_axis.legend(handles=per_axis_legend, loc='upper right', fontsize=fontsize_legend, ncol=6)
    
    # 6. Raw Merged Phases - row 3, full width
    ax_raw = fig.add_subplot(gs[3, :])
    ax_raw.set_xlim(frames[0], frames[-1])
    ax_raw.set_ylim(-0.5, 1.5)
    
    # Color scheme for raw merged (activity level): blue gradient by #active axes
    for i, seg in enumerate(merged_segments):
        start, end = seg['start'], seg['end']
        x_label, y_label, z_label = seg['x'], seg['y'], seg['z']
        
        moving_count = sum(1 for l in [x_label, y_label, z_label] if l != 0)
        
        if moving_count == 0:
            color = color_raw_0ax  # Light gray - stable
            alpha = 0.7
        elif moving_count == 1:
            color = color_raw_1ax  # Light blue - 1 axis
            alpha = 0.8
        elif moving_count == 2:
            color = color_raw_2ax  # Medium blue - 2 axes
            alpha = 0.85
        else:
            color = color_raw_3ax  # Dark - all 3 axes
            alpha = 0.9
        
        ax_raw.barh(0.5, end - start, left=start + frame_offset, 
                    height=0.8, color=color, alpha=alpha, edgecolor='white', linewidth=1)
        
        segment_text = merged_segment_to_text(seg)
        if end - start > 8:
            ax_raw.text((start + end) / 2 + frame_offset, 0.5, segment_text,
                       ha='center', va='center', fontsize=fontsize_text, fontweight='bold',
                       color='white' if moving_count >= 2 else 'black')
        
        ax_raw.text((start + end) / 2 + frame_offset, -0.5, f'{i+1}',
                   ha='center', va='bottom', fontsize=fontsize_label, color='gray')
    
    ax_raw.set_yticks([0.5])
    ax_raw.set_yticklabels(['Raw'], rotation='vertical', va='center', fontsize=fontsize_label)
    ax_raw.set_title(f'Raw Merged Phases (boundaries from all axis changes)', fontsize=fontsize_title)
    ax_raw.grid(True, alpha=0.3, axis='x')
    
    # Add gripper markers to raw merged phases
    for start, end in gripper_events['closing']:
        ax_raw.axvline(start + frame_offset, color=color_close, linestyle='--', 
                      linewidth=2, alpha=0.8, zorder=100)
        ax_raw.axvline(end + frame_offset, color=color_close, linestyle=':', 
                      linewidth=2, alpha=0.6, zorder=100)
        ax_raw.axvspan(start + frame_offset, end + frame_offset, 
                      facecolor=color_close, alpha=0.1, zorder=0)
    
    for start, end in gripper_events['opening']:
        ax_raw.axvline(start + frame_offset, color=color_open, linestyle='--', 
                      linewidth=2, alpha=0.8, zorder=100)
        ax_raw.axvline(end + frame_offset, color=color_open, linestyle=':', 
                      linewidth=2, alpha=0.6, zorder=100)
        ax_raw.axvspan(start + frame_offset, end + frame_offset, 
                      facecolor=color_open, alpha=0.1, zorder=0)
    
    raw_legend = [
        Patch(facecolor='#BDC3C7', label='Stable (0)'),
        Patch(facecolor='#74B9FF', label='1 axis'),
        Patch(facecolor='#0984E3', label='2 axes'),
        Patch(facecolor='#2D3436', label='3 axes'),
    ]
    # ax_raw.legend(handles=raw_legend, loc='upper right', fontsize=fontsize_legend, ncol=4)
    
    # 7. Final Combined Phases (before slow motion) - row 4, full width
    ax_combined = fig.add_subplot(gs[4, :])
    ax_combined.set_xlim(frames[0], frames[-1])
    ax_combined.set_ylim(-0.5, 1.5)
    
    for i, seg in enumerate(combined_phases):
        start, end = seg['start'], seg['end']
        x_label, y_label, z_label = seg['x'], seg['y'], seg['z']
        
        moving_count = sum(1 for l in [x_label, y_label, z_label] if l != 0)
        
        if moving_count == 0:
            color = '#95A5A6'  # Gray for stable
        else:
            color = combined_colors[i % len(combined_colors)]
        
        ax_combined.barh(0.5, end - start, left=start + frame_offset, 
                        height=0.8, color=color, alpha=0.85, edgecolor='white', linewidth=2)
        
        segment_text = combined_phase_to_text(seg)
        if end - start > 8:
            ax_combined.text((start + end) / 2 + frame_offset, 0.5, segment_text,
                            ha='center', va='center', fontsize=fontsize_text, fontweight='bold',
                            color='white' if moving_count > 0 else 'black')
        
        ax_combined.text((start + end) / 2 + frame_offset, -0.5, f'P{i+1}',
                        ha='center', va='bottom', fontsize=fontsize_label, fontweight='bold', color='#2C3E50')
    
    ax_combined.set_yticks([0.5])
    ax_combined.set_yticklabels(['Combined'], rotation='vertical', va='center', fontsize=fontsize_label)
    if not show_slow_motion:
        ax_combined.set_xlabel('Frame', fontsize=fontsize_label)
    ax_combined.set_title(f'Final Combined Phases (merged unless direction reversal)', 
                         fontsize=fontsize_title, fontweight='bold')
    ax_combined.grid(True, alpha=0.3, axis='x')
    
    # Add gripper markers to combined phases
    for start, end in gripper_events['closing']:
        ax_combined.axvline(start + frame_offset, color=color_close, linestyle='--', 
                          linewidth=2, alpha=0.8, zorder=100)
        ax_combined.axvline(end + frame_offset, color=color_close, linestyle=':', 
                          linewidth=2, alpha=0.6, zorder=100)
        ax_combined.axvspan(start + frame_offset, end + frame_offset, 
                          facecolor=color_close, alpha=0.1, zorder=0)
    
    for start, end in gripper_events['opening']:
        ax_combined.axvline(start + frame_offset, color=color_open, linestyle='--', 
                          linewidth=2, alpha=0.8, zorder=100)
        ax_combined.axvline(end + frame_offset, color=color_open, linestyle=':', 
                          linewidth=2, alpha=0.6, zorder=100)
        ax_combined.axvspan(start + frame_offset, end + frame_offset, 
                          facecolor=color_open, alpha=0.1, zorder=0)
    
    # 8. Final with Slow Motion (optional)
    if show_slow_motion:
        ax_slow = fig.add_subplot(gs[5, :])
        ax_slow.set_xlim(frames[0], frames[-1])
        ax_slow.set_ylim(-0.5, 1.5)
        
        for i, seg in enumerate(combined_phases_with_slow):
            start, end = seg['start'], seg['end']
            x_label, y_label, z_label = seg['x'], seg['y'], seg['z']
            
            moving_count = sum(1 for l in [x_label, y_label, z_label] if l != 0)
            has_slow = any(abs(l) == 2 for l in [x_label, y_label, z_label])
            
            if moving_count == 0:
                color = '#95A5A6'  # Gray for stable
            elif has_slow and moving_count == sum(1 for l in [x_label, y_label, z_label] if abs(l) == 2):
                # All motion is slow
                color = '#BDC3C7'  # Light gray for slow-only
            else:
                color = combined_colors[i % len(combined_colors)]
            
            ax_slow.barh(0.5, end - start, left=start + frame_offset, 
                        height=0.8, color=color, alpha=0.85, edgecolor='white', linewidth=2)
            
            segment_text = combined_phase_to_text(seg)
            if end - start > 8:
                ax_slow.text((start + end) / 2 + frame_offset, 0.5, segment_text,
                            ha='center', va='center', fontsize=fontsize_text, fontweight='bold',
                            color='white' if moving_count > 0 and not (has_slow and moving_count == sum(1 for l in [x_label, y_label, z_label] if abs(l) == 2)) else 'black')
            
            ax_slow.text((start + end) / 2 + frame_offset, -0.5, f'P{i+1}',
                        ha='center', va='bottom', fontsize=fontsize_label, fontweight='bold', color='#2C3E50')
        
        ax_slow.set_yticks([0.5])
        ax_slow.set_yticklabels(['+ Slow'], rotation='vertical', va='center', fontsize=fontsize_label)
        ax_slow.set_xlabel('Frame', fontsize=fontsize_label)
        ax_slow.set_title(f'With Slow Motion Detection ({len(combined_phases_with_slow)} phases, ⇑⇓=slow)', 
                         fontsize=fontsize_title, fontweight='bold')
        ax_slow.grid(True, alpha=0.3, axis='x')
        
        # Add gripper markers to slow motion phases
        for start, end in gripper_events['closing']:
            ax_slow.axvline(start + frame_offset, color=color_close, linestyle='--', 
                           linewidth=2, alpha=0.8, zorder=100)
            ax_slow.axvline(end + frame_offset, color=color_close, linestyle=':', 
                           linewidth=2, alpha=0.6, zorder=100)
            ax_slow.axvspan(start + frame_offset, end + frame_offset, 
                           facecolor=color_close, alpha=0.1, zorder=0)
        
        for start, end in gripper_events['opening']:
            ax_slow.axvline(start + frame_offset, color=color_open, linestyle='--', 
                           linewidth=2, alpha=0.8, zorder=100)
            ax_slow.axvline(end + frame_offset, color=color_open, linestyle=':', 
                           linewidth=2, alpha=0.6, zorder=100)
            ax_slow.axvspan(start + frame_offset, end + frame_offset, 
                           facecolor=color_open, alpha=0.1, zorder=0)
        
        slow_legend = [
            Patch(facecolor='#BDC3C7', label='Slow (⇑⇓)'),
            Patch(facecolor='#95A5A6', label='Stable'),
        ]
        ax_slow.legend(handles=slow_legend, loc='upper right', fontsize=fontsize_legend, ncol=2)

    # =========================================================================
    # 9 & 10. VL Correction rows (optional)
    #   Row n-2: "VL Raw"    – subtask spans from the VLM divided_events output
    #                          (raw VL-fps frames × vl_to_traj → trajectory frames)
    #   Row n-1: "Corrected" – corrected movement timeline (intervals.json) +
    #                          corrected event keypoints (events.json)
    # =========================================================================
    if n_corr_rows == 2:
        # ── colour / style constants ──────────────────────────────────────────
        subtask_palette = ['#1D6FA4', '#B7950B', '#1D8348', '#7D3C98', '#A04000']
        # Phase background alphas (light fill only, boundaries shown as lines)
        phase_alpha = {'targeting': 0.12, 'interaction': 0.30, 'result': 0.18}
        phase_hatch = {'targeting': '///', 'interaction': None, 'result': '...'}

        event_colors = {
            'grasp':       '#6C3483',
            'detach':      '#2874A6',
            'release':     '#E67E22',
            'interact':    '#1D8348',
            'land':        '#CB4335',
            'attach_hang': '#1D8348',
            'attach_drop': '#CB4335',
            'contact':     '#1D8348',
        }
        event_markers = {
            'grasp':       '▼',
            'detach':      '◆',
            'release':     '▲',
            'interact':    '★',
            'land':        '●',
            'attach_hang': '★',
            'attach_drop': '●',
            'contact':     '★',
        }
        _canonical_name = {
            'attach_hang': 'interact',
            'attach_drop': 'land',
            'contact':     'interact',
        }

        def _setup_corr_ax(ax, title, show_xlabel=False):
            """Configure a correction-row axes to match Final Combined Phases style exactly."""
            ax.set_xlim(frames[0], frames[-1])
            ax.set_ylim(-0.5, 1.5)
            ax.set_yticks([0.5])
            ax.grid(True, alpha=0.3, axis='x')
            ax.set_title(title, fontsize=fontsize_title, fontweight='bold')
            if show_xlabel:
                ax.set_xlabel('Frame', fontsize=fontsize_label)

        def _draw_phase_span(ax, pstart, pend, color, phase_name):
            """Background fill for a phase span."""
            alpha  = phase_alpha[phase_name]
            hatch  = phase_hatch[phase_name]
            # Targeting: draw a solid fill first then a darker hatch overlay so it's visible
            if phase_name == 'targeting':
                ax.axvspan(pstart, pend, facecolor=color, alpha=0.25,
                           hatch=None, edgecolor='none', zorder=0)
                ax.axvspan(pstart, pend, facecolor='none', alpha=1.0,
                           hatch='////', edgecolor=color, linewidth=0.8, zorder=1)
            else:
                ax.axvspan(pstart, pend,
                           facecolor=color, alpha=alpha,
                           hatch=hatch, edgecolor='none', zorder=0)

        def _draw_boundary_line(ax, frame, color):
            """Solid vertical boundary line at phase/subtask start."""
            ax.axvline(frame, color=color, linestyle='-', linewidth=2.0,
                       alpha=0.85, zorder=80)

        def _draw_segment_label(ax, pstart, pend, label_str, color):
            """Label centred inside span at y=0.5 (same level as phase text in Combined row)."""
            mid = (pstart + pend) / 2
            if pend - pstart > 6:
                ax.text(mid, 0.5, label_str,
                        ha='center', va='center',
                        fontsize=fontsize_text, fontweight='bold',
                        color='white', zorder=85)

        def _draw_event_label(ax, ev_frame, prim, label_idx):
            """Draw event: solid colored line + marker symbol + black text at bottom (P1 level)."""
            ec = event_colors.get(prim, '#555555')
            em = event_markers.get(prim, '|')
            display_name = _canonical_name.get(prim, prim)
            # Solid event line (full bar height)
            ax.axvline(ev_frame, color=ec, linestyle='-', linewidth=2.0,
                       alpha=0.95, zorder=90)
            # Symbol + label at bottom, same y as "P1" text in Combined row
            ax.text(ev_frame - 0.5, -0.48,
                    f'{em} {display_name}',
                    ha='left', va='bottom',
                    fontsize=fontsize_label, color='black',
                    fontweight='normal', zorder=92)

        def _add_gripper_lines(ax):
            for gs_start, gs_end in gripper_events['closing']:
                ax.axvline(gs_start + frame_offset, color=color_close, linestyle='--',
                           linewidth=1.5, alpha=0.7, zorder=100)
                ax.axvline(gs_end   + frame_offset, color=color_close, linestyle=':',
                           linewidth=1.5, alpha=0.5, zorder=100)
                ax.axvspan(gs_start + frame_offset, gs_end + frame_offset,
                           facecolor=color_close, alpha=0.08, zorder=0)
            for go_start, go_end in gripper_events['opening']:
                ax.axvline(go_start + frame_offset, color=color_open, linestyle='--',
                           linewidth=1.5, alpha=0.7, zorder=100)
                ax.axvline(go_end   + frame_offset, color=color_open, linestyle=':',
                           linewidth=1.5, alpha=0.5, zorder=100)
                ax.axvspan(go_start + frame_offset, go_end + frame_offset,
                           facecolor=color_open, alpha=0.08, zorder=0)

        vl_row_idx   = 2 + n_seg_rows
        corr_row_idx = 2 + n_seg_rows + 1

        # ── ROW n-2: VL Model Output ──────────────────────────────────────────
        ax_vl = fig.add_subplot(gs[vl_row_idx, :])
        ax_vl.set_yticklabels(['VL Raw'], rotation='vertical', va='center',
                               fontsize=fontsize_label)
        _setup_corr_ax(ax_vl,
                       'VL Model Output  (subtask phases from raw VLM output)')

        seen_phase_types = set()
        if vl_raw_data:
            subtask_keys = sorted(k for k in vl_raw_data if k.startswith('subtask'))
            ev_label_idx = 0
            for si, sk in enumerate(subtask_keys):
                st   = vl_raw_data[sk]
                sc   = subtask_palette[si % len(subtask_palette)]

                # Collect all phases and draw background fills + boundary lines
                phase_spans = {}
                for phase_name in ('targeting', 'interaction', 'result'):
                    phase = st.get(phase_name)
                    if phase is None:
                        continue
                    pstart = int(phase.get('start_time', 0) * vl_to_traj) + frame_offset
                    pend   = int(phase.get('end_time',   0) * vl_to_traj) + frame_offset
                    if pend <= pstart:
                        pend = pstart + 1
                    phase_spans[phase_name] = (pstart, pend)
                    _draw_phase_span(ax_vl, pstart, pend, sc, phase_name)
                    seen_phase_types.add(phase_name)

                # Draw one solid boundary line at the subtask start
                first_phase = st.get('targeting') or st.get('interaction')
                if first_phase:
                    sub_start = int(first_phase.get('start_time', 0) * vl_to_traj) + frame_offset
                    _draw_boundary_line(ax_vl, sub_start, sc)

                # Draw solid boundary line at interaction start (tar→int boundary)
                if 'interaction' in st:
                    int_start = int(st['interaction'].get('start_time', 0) * vl_to_traj) + frame_offset
                    _draw_boundary_line(ax_vl, int_start, sc)

                # Segment labels at bar centre
                for phase_name, (pstart, pend) in phase_spans.items():
                    _draw_segment_label(ax_vl, pstart, pend,
                                        f'S{si+1}:{phase_name[:3]}', sc)

                # Event markers from connections
                for phase_name in ('targeting', 'interaction', 'result'):
                    phase = st.get(phase_name)
                    if phase is None:
                        continue
                    for conn in phase.get('connections', []):
                        prim     = conn[1]
                        vl_frame = conn[2]
                        ev_frame = int(vl_frame * vl_to_traj) + frame_offset
                        _draw_event_label(ax_vl, ev_frame, prim, ev_label_idx)
                        ev_label_idx += 1

        # Legend: phase fill patterns only
        from matplotlib.lines import Line2D as _L2D
        legend_els = []
        for pt, lbl in [('targeting', 'Targeting'), ('interaction', 'Interaction'),
                         ('result', 'Result')]:
            if pt in seen_phase_types:
                if pt == 'targeting':
                    legend_els.append(
                        Patch(facecolor='#555555', alpha=0.5,
                              hatch='////', edgecolor='#333333',
                              linewidth=0.8, label=lbl))
                else:
                    legend_els.append(
                        Patch(facecolor='#555555', alpha=phase_alpha[pt],
                              hatch=phase_hatch[pt], edgecolor='#333333',
                              linewidth=0.6, label=lbl))
        if legend_els:
            ax_vl.legend(handles=legend_els, loc='upper right',
                         fontsize=fontsize_legend, ncol=len(legend_els))
        ax_vl.set_yticks([0.5])
        ax_vl.set_yticklabels(['VL Raw'], rotation='vertical', va='center',
                               fontsize=fontsize_label)

        # ── ROW n-1: Corrected Timeline ───────────────────────────────────────
        ax_corr = fig.add_subplot(gs[corr_row_idx, :])
        _setup_corr_ax(ax_corr,
                       'Corrected Timeline',
                       show_xlabel=True)

        if vl_events:
            subtask_keys = sorted(k for k in vl_events if k.startswith('subtask'))
            ev_label_idx = 0
            for si, sk in enumerate(subtask_keys):
                st = vl_events[sk]
                sc = subtask_palette[si % len(subtask_palette)]

                # Build corrected phase spans.
                # Targeting end = frame of first connection in interaction (not end_frame).
                # Interaction covers from that first-event frame to the last connection frame.
                inter = st.get('interaction', {})
                result_ph = st.get('result')
                connections = inter.get('connections', [])

                # Start of subtask = start_frame of targeting (or interaction if no targeting)
                tgt = st.get('targeting', {})
                sub_start = tgt.get('start_frame', inter.get('start_frame', 0)) + frame_offset

                if connections:
                    # Targeting: subtask_start → first connection frame
                    first_ev_frame = connections[0][2] + frame_offset
                    # Interaction: first → last connection frame (or interaction end_frame)
                    last_ev_frame  = connections[-1][2] + frame_offset
                    inter_end      = inter.get('end_frame', connections[-1][2]) + frame_offset
                    # Use the larger of last_ev_frame and inter end_frame
                    int_end = max(last_ev_frame, inter_end)

                    tgt_span  = (sub_start,    first_ev_frame)
                    int_span  = (first_ev_frame, int_end)
                else:
                    # No events: targeting spans whole subtask
                    inter_end = inter.get('end_frame', tgt.get('end_frame', sub_start)) + frame_offset
                    tgt_span  = (sub_start, inter_end)
                    int_span  = None

                # Result phase
                res_span = None
                if result_ph:
                    res_conns = result_ph.get('connections', [])
                    r_start = int_span[1] if int_span else sub_start
                    r_end   = result_ph.get('end_frame', r_start) + frame_offset
                    if res_conns:
                        r_end = max(r_end, res_conns[-1][2] + frame_offset)
                    res_span = (r_start, r_end)

                # Draw background fills
                _draw_phase_span(ax_corr, *tgt_span, sc, 'targeting')
                if int_span:
                    _draw_phase_span(ax_corr, *int_span, sc, 'interaction')
                if res_span:
                    _draw_phase_span(ax_corr, *res_span, sc, 'result')

                # Solid boundary line at subtask start and interaction start
                _draw_boundary_line(ax_corr, sub_start, sc)
                if int_span:
                    _draw_boundary_line(ax_corr, int_span[0], sc)

                # Segment labels
                _draw_segment_label(ax_corr, *tgt_span,  f'S{si+1}:tar', sc)
                if int_span:
                    _draw_segment_label(ax_corr, *int_span, f'S{si+1}:int', sc)

                # Event markers
                for phase_name in ('targeting', 'interaction', 'result'):
                    phase = st.get(phase_name)
                    if phase is None:
                        continue
                    for conn in phase.get('connections', []):
                        prim     = conn[1]
                        ev_frame = conn[2] + frame_offset
                        _draw_event_label(ax_corr, ev_frame, prim, ev_label_idx)
                        ev_label_idx += 1

        ax_corr.set_yticks([0.5])
        ax_corr.set_yticklabels(['Corrected'], rotation='vertical', va='center',
                                fontsize=fontsize_label)

    window_info = f"Multi-scale [{','.join(map(str, multi_windows))}]" if use_multi_scale else f"Window={window_size}"
    # plt.suptitle(f'Trajectory Analysis V3 - {task.upper()} Episode {episode}\n'
    #             f'Thresholds: X={thresholds.x*1000:.1f}, Y={thresholds.y*1000:.1f}, Z={thresholds.z*1000:.1f} mm/s | '
    #             f'{window_info}', fontsize=fontsize_title, y=0.995)
    
    if output_path:
        plt.savefig(output_path, dpi=220, bbox_inches='tight')
        print(f"Saved visualization to: {output_path}")
    else:
        plt.show()
    
    # Print segment summary
    print("\n" + "=" * 80)
    print(f"TRAJECTORY ANALYSIS V3 - {task.upper()} Episode {episode}")
    print("=" * 80)
    print(f"\nThresholds: X={thresholds.x*1000:.2f}, Y={thresholds.y*1000:.2f}, Z={thresholds.z*1000:.2f} mm/s")
    print(f"Window: {window_info}")
    print(f"Frame range: {frame_offset} - {frame_offset + len(positions)}")
    
    print("\n--- Per-Axis Segments (fast motion only) ---")
    for axis_name, segments in zip(axis_names, [segments_x, segments_y, segments_z]):
        print(f"\n{axis_name} axis:")
        for start, end, label in segments:
            label_text = label_to_text(label)
            print(f"  [{start + frame_offset:>3}, {end + frame_offset:>3}): {label_text}")
    
    print("\n--- Raw Merged Phases ---")
    for i, seg in enumerate(merged_segments):
        start, end = seg['start'] + frame_offset, seg['end'] + frame_offset
        description = merged_segment_to_text(seg)
        print(f"  {i+1}. [{start:>3}, {end:>3}): {description}")
    
    print("\n--- Final Combined Phases (merged unless direction reversal) ---")
    for i, seg in enumerate(combined_phases):
        start, end = seg['start'] + frame_offset, seg['end'] + frame_offset
        description = combined_phase_to_text(seg)
        print(f"  P{i+1}. [{start:>3}, {end:>3}): {description}")
    
    print("\n--- Final Phases with Slow Motion Detection ---")
    for i, seg in enumerate(combined_phases_with_slow):
        start, end = seg['start'] + frame_offset, seg['end'] + frame_offset
        description = combined_phase_to_text(seg)
        print(f"  P{i+1}. [{start:>3}, {end:>3}): {description}")
    
    print("\n" + "=" * 80)
    
    return {
        'segments_x': segments_x,
        'segments_y': segments_y,
        'segments_z': segments_z,
        'merged_segments': merged_segments,
        'combined_phases': combined_phases,
        'combined_phases_with_slow': combined_phases_with_slow,
        'labels': labels,
        'smoothed': smoothed,
        'velocities': velocities_smoothed,
        'thresholds': thresholds,
        'window_size': window_size
    }


def main():
    parser = argparse.ArgumentParser(description="V3 Trajectory Visualization with Per-Axis Analysis")
    parser.add_argument('--hdf5_path', type=str, required=True,
                        help='Path to HDF5 dataset')
    parser.add_argument('--episode', type=int, required=True,
                        help='Episode number to visualize')
    parser.add_argument('--task', type=str, default='tool_hang',
                        choices=['tool_hang', 'square', 'can'],
                        help='Task name for default thresholds')
    parser.add_argument('--frame_start', type=int, default=None,
                        help='Start frame (optional)')
    parser.add_argument('--frame_end', type=int, default=None,
                        help='End frame (optional)')
    parser.add_argument('--window', type=int, default=None,
                        help='Smoothing window size (override task default)')
    parser.add_argument('--thresh_x', type=float, default=None,
                        help='X-axis velocity threshold (m/s)')
    parser.add_argument('--thresh_y', type=float, default=None,
                        help='Y-axis velocity threshold (m/s)')
    parser.add_argument('--thresh_z', type=float, default=None,
                        help='Z-axis velocity threshold (m/s)')
    parser.add_argument('--multi-scale', action='store_true',
                        help='Enable multi-scale window analysis (default: single window)')
    parser.add_argument('--no-slow-motion', action='store_true',
                        help='Hide the "With Slow Motion Detection" bottom row')
    parser.add_argument('--show-correction', action='store_true',
                        help='Add VL Raw + Corrected rows using events.json / intervals.json')
    parser.add_argument('--events_json', type=str, default=None,
                        help='Path to corrected events.json (auto-discovered if omitted)')
    parser.add_argument('--intervals_json', type=str, default=None,
                        help='Path to corrected intervals.json (auto-discovered if omitted)')
    parser.add_argument('--vl_fps', type=float, default=5.0,
                        help='FPS at which the VLM received video frames (default: 5)')
    parser.add_argument('--trajectory_fps', type=float, default=25.0,
                        help='Trajectory recording FPS (default: 25; used for VL\u2192traj frame mapping)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for figure')
    
    args = parser.parse_args()
    
    frame_range = None
    if args.frame_start is not None and args.frame_end is not None:
        frame_range = (args.frame_start, args.frame_end)
    
    # Build custom thresholds if any provided
    custom_thresholds = None
    base_thresh = TASK_THRESHOLDS.get(args.task, AxisThresholds())
    if args.thresh_x is not None or args.thresh_y is not None or args.thresh_z is not None:
        custom_thresholds = AxisThresholds(
            x=args.thresh_x if args.thresh_x is not None else base_thresh.x,
            y=args.thresh_y if args.thresh_y is not None else base_thresh.y,
            z=args.thresh_z if args.thresh_z is not None else base_thresh.z
        )
    
    visualize_trajectory_v3(
        hdf5_path=args.hdf5_path,
        episode=args.episode,
        task=args.task,
        frame_range=frame_range,
        custom_thresholds=custom_thresholds,
        custom_window=args.window,
        use_multi_scale=getattr(args, 'multi_scale', False),
        show_slow_motion=not getattr(args, 'no_slow_motion', False),
        show_correction=getattr(args, 'show_correction', False),
        events_json_path=getattr(args, 'events_json', None),
        intervals_json_path=getattr(args, 'intervals_json', None),
        vl_imglist_path=getattr(args, 'vl_imglist', None),
        vl_fps=getattr(args, 'vl_fps', 5.0),
        trajectory_fps=getattr(args, 'trajectory_fps', 25.0),
        output_path=args.output
    )


if __name__ == '__main__':
    main()

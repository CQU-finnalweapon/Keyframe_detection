"""
Event Corrector V3 - Using Trajectory Segmentation V3

Implements event correction based on V3 trajectory segmentation:
- V3 movement timeline: already provides moving/stable intervals with motion signatures
- V3 gripper timeline: already provides opening/closing intervals with proper filtering

Main differences from V2:
1. No Level 1 needed - V3 already separates timelines into Interval objects
2. No gap filling needed - V3 already handles gaps in segmentation
3. Directly use Interval objects instead of dict format
4. Movement intervals have motion_signature field with direction info

Author: TASE Project
Date: Jan 2025
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from .trajectory_segmentation_v3 import Interval


# ═══════════════════════════════════════════════════════════════════════════
# HELPER: ANALYZE INTERVAL CONTEXT
# ═══════════════════════════════════════════════════════════════════════════

def analyze_interval_context(
    interval: Interval,
    positions: np.ndarray,
    quaternions: Optional[np.ndarray] = None
) -> Dict:
    """
    Analyze physical context of a movement interval.
    
    For V3 intervals, we can use the motion_signature field which already
    contains direction information (e.g., "x↑, y↓, z↓").
    
    Args:
        interval: Movement interval (Interval object with motion_signature)
        positions: Robot positions (N, 3)
        quaternions: Robot quaternions (optional)
        
    Returns:
        Dict with context information
    """
    # Note: interval uses 1-indexed start/end, convert to 0-indexed for array slicing
    start = interval.start - 1
    end = interval.end - 1
    
    if end <= start or end > len(positions):
        return {
            "has_downward": False,
            "dominant_direction": np.array([0., 0., 0.]),
            "displacement": 0.,
            "duration": 0,
            "pattern": "invalid"
        }
    
    # Compute displacement and direction
    segment_positions = positions[start:end]
    displacement_vector = segment_positions[-1] - segment_positions[0]
    displacement = np.linalg.norm(displacement_vector)
    duration = end - start
    
    # Parse motion signature if available
    # motion_signature is a dict like {'x': 1, 'y': -1, 'z': -2} where:
    #   1 = fast up (↑), -1 = fast down (↓), 2 = slow up (⇑), -2 = slow down (⇓), 0 = stable
    has_downward = False
    dominant_direction = np.array([0., 0., 0.])
    
    if hasattr(interval, 'motion_signature') and interval.motion_signature:
        sig = interval.motion_signature
        if isinstance(sig, dict):
            # Parse z-axis
            z_val = sig.get('z', 0)
            if z_val < 0:  # -1 or -2 means downward
                has_downward = True
                dominant_direction[2] = -1.0
            elif z_val > 0:  # 1 or 2 means upward
                dominant_direction[2] = 1.0
            
            # Parse x-axis
            x_val = sig.get('x', 0)
            if x_val > 0:
                dominant_direction[0] = 1.0
            elif x_val < 0:
                dominant_direction[0] = -1.0
            
            # Parse y-axis
            y_val = sig.get('y', 0)
            if y_val > 0:
                dominant_direction[1] = 1.0
            elif y_val < 0:
                dominant_direction[1] = -1.0
    else:
        # Fallback: compute from displacement if no motion signature
        if displacement > 1e-4:
            dominant_direction = displacement_vector / displacement
            has_downward = dominant_direction[2] < -0.1
    
    # Extract pattern from motion signature
    pattern = interval.motion_signature if hasattr(interval, 'motion_signature') else "unknown"
    
    return {
        "has_downward": has_downward,
        "dominant_direction": dominant_direction,
        "displacement": displacement,
        "duration": duration,
        "pattern": pattern
    }


# ═══════════════════════════════════════════════════════════════════════════
# CONTEXT-AWARE MATCHING
# ═══════════════════════════════════════════════════════════════════════════

def match_attach_approach(
    all_intervals: List[Interval],
    release_interval: Interval
) -> Optional[int]:
    """
    Match attach_approach (formerly attach_hang) event.
    
    Returns the first frame of the moving stage just before release.
    Logic:
    - If release starts with moving: use first frame of that moving stage
    - If release starts with stable: use first frame of last moving stage before stable
    
    Args:
        all_intervals: List of ALL Interval objects (movement timeline)
        release_interval: Release opening Interval
        
    Returns:
        Frame index (1-indexed) or None
    """
    if not all_intervals:
        return None
    
    # Find the interval that contains release start
    release_start = release_interval.start
    interval_at_release = None
    
    for interval in all_intervals:
        if interval.start <= release_start < interval.end:
            interval_at_release = interval
            break
    
    if not interval_at_release:
        return None
    
    # Case 1: Release starts with moving - use first frame of this moving stage
    if interval_at_release.label == 'moving':
        return interval_at_release.start
    
    # Case 2: Release starts with stable - find last moving stage before it
    # Go backwards from the stable interval to find the last moving interval
    for i in range(len(all_intervals) - 1, -1, -1):
        interval = all_intervals[i]
        # Must end at or before the stable interval starts
        if interval.end <= interval_at_release.start and interval.label == 'moving':
            return interval.start
    
    # Fallback: no moving interval found
    return None


def match_attach_hang_with_context(
    all_intervals: List[Interval],
    release_interval: Interval
) -> Optional[int]:
    """
    Match attach_approach (formerly attach_hang) event.
    
    Returns the first frame of the last moving stage before release.
    Logic:
    - If release starts with moving: use first frame of that moving stage
    - If release starts with stable: use first frame of last moving stage before stable
    
    Args:
        all_intervals: List of ALL Interval objects (movement timeline)
        release_interval: Release opening Interval
        
    Returns:
        Frame index (1-indexed) or None
    """
    if not all_intervals:
        return None
    
    # Find the interval that contains release start
    release_start = release_interval.start
    interval_at_release = None
    
    for interval in all_intervals:
        if interval.start <= release_start < interval.end:
            interval_at_release = interval
            break
    
    if not interval_at_release:
        return None
    
    # Case 1: Release starts with moving - use first frame of this moving stage
    if interval_at_release.label == 'moving':
        return interval_at_release.start
    
    # Case 2: Release starts with stable - find last moving stage before it
    # Go backwards from the stable interval to find the last moving interval
    for i in range(len(all_intervals) - 1, -1, -1):
        interval = all_intervals[i]
        # Must end at or before the stable interval starts
        if interval.end <= interval_at_release.start and interval.label == 'moving':
            return interval.start
    
    # Fallback: no moving interval found
    return None


def match_attach_hang_with_context(
    moving_intervals: List[Interval],
    release_interval: Interval,
    positions: np.ndarray,
    quaternions: Optional[np.ndarray] = None,
    task: str = "square"
) -> Optional[Interval]:
    """
    DEPRECATED: Use match_attach_approach instead.
    This function is kept for backward compatibility but not used in V3.
    """
    if not moving_intervals:
        return None
    
    # Find moving intervals that end before release (convert to 1-indexed comparison)
    candidates = [m for m in moving_intervals if m.end <= release_interval.start]
    
    if not candidates:
        return None
    
    # Analyze each candidate
    scored_candidates = []
    for interval in candidates:
        context = analyze_interval_context(interval, positions, quaternions)
        
        # Task-specific scoring
        if task == "square":
            # Strict downward constraint
            if context["has_downward"] and context["dominant_direction"][2] < -0.3:
                score = 100
            else:
                score = 0
        elif task == "tool_hang":
            # Relaxed downward
            if context["has_downward"]:
                score = 80 + abs(context["dominant_direction"][2]) * 20
            else:
                score = 0
        else:
            # Default: prefer downward motion
            if context["has_downward"]:
                score = 50
            else:
                score = 10
        
        if score > 0:
            scored_candidates.append((interval, score, context))
    
    # Return best candidate
    if scored_candidates:
        scored_candidates.sort(key=lambda x: (-x[1], -x[0].start))  # Sort by score desc, then recency
        return scored_candidates[0][0]
    
    return None


def match_attach_drop_with_timing(
    release_interval: Interval,
    events: List[Dict],
    current_event_idx: int,
    frame_offset: int = 0
) -> int:
    """
    Match attach_drop using timing relative to release.
    
    Strategy:
    - Use end of opening action OR end of subtask (whichever is earlier)
    
    Args:
        release_interval: Last release opening Interval
        events: All events
        current_event_idx: Index of attach_drop event
        frame_offset: Offset to add to frame indices
        
    Returns:
        Corrected frame index
    """
    # Use end of opening interval (convert from 1-indexed to 0-indexed, then add offset)
    corrected_frame = release_interval.end - 1 + frame_offset
    
    # Check if there's a subtask end before this
    for i in range(current_event_idx + 1, len(events)):
        if events[i].get("type") in ["subtask_end", "episode_end"]:
            subtask_end_frame = events[i].get("frame_idx", float('inf'))
            corrected_frame = min(corrected_frame, subtask_end_frame)
            break
    
    return corrected_frame


# ═══════════════════════════════════════════════════════════════════════════
# MAIN EVENT CORRECTION FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

def correct_events_v3(
    events: List[Dict],
    timelines: Dict[str, List[Interval]],
    positions: np.ndarray,
    quaternions: Optional[np.ndarray] = None,
    task: str = "square",
    frame_offset: int = 0
) -> List[Dict]:
    """
    Correct event timestamps using V3 trajectory segmentation.
    
    V3 advantages:
    - Movement timeline already has proper moving/stable separation with motion signatures
    - Gripper timeline already filtered for false grasps
    - Interval objects with 1-indexed start/end (inclusive ranges)
    
    Matching rules:
    - grasp → first frame of closing gripper (closing.start - 1 in 0-indexed)
    - release → first frame of opening gripper (opening.start - 1 in 0-indexed)
    - detach → first moving frame after grasp OR end of closing (whichever later)
    - attach_approach (attach_hang) → first frame of moving stage just before release
      * If release starts with moving: use first frame of that moving stage
      * If release starts with stable: use first frame of last moving stage before stable
    - attach_drop → end of opening OR end of subtask
    
    Args:
        events: List of VL-predicted events (with 'type' and 'frame_idx')
        timelines: V3 segmentation output with 'movement' and 'gripper' Interval lists
        positions: Robot positions (N, 3)
        quaternions: Robot quaternions (optional)
        task: Task name for context-aware matching
        frame_offset: Offset to add to corrected frame indices
        
    Returns:
        Events with corrected timestamps
    """
    print(f"\n  === Event Correction V3 ===")
    print(f"  Task: {task}")
    
    movement_timeline = timelines.get("movement", [])
    gripper_timeline = timelines.get("gripper", [])
    print(f"    Movement intervals: {len(movement_timeline)}")
    print(f"    Gripper intervals: {len(gripper_timeline)}")
    
    # Separate moving and stable intervals
    moving_intervals = [m for m in movement_timeline if m.label == 'moving']
    stable_intervals = [m for m in movement_timeline if m.label == 'stable']
    print(f"    Moving: {len(moving_intervals)}, Stable: {len(stable_intervals)}")
    
    # Track used intervals
    used_gripper = set()
    used_movement = set()
    
    # Match grasp -> closing
    grasp_closing_map = {}
    grasp_idx = 0
    for event_idx, event in enumerate(events):
        if event["type"] == "grasp":
            # Find next unused closing
            for i, interval in enumerate(gripper_timeline):
                if interval.label == "closing" and i not in used_gripper:
                    used_gripper.add(i)
                    # Convert 1-indexed to 0-indexed, then add offset
                    event["corrected_frame_idx"] = interval.start - 1 + frame_offset
                    grasp_closing_map[grasp_idx] = interval
                    grasp_idx += 1
                    break
    
    # Match release -> opening
    release_opening_map = {}
    release_idx = 0
    for event_idx, event in enumerate(events):
        if event["type"] == "release":
            # Find next unused opening after last grasp
            for i, interval in enumerate(gripper_timeline):
                if interval.label == "opening" and i not in used_gripper:
                    used_gripper.add(i)
                    # Convert 1-indexed to 0-indexed, then add offset
                    event["corrected_frame_idx"] = interval.start - 1 + frame_offset
                    release_opening_map[release_idx] = interval
                    release_idx += 1
                    break
    
    # Match detach -> moving after grasp OR end of closing
    for event_idx, event in enumerate(events):
        if event["type"] == "detach":
            # Find preceding grasp
            preceding_grasp = None
            grasp_interval = None
            for i in range(event_idx - 1, -1, -1):
                if events[i]["type"] == "grasp":
                    preceding_grasp = events[i]
                    # Find corresponding closing interval
                    for g_idx, g_int in grasp_closing_map.items():
                        if g_int.start - 1 + frame_offset == preceding_grasp.get("corrected_frame_idx"):
                            grasp_interval = g_int
                            break
                    break
            
            if grasp_interval:
                # Detach logic:
                # - If robot is moving during grasp (grasp overlaps with movement), use end of closing
                # - Otherwise, use first frame of next moving stage after grasp
                # - Take whichever is later (max)
                
                # Check if grasp overlaps with any existing movement interval
                grasp_during_movement = False
                for interval in moving_intervals:
                    # Check if grasp.end falls within a movement interval [start, end)
                    if interval.start <= grasp_interval.end < interval.end:
                        grasp_during_movement = True
                        break
                
                # Default: end of closing
                detach_frame = grasp_interval.end - 1 + frame_offset
                
                # If NOT moving during grasp, try to find first moving after grasp
                if not grasp_during_movement:
                    for i, interval in enumerate(moving_intervals):
                        if interval.start > grasp_interval.end and i not in used_movement:
                            # First moving after grasp
                            first_moving_frame = interval.start - 1 + frame_offset
                            detach_frame = max(detach_frame, first_moving_frame)
                            used_movement.add(i)
                            break
                
                event["corrected_frame_idx"] = detach_frame
    
    # Context-aware matching for attach events
    print(f"  Context-Aware Matching:")
    
    for event_idx, event in enumerate(events):
        event_type = event["type"]
        has_correction = event.get("corrected_frame_idx") is not None
        
        # Skip events that already have corrections (except attach events which we always recalculate)
        if has_correction and event_type not in ["attach_hang", "attach", "attach_approach"]:
            continue
        
        if event_type in ["attach_hang", "attach", "attach_approach"]:
            # Find FOLLOWING release (attach_approach happens before release)
            following_release = None
            release_interval = None
            for i in range(event_idx + 1, len(events)):
                if events[i]["type"] == "release":
                    following_release = events[i]
                    # Find corresponding opening interval
                    for r_idx, r_int in release_opening_map.items():
                        if r_int.start - 1 + frame_offset == following_release.get("corrected_frame_idx"):
                            release_interval = r_int
                            break
                    break
            
            if release_interval:
                # Use new attach_approach logic
                all_movement_intervals = moving_intervals + stable_intervals
                # Sort by start time
                all_movement_intervals.sort(key=lambda x: x.start)
                
                approach_frame = match_attach_approach(all_movement_intervals, release_interval)
                
                if approach_frame:
                    event["corrected_frame_idx"] = approach_frame - 1 + frame_offset
                    # Find which interval this frame belongs to for logging
                    for interval in all_movement_intervals:
                        if interval.start == approach_frame:
                            print(f"    {event_type}: Frame {approach_frame} (start of {interval.label} [{interval.start}, {interval.end}))")
                            break
                else:
                    # Fallback: use last interval before release
                    for i, interval in enumerate(moving_intervals + stable_intervals):
                        if interval.end <= release_interval.start and i not in used_movement:
                            event["corrected_frame_idx"] = interval.start - 1 + frame_offset
                            break
        
        elif event_type == "attach_drop":
            # Use timing-based matching
            if release_opening_map:
                last_release_interval = release_opening_map[max(release_opening_map.keys())]
                corrected_frame = match_attach_drop_with_timing(
                    last_release_interval,
                    events,
                    event_idx,
                    frame_offset
                )
                event["corrected_frame_idx"] = corrected_frame
                print(f"    {event_type}: Timing-based match to frame {corrected_frame}")
    
    # Ensure chronological order
    last_frame = 0
    for event in events:
        event_type = event.get("type", "")
        
        # Default to VL prediction if no correction was applied
        if "corrected_frame_idx" not in event or event["corrected_frame_idx" ] is None:
            event["corrected_frame_idx"] = event["frame_idx"]
        
        # Ensure chronological order for all events
        # attach_drop naturally comes after release (it's at opening.end which is after opening.start where release is)
        if event["corrected_frame_idx"] < last_frame:
            event["corrected_frame_idx"] = last_frame + 1
        last_frame = event["corrected_frame_idx"]
    
    return events

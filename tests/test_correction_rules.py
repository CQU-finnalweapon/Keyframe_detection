#!/usr/bin/env python3
"""
Verify correction results match our defined event correction rules.

Correction Rules (V3):
- grasp: the first frame of closing gripper
- release: the first frame of opening gripper
- detach: either "first frame of next moving stage after grasp, (IMPORTANT condition) if not move during at the end of grasping" OR "end frame of gripper closing", whichever is later
- attach_approach (attach_hang): first frame of moving stage just before release
  * If release starts with moving: use first frame of that moving stage
  * If release starts with stable: use first frame of last moving stage before stable
- attach_drop: after release, end frame of opening gripper OR end frame of subtask

Known Edge Cases (marked for review):
- SEGMENTATION_MISSING_PHASE: When final z↓ insertion motion is too short/slow to be detected
  as a separate moving segment, grasp+release end up in same moving stage
- CHRONOLOGICAL_BUMP: attach_approach may get bumped up by chronological ordering enforcement

Usage:
    python tests/test_correction_rules.py --task square
    python tests/test_correction_rules.py --task square --episode 16
    python tests/test_correction_rules.py --task square --episode_list "16,17"
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime


def load_trajectory_intervals(episode_dir: str) -> Dict:
    """Load trajectory intervals JSON."""
    intervals_path = os.path.join(episode_dir, 'trajectory_intervals.json')
    if not os.path.exists(intervals_path):
        return None
    with open(intervals_path, 'r') as f:
        return json.load(f)


def load_corrected_events(episode_dir: str) -> Dict:
    """Load corrected events JSON."""
    events_path = os.path.join(episode_dir, 'corrected_events.json')
    if not os.path.exists(events_path):
        return None
    with open(events_path, 'r') as f:
        return json.load(f)


def extract_events_from_corrected(corrected_data: Dict) -> List[Dict]:
    """Extract events from corrected JSON structure."""
    events = []
    subtask_idx = 1
    while f"subtask{subtask_idx}" in corrected_data:
        subtask = corrected_data[f"subtask{subtask_idx}"]
        for phase in ['targeting', 'interaction', 'result']:
            if phase in subtask and 'connections' in subtask[phase]:
                for conn in subtask[phase]['connections']:
                    if isinstance(conn[0], list):
                        edge = conn[0]
                        primitive = conn[1]
                        frame = conn[2]
                    else:
                        edge = [conn[0], conn[1]]
                        primitive = conn[2]
                        frame = conn[3]
                    events.append({
                        'type': primitive,
                        'frame': frame,
                        'subtask': subtask_idx,
                        'phase': phase,
                        'edge': edge
                    })
        subtask_idx += 1
    # Sort by frame
    events.sort(key=lambda x: x['frame'])
    return events


def parse_intervals(intervals_data: Dict) -> Tuple[List[Dict], List[Dict]]:
    """Parse movement and gripper intervals from JSON."""
    movement = []
    for m in intervals_data.get('movement_timeline', intervals_data.get('movement', [])):
        movement.append({
            'start': m['start'],
            'end': m['end'],
            'label': m['label']
        })
    
    gripper = []
    for g in intervals_data.get('gripper_timeline', intervals_data.get('gripper', [])):
        gripper.append({
            'start': g['start'],
            'end': g['end'],
            'label': g['label']
        })
    
    return movement, gripper


def find_interval_at_frame(intervals: List[Dict], frame: int) -> Optional[Dict]:
    """Find which interval contains the given frame (1-indexed)."""
    for interval in intervals:
        if interval['start'] <= frame < interval['end']:
            return interval
    return None


def verify_grasp(event_frame: int, gripper_intervals: List[Dict], closing_idx: int) -> Tuple[bool, str, Optional[Dict]]:
    """
    Verify grasp rule: first frame of closing gripper.
    Frame should be closing.start - 1 (0-indexed).
    
    Returns matched closing interval to help with detach verification.
    
    NOTE: closing_idx is a hint but we also search by frame to handle
    spurious gripper intervals (e.g., 1-frame noise spikes).
    """
    closings = [g for g in gripper_intervals if g['label'] == 'closing']
    
    # First try to find closing by frame match (more robust)
    for closing in closings:
        expected_frame = closing['start'] - 1  # Convert to 0-indexed
        if event_frame == expected_frame:
            return True, f"✓ grasp at frame {event_frame} matches closing [{closing['start']}, {closing['end']})", closing
    
    # Fallback to index-based if no frame match
    if closing_idx >= len(closings):
        return False, f"✗ grasp at frame {event_frame}, no matching closing interval found", None
    
    closing = closings[closing_idx]
    expected_frame = closing['start'] - 1  # Convert to 0-indexed
    return False, f"✗ grasp at frame {event_frame}, expected {expected_frame} from closing [{closing['start']}, {closing['end']})", closing


def verify_release(event_frame: int, gripper_intervals: List[Dict], opening_idx: int) -> Tuple[bool, str, Optional[Dict]]:
    """
    Verify release rule: first frame of opening gripper.
    Frame should be opening.start - 1 (0-indexed).
    
    Returns matched opening interval to help with attach_drop verification.
    
    NOTE: opening_idx is a hint but we also search by frame to handle
    spurious gripper intervals.
    """
    openings = [g for g in gripper_intervals if g['label'] == 'opening']
    
    # First try to find opening by frame match (more robust)
    for opening in openings:
        expected_frame = opening['start'] - 1  # Convert to 0-indexed
        if event_frame == expected_frame:
            return True, f"✓ release at frame {event_frame} matches opening [{opening['start']}, {opening['end']})", opening
    
    # Fallback to index-based if no frame match
    if opening_idx >= len(openings):
        return False, f"✗ release at frame {event_frame}, no matching opening interval found", None
    
    opening = openings[opening_idx]
    expected_frame = opening['start'] - 1  # Convert to 0-indexed
    return False, f"✗ release at frame {event_frame}, expected {expected_frame} from opening [{opening['start']}, {opening['end']})", opening


def verify_detach(event_frame: int, grasp_frame: int, movement_intervals: List[Dict], 
                  matched_closing: Optional[Dict]) -> Tuple[bool, str]:
    """
    Verify detach rule: 
    - If grasp happens during movement: use end frame of closing
    - Otherwise: max("first frame of next moving stage after grasp", "end frame of gripper closing")
    
    Takes the matched closing interval from verify_grasp instead of an index.
    """
    if matched_closing is None:
        return False, "✗ detach cannot be verified: no matched closing interval from grasp"
    
    closing = matched_closing
    closing_end_frame = closing['end'] - 1  # Convert to 0-indexed (end-1 is last frame of closing)
    
    # Check if grasp happens during movement
    # closing.end falls within a movement interval [start, end)?
    grasp_during_movement = False
    for m in movement_intervals:
        if m['label'] == 'moving' and m['start'] <= closing['end'] < m['end']:
            grasp_during_movement = True
            break
    
    if grasp_during_movement:
        # Just use closing end
        expected_frame = closing_end_frame
        if event_frame == expected_frame:
            return True, f"✓ detach at frame {event_frame} = closing_end (grasp during movement)"
        else:
            return False, f"✗ detach at frame {event_frame}, expected {expected_frame} = closing_end (grasp during movement)"
    else:
        # Find first moving interval after grasp
        first_moving_after_grasp = None
        for m in movement_intervals:
            if m['label'] == 'moving' and m['start'] > closing['end']:
                first_moving_after_grasp = m
                break
        
        if first_moving_after_grasp:
            moving_start_frame = first_moving_after_grasp['start'] - 1  # Convert to 0-indexed
            expected_frame = max(closing_end_frame, moving_start_frame)
            
            if event_frame == expected_frame:
                return True, f"✓ detach at frame {event_frame} = max(closing_end={closing_end_frame}, moving_start={moving_start_frame})"
            else:
                return False, f"✗ detach at frame {event_frame}, expected {expected_frame} = max(closing_end={closing_end_frame}, moving_start={moving_start_frame})"
        else:
            # No moving after grasp, use closing end
            if event_frame == closing_end_frame:
                return True, f"✓ detach at frame {event_frame} = closing_end (no moving after grasp)"
            else:
                return False, f"✗ detach at frame {event_frame}, expected {closing_end_frame} = closing_end (no moving after grasp)"


def verify_attach_approach(event_frame: int, release_frame: int, movement_intervals: List[Dict],
                           gripper_intervals: List[Dict], opening_idx: int,
                           grasp_frame: int = None, detach_frame: int = None) -> Tuple[bool, str, Optional[str]]:
    """
    Verify attach_approach (attach_hang) rule:
    - Find interval containing release.start
    - If moving: use first frame of that moving stage
    - If stable: use first frame of last moving stage before stable
    
    Edge case: If grasp and release are in the SAME moving stage, 
    this indicates SEGMENTATION_MISSING_PHASE - the final insertion motion
    wasn't detected as a separate segment.
    
    Returns:
        (success, message, review_flag)
        review_flag: None if OK, or a string like "SEGMENTATION_MISSING_PHASE"
    """
    openings = [g for g in gripper_intervals if g['label'] == 'opening']
    if opening_idx >= len(openings):
        return False, f"No opening interval #{opening_idx} found", None
    
    opening = openings[opening_idx]
    release_start_1indexed = opening['start']  # 1-indexed
    
    # Find interval at release start
    interval_at_release = find_interval_at_frame(movement_intervals, release_start_1indexed)
    
    if not interval_at_release:
        return False, f"✗ No movement interval found at release frame {release_start_1indexed}", None
    
    if interval_at_release['label'] == 'moving':
        # Case 1: Release starts with moving - use first frame of this moving stage
        expected_frame = interval_at_release['start'] - 1  # Convert to 0-indexed
        
        # Edge case: Check if grasp is also in the same moving stage
        if grasp_frame is not None:
            grasp_1indexed = grasp_frame + 1
            grasp_in_same_moving = (interval_at_release['start'] <= grasp_1indexed < interval_at_release['end'])
            
            if grasp_in_same_moving:
                # SEGMENTATION ISSUE: grasp and release in same moving stage
                # This typically means the final z↓ insertion motion wasn't detected
                review_flag = "SEGMENTATION_MISSING_PHASE"
                
                # The corrector will compute expected_frame (start of moving stage)
                # But chronological enforcement bumps it to after detach
                # Accept EITHER:
                # 1. The expected frame from rule (start of moving - 1)
                # 2. The detach frame (due to chronological bump) 
                # 3. detach_frame + 1 (one frame after detach)
                
                acceptable_frames = [expected_frame]
                if detach_frame is not None:
                    acceptable_frames.extend([detach_frame, detach_frame + 1])
                
                if event_frame in acceptable_frames:
                    return True, f"✓ attach_approach at frame {event_frame} [REVIEW:{review_flag}] grasp & release in same moving [{interval_at_release['start']}, {interval_at_release['end']})", review_flag
                else:
                    return False, f"✗ attach_approach at frame {event_frame}, acceptable={acceptable_frames} [REVIEW:{review_flag}] grasp & release in same moving stage", review_flag
        
        if event_frame == expected_frame:
            return True, f"✓ attach_approach at frame {event_frame}: release at {release_start_1indexed} in moving [{interval_at_release['start']}, {interval_at_release['end']})", None
        else:
            return False, f"✗ attach_approach at frame {event_frame}, expected {expected_frame}: release at {release_start_1indexed} in moving [{interval_at_release['start']}, {interval_at_release['end']})", None
    else:
        # Case 2: Release starts with stable - find last moving before stable
        last_moving_before = None
        for m in movement_intervals:
            if m['label'] == 'moving' and m['end'] <= interval_at_release['start']:
                last_moving_before = m
        
        if last_moving_before:
            expected_frame = last_moving_before['start'] - 1  # Convert to 0-indexed
            if event_frame == expected_frame:
                return True, f"✓ attach_approach at frame {event_frame}: release at {release_start_1indexed} in stable [{interval_at_release['start']}, {interval_at_release['end']}), found moving [{last_moving_before['start']}, {last_moving_before['end']})", None
            else:
                return False, f"✗ attach_approach at frame {event_frame}, expected {expected_frame}: release in stable, should use moving [{last_moving_before['start']}, {last_moving_before['end']})", None
        else:
            return False, f"✗ attach_approach at frame {event_frame}: no moving interval found before stable [{interval_at_release['start']}, {interval_at_release['end']})", None


def verify_attach_drop(event_frame: int, release_frame: int, gripper_intervals: List[Dict],
                       opening_idx: int, total_frames: int) -> Tuple[bool, str]:
    """
    Verify attach_drop rule:
    After release, end frame of opening gripper OR end frame of subtask.
    """
    openings = [g for g in gripper_intervals if g['label'] == 'opening']
    if opening_idx >= len(openings):
        return False, f"No opening interval #{opening_idx} found"
    
    opening = openings[opening_idx]
    opening_end_frame = opening['end'] - 1  # Convert to 0-indexed
    
    # attach_drop should be at opening end or subtask end
    if event_frame == opening_end_frame or event_frame >= total_frames - 5:  # Allow some tolerance for subtask end
        return True, f"✓ attach_drop at frame {event_frame} matches opening end [{opening['start']}, {opening['end']})"
    else:
        return False, f"✗ attach_drop at frame {event_frame}, expected {opening_end_frame} from opening [{opening['start']}, {opening['end']})"


def verify_episode(episode_dir: str, verbose: bool = True) -> Tuple[int, int, List[str], List[str]]:
    """
    Verify all correction rules for a single episode.
    
    Returns:
        (passed_count, total_count, error_messages, review_flags)
    """
    intervals_data = load_trajectory_intervals(episode_dir)
    corrected_data = load_corrected_events(episode_dir)
    
    if not intervals_data or not corrected_data:
        return 0, 0, [f"Missing data files in {episode_dir}"], []
    
    movement_intervals, gripper_intervals = parse_intervals(intervals_data)
    events = extract_events_from_corrected(corrected_data)
    
    passed = 0
    total = 0
    errors = []
    review_flags = []  # Track episodes that need manual review
    
    # Track indices for multi-subtask scenarios
    closing_idx = 0
    opening_idx = 0
    grasp_frame = None
    release_frame = None
    detach_frame = None
    matched_closing = None  # Track the matched closing interval for this subtask
    matched_opening = None  # Track the matched opening interval for this subtask
    
    for event in events:
        event_type = event['type']
        event_frame = event['frame']
        
        if event_type == 'grasp':
            total += 1
            success, msg, matched_closing = verify_grasp(event_frame, gripper_intervals, closing_idx)
            if success:
                passed += 1
            else:
                errors.append(msg)
            if verbose:
                print(f"  {msg}")
            grasp_frame = event_frame
            closing_idx += 1
            
        elif event_type == 'release':
            total += 1
            success, msg, matched_opening = verify_release(event_frame, gripper_intervals, opening_idx)
            if success:
                passed += 1
            else:
                errors.append(msg)
            if verbose:
                print(f"  {msg}")
            release_frame = event_frame
            opening_idx += 1
            
        elif event_type == 'detach':
            total += 1
            if grasp_frame is not None:
                success, msg = verify_detach(event_frame, grasp_frame, movement_intervals, 
                                            matched_closing)
            else:
                success, msg = False, "✗ detach without preceding grasp"
            if success:
                passed += 1
            else:
                errors.append(msg)
            if verbose:
                print(f"  {msg}")
            detach_frame = event_frame
                
        elif event_type in ['attach_hang', 'attach_approach', 'attach']:
            total += 1
            # Find the following release for this attach event
            following_release_frame = None
            for e in events:
                if e['type'] == 'release' and e['frame'] > event_frame:
                    following_release_frame = e['frame']
                    break
            
            if following_release_frame is not None:
                # Find which opening index corresponds to this release
                current_opening_idx = opening_idx
                success, msg, review_flag = verify_attach_approach(event_frame, following_release_frame, 
                                                     movement_intervals, gripper_intervals, current_opening_idx,
                                                     grasp_frame=grasp_frame, detach_frame=detach_frame)
                if review_flag:
                    review_flags.append(review_flag)
            else:
                success, msg = False, f"✗ {event_type} without following release"
            if success:
                passed += 1
            else:
                errors.append(msg)
            if verbose:
                print(f"  {msg}")
                
        elif event_type == 'attach_drop':
            total += 1
            if release_frame is not None:
                success, msg = verify_attach_drop(event_frame, release_frame, gripper_intervals,
                                                 opening_idx - 1, intervals_data.get('total_frames', 1000))
            else:
                success, msg = False, "✗ attach_drop without preceding release"
            if success:
                passed += 1
            else:
                errors.append(msg)
            if verbose:
                print(f"  {msg}")
    
    return passed, total, errors, review_flags


def verify_task(task: str, data_root: str, episodes: List[int] = None, verbose: bool = True) -> Dict:
    """
    Verify all episodes for a task.
    
    Returns:
        Summary dict with passed/total counts and review cases
    """
    task_dir = os.path.join(data_root, 'corrected_events', task)
    
    if not os.path.exists(task_dir):
        print(f"Task directory not found: {task_dir}")
        return {'passed': 0, 'total': 0, 'episodes_passed': 0, 'episodes_total': 0}
    
    total_passed = 0
    total_events = 0
    episodes_passed = 0
    episodes_total = 0
    all_errors = {}
    all_reviews = {}  # Track episodes needing manual review
    
    # Find all episode directories
    episode_dirs = []
    for range_folder in sorted(os.listdir(task_dir)):
        range_path = os.path.join(task_dir, range_folder)
        if os.path.isdir(range_path) and '_' in range_folder:
            for ep_folder in sorted(os.listdir(range_path)):
                if ep_folder.startswith('episode_'):
                    ep_idx = int(ep_folder.split('_')[1])
                    if episodes is None or ep_idx in episodes:
                        episode_dirs.append((ep_idx, os.path.join(range_path, ep_folder)))
    
    episode_dirs.sort(key=lambda x: x[0])
    
    print(f"\n{'='*60}")
    print(f"Verifying Task: {task}")
    print(f"Episodes to verify: {len(episode_dirs)}")
    print('='*60)
    
    for ep_idx, ep_dir in episode_dirs:
        episodes_total += 1
        if verbose:
            print(f"\n--- Episode {ep_idx} ---")
        
        passed, total, errors, reviews = verify_episode(ep_dir, verbose=verbose)
        total_passed += passed
        total_events += total
        
        if reviews:
            all_reviews[ep_idx] = reviews
        
        if total > 0 and passed == total:
            episodes_passed += 1
            if verbose:
                if reviews:
                    print(f"  ✓ All {passed}/{total} events verified [NEEDS REVIEW: {reviews}]")
                else:
                    print(f"  ✓ All {passed}/{total} events verified")
        elif total > 0:
            all_errors[ep_idx] = errors
            if verbose:
                print(f"  ✗ {passed}/{total} events verified")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Task: {task} - Summary")
    print(f"{'='*60}")
    print(f"Episodes: {episodes_passed}/{episodes_total} passed ({100*episodes_passed/max(episodes_total,1):.1f}%)")
    print(f"Events: {total_passed}/{total_events} passed ({100*total_passed/max(total_events,1):.1f}%)")
    
    if all_errors:
        print(f"\nFailed episodes: {sorted(all_errors.keys())}")
    
    if all_reviews:
        print(f"\n{'='*60}")
        print("Episodes needing manual review:")
        print('='*60)
        for ep_idx, flags in sorted(all_reviews.items()):
            print(f"  Episode {ep_idx}: {', '.join(flags)}")
        
        # Write review file
        review_file = os.path.join(data_root, 'corrected_events', task, f'review_{task}.txt')
        with open(review_file, 'w') as f:
            f.write(f"# Manual Review Required - {task}\n")
            f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("# These episodes have edge cases that may need trajectory segmentation adjustment\n\n")
            for ep_idx, flags in sorted(all_reviews.items()):
                f.write(f"Episode {ep_idx}:\n")
                for flag in flags:
                    if flag == "SEGMENTATION_MISSING_PHASE":
                        f.write(f"  - {flag}: Final insertion motion (z↓) not detected as separate segment\n")
                        f.write(f"    Consider lowering 'z' velocity threshold for this task\n")
                    else:
                        f.write(f"  - {flag}\n")
                f.write("\n")
        print(f"\nReview file written to: {review_file}")
    
    return {
        'passed': total_passed,
        'total': total_events,
        'episodes_passed': episodes_passed,
        'episodes_total': episodes_total,
        'errors': all_errors,
        'reviews': all_reviews
    }


def main():
    parser = argparse.ArgumentParser(description='Verify correction results against defined rules')
    parser.add_argument('--task', type=str, required=True, 
                        help='Task name (can, square, tool_hang)')
    parser.add_argument('--episode', type=int, default=None,
                        help='Single episode to verify')
    parser.add_argument('--episode_list', type=str, default=None,
                        help='Comma-separated list of episodes (e.g., "5,7,9")')
    parser.add_argument('--data_root', type=str, default=None,
                        help='Root directory for generated data (default: ./data)')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Only show summary, not per-event details')
    
    args = parser.parse_args()
    
    # Determine data root
    if args.data_root:
        data_root = os.path.expanduser(args.data_root)
    else:
        # Default: relative to script location
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_root = os.path.join(script_dir, 'data')
    
    # Parse episodes
    episodes = None
    if args.episode is not None:
        episodes = [args.episode]
    elif args.episode_list is not None:
        episodes = [int(ep.strip()) for ep in args.episode_list.split(',')]
    
    # Verify
    result = verify_task(args.task, data_root, episodes, verbose=not args.quiet)
    
    # Exit code based on success
    if result['episodes_passed'] == result['episodes_total'] and result['episodes_total'] > 0:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()

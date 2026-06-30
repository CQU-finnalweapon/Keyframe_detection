#!/usr/bin/env python3
"""
Merge Trajectory and VL Analysis Pipeline

End-to-end script that:
1. Loads robot trajectory data from HDF5
2. Segments trajectory into intervals (moving/stable, closing/opening)
3. Loads VL model JSON output with interaction events
4. Corrects VL event timestamps using trajectory intervals
5. Saves corrected VL JSON with accurate timestamps

Usage:
    python merge_trajectory_vl.py \
        --hdf5_path ./data/robomimic/datasets/can/ph/low_dim_abs.hdf5 \
        --vl_json ./data/divided_events/can/json_sg/imglist_episode_0_*.json \
        --episode 0 \
        --output_dir ./data/corrected_events/
"""

import os
import sys
import argparse
import h5py
import numpy as np
import json
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.trajectory_segmentation_v3 import (
    segment_trajectory_v3,
    format_timelines_for_output,
    format_timelines_readable,
    parse_trajectory_intervals_v3
)
from src.vl_parser import (
    load_vl_json,
    parse_vl_events,
    save_corrected_vl_json,
    get_vl_fps
)
from src.event_corrector_v3 import correct_events_v3
from src.trajectory_utils import load_trajectory_from_hdf5
import config as _cfg

# Global verbosity flag
VERBOSE = True

def log(msg: str):
    """Print only if verbose mode is enabled."""
    if VERBOSE:
        print(msg)


def print_correction_summary(original_events, corrected_events):
    """Print a summary of event corrections."""
    print("\n  === Correction Summary ===")
    total_shift = 0
    corrections = 0
    for orig, corr in zip(original_events, corrected_events):
        orig_frame = orig.get('frame_idx', orig.get('vl_frame', 0))
        corr_frame = corr.get('corrected_frame_idx', corr.get('frame_idx', 0))
        shift = corr_frame - orig_frame
        total_shift += abs(shift)
        corrections += 1
        print(f"  {corr['type']}: {orig_frame} → {corr_frame} (shift: {shift:+d})")
    print(f"  Total: {corrections} corrections, {total_shift} frames shifted")


def get_episode_subfolder(episode_idx: int, episodes_per_folder: int = 20) -> str:
    """
    Get subfolder name for episode organization.
    
    Args:
        episode_idx: Episode index
        episodes_per_folder: Number of episodes per subfolder (default: 20)
        
    Returns:
        Subfolder name like '000_019', '020_039', etc.
    """
    start = (episode_idx // episodes_per_folder) * episodes_per_folder
    end = start + episodes_per_folder - 1
    return f"{start:03d}_{end:03d}"


def main():
    parser = argparse.ArgumentParser(description="Merge trajectory and VL analysis")
    parser.add_argument('--hdf5_path', type=str, required=True,
                        help='Path to HDF5 dataset file')
    parser.add_argument('--vl_json', type=str, required=True,
                        help='Path to VL model JSON output (template for all episodes)')
    parser.add_argument('--episode', type=int, default=None,
                        help='Single episode index to process (if not specified, process all episodes)')
    parser.add_argument('--num_episodes', type=int, default=None,
                        help='Number of episodes to process from 0 to N-1 (if not specified, process all in HDF5)')
    parser.add_argument('--episode_list', type=str, default=None,
                        help='Comma-separated list of episode indices to process (e.g., "5,7,9")')
    parser.add_argument('--output_dir', type=str, default='./data/corrected_events/',
                        help='Output directory for corrected JSON')
    parser.add_argument('--demo_prefix', type=str, default='demo',
                        help='Prefix for demo keys in HDF5')
    
    # Trajectory segmentation parameters
    parser.add_argument('--position_threshold', type=float, default=5e-3,
                        help='Position movement threshold (m)')
    parser.add_argument('--rotation_threshold', type=float, default=1e-2,
                        help='Rotation movement threshold (rad)')
    parser.add_argument('--gripper_threshold', type=float, default=1e-4,
                        help='Gripper change threshold')
    parser.add_argument('--stable_window', type=int, default=5,
                        help='Window size for stable detection')
    parser.add_argument('--max_gap', type=int, default=5,
                        help='Maximum gap to fill between similar-direction large motions')
    parser.add_argument('--direction_threshold_deg', type=float, default=90.0,
                        help='Angle threshold (degrees) for direction-based splitting (default: 90)')
    parser.add_argument('--direction_similarity_threshold', type=float, default=0.5,
                        help='Cosine similarity threshold for direction-based gap filling (default: 0.5)')
    parser.add_argument('--no-gap-filling', action='store_true',
                        help='Disable gap filling to see raw segmentation')
    
    # Timestamp correction parameters
    parser.add_argument('--frame_offset', type=int, default=0,
                        help='Offset between trajectory steps and video frames')
    parser.add_argument('--sampling_interval', type=float, default=0.04,
                        help='Time between trajectory steps (seconds, e.g., 0.04 for 25Hz)')
    
    # Output control
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Reduce terminal output (only show summary)')
    parser.add_argument('--debug_segmentation', action='store_true',
                        help='Include stage-by-stage segmentation breakdown in trajectory_intervals.txt')
    parser.add_argument('--task', type=str, default='square',
                        help='Task name for V3 segmentation (tool_hang, square, can)')
    
    args = parser.parse_args()
    
    # Set global verbosity
    global VERBOSE
    VERBOSE = not args.quiet
    
    # Expand paths
    args.hdf5_path = os.path.expanduser(args.hdf5_path)
    args.vl_json = os.path.expanduser(args.vl_json)
    args.output_dir = os.path.expanduser(args.output_dir)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine which episodes to process
    if args.episode is not None:
        # Process single episode
        episodes_to_process = [args.episode]
    elif args.episode_list is not None:
        # Process specific list of episodes
        episodes_to_process = [int(ep.strip()) for ep in args.episode_list.split(',')]
    else:
        # Process multiple episodes from 0 to N-1
        with h5py.File(args.hdf5_path, 'r') as f:
            all_demos = sorted([k for k in f['data'].keys() if k.startswith(args.demo_prefix)])
            if args.num_episodes is not None:
                episodes_to_process = list(range(min(args.num_episodes, len(all_demos))))
            else:
                episodes_to_process = list(range(len(all_demos)))
    
    print("=" * 80)
    print("Trajectory-VL Merge Pipeline")
    print("=" * 80)
    print(f"Processing {len(episodes_to_process)} episodes")
    print("=" * 80)
    
    # Check if VL JSON is per-episode or template
    vl_json_is_template = 'episode_0' in args.vl_json or '_episode_0_' in args.vl_json
    
    # Try to detect per-episode VL data sources
    vl_events_dir = None
    if os.path.exists(args.vl_json):
        # Single template file provided
        log("\n[Initialization] Loading VL model output template...")
        vl_data = load_vl_json(args.vl_json)
        template_events = parse_vl_events(vl_data)
        log(f"  Template has {len(template_events)} events")
        log(f"  NOTE: Using same template for all episodes (VL events may not match!)")
    else:
        # Check if it's a directory pattern
        vl_events_dir = os.path.dirname(args.vl_json)
        log(f"\n[Initialization] Will load per-episode VL events from: {vl_events_dir}")
        vl_data = None
        template_events = None
    
    # Process each episode
    successful_episodes = 0
    failed_episodes = []
    
    for ep_idx in episodes_to_process:
        log(f"\n{'='*80}")
        log(f"Processing Episode {ep_idx}")
        log('='*80)
        
        try:
            # Step 1: Load trajectory data
            log(f"\n[Step 1/5] Loading trajectory data...")
            trajectory, action_gripper, _ = load_trajectory_from_hdf5(
                args.hdf5_path,
                ep_idx,
                args.demo_prefix
            )
            log(f"  Trajectory length: {len(trajectory)} timesteps")
            
            # Step 2: Segment trajectory (V3)
            log(f"\n[Step 2/5] Segmenting trajectory (V3 - task: {args.task})...")
            timelines = segment_trajectory_v3(
                trajectory,
                action_gripper,
                task=args.task,
                gripper_threshold=args.gripper_threshold,
                max_gap=args.max_gap
            )
            debug_info = None
            
            # Print timeline summaries (only in verbose mode)
            movement_intervals = timelines['movement']
            gripper_intervals = timelines['gripper']
            log(f"  Movement timeline: {len(movement_intervals)} intervals")
            for iv in movement_intervals:
                dir_info = ""
                if iv.direction is not None and iv.label == "moving":
                    dir_info = f" dir=[{iv.direction[0]:.2f}, {iv.direction[1]:.2f}, {iv.direction[2]:.2f}]"
                log(f"    {iv.to_string()}{dir_info}")
            log(f"  Gripper timeline: {len(gripper_intervals)} intervals")
            for iv in gripper_intervals:
                log(f"    {iv.to_string()}")
            
            # Format for event corrector
            formatted = format_timelines_for_output(timelines)
            trajectory_intervals = parse_trajectory_intervals_v3(formatted)
            
            # Step 3: Load VL events (per-episode if available, else use template)
            import copy
            ep_vl_data = None
            ep_vl_events = None
            
            # Try to find per-episode VL events file
            if vl_events_dir:
                # Look for per-episode file with pattern
                episode_vl_pattern = args.vl_json.replace('episode_0', f'episode_{ep_idx}')
                if os.path.exists(episode_vl_pattern):
                    ep_vl_data = load_vl_json(episode_vl_pattern)
                    ep_vl_events = parse_vl_events(ep_vl_data)
                    log(f"  Loaded per-episode VL events: {len(ep_vl_events)} events")
            
            # Fall back to template if no per-episode file
            if ep_vl_events is None and template_events is not None:
                ep_vl_events = copy.deepcopy(template_events)
                ep_vl_data = vl_data
                log(f"  Using template VL events: {len(ep_vl_events)} events (may not match episode!)")
            
            if ep_vl_events is None:
                raise ValueError("No VL events available for this episode")
            
            # Step 4: Match events to trajectory and correct timestamps
            log(f"\n[Step 3/5] Matching events to trajectory intervals...")
            # V3 uses direct Interval objects, no dict conversion needed
            corrected_events = correct_events_v3(
                ep_vl_events,
                timelines,
                trajectory[:, :3],  # positions
                None,  # quaternions (optional)
                task=args.task,
                frame_offset=args.frame_offset
            )
            
            # Print correction summary (only in verbose mode)
            if VERBOSE:
                log(f"\n[Step 4/5] Correction summary...")
                print_correction_summary(ep_vl_events, corrected_events)
            
            # Step 5: Save corrected JSON
            log(f"\n[Step 5/5] Saving corrected JSON...")
            
            # Create subfolder for this episode range, then episode-specific subfolder
            range_subfolder = get_episode_subfolder(ep_idx)
            episode_subfolder = f"episode_{ep_idx}"
            episode_dir = os.path.join(args.output_dir, range_subfolder, episode_subfolder)
            os.makedirs(episode_dir, exist_ok=True)
            
            # File naming: direct names (no episode_ prefix since folder is episode-specific)
            output_filename = "corrected_events.json"
            output_path = os.path.join(episode_dir, output_filename)
            
            save_corrected_vl_json(corrected_events, ep_vl_data, output_path)
            log(f"  Saved to: {output_path}")
            
            # Save trajectory intervals in V2 format (separate timelines)
            intervals_output_path = os.path.join(episode_dir, "trajectory_intervals.json")
            
            # Convert Interval objects to serializable format with readable strings
            movement_list = []
            for iv in movement_intervals:
                d = {
                    'interval': iv.to_string(),  # Readable: "[1, 35)(moving)"
                    'start': iv.start, 
                    'end': iv.end, 
                    'label': iv.label
                }
                if iv.direction is not None:
                    d['direction'] = [round(x, 3) for x in iv.direction.tolist()]
                    d['direction_str'] = f"[{iv.direction[0]:.2f}, {iv.direction[1]:.2f}, {iv.direction[2]:.2f}]"
                if iv.speed is not None:
                    d['speed'] = round(float(iv.speed), 4)
                movement_list.append(d)
            
            gripper_list = []
            for iv in gripper_intervals:
                gripper_list.append({
                    'interval': iv.to_string(),  # Readable: "[53, 59)(closing)"
                    'start': iv.start, 
                    'end': iv.end, 
                    'label': iv.label
                })
            
            with open(intervals_output_path, 'w') as f:
                json.dump({
                    "episode": ep_idx,
                    "version": "v2",
                    "summary": {
                        "movement_count": len(movement_intervals),
                        "gripper_count": len(gripper_intervals),
                        "moving_intervals": sum(1 for iv in movement_intervals if iv.label == 'moving'),
                        "stable_intervals": sum(1 for iv in movement_intervals if iv.label == 'stable'),
                        "adjust_intervals": sum(1 for iv in movement_intervals if iv.label == 'adjust'),
                        "opening_intervals": sum(1 for iv in gripper_intervals if iv.label == 'opening'),
                        "closing_intervals": sum(1 for iv in gripper_intervals if iv.label == 'closing')
                    },
                    "movement_timeline": movement_list,
                    "gripper_timeline": gripper_list,
                    "parameters": {
                        "position_threshold": args.position_threshold,
                        "rotation_threshold": args.rotation_threshold,
                        "gripper_threshold": args.gripper_threshold,
                        "stable_window": args.stable_window,
                        "max_gap": args.max_gap,
                        "direction_threshold_deg": args.direction_threshold_deg
                    }
                }, f, indent=2)
            
            # Also save readable txt version
            txt_output_path = os.path.join(episode_dir, "trajectory_intervals.txt")
            readable_txt = format_timelines_readable(timelines, episode=ep_idx)
            with open(txt_output_path, 'w') as f:
                f.write(readable_txt)
            
            successful_episodes += 1
            print(f"✓ Episode {ep_idx} completed successfully")
            
        except Exception as e:
            print(f"✗ Episode {ep_idx} failed: {str(e)}")
            import traceback
            traceback.print_exc()
            failed_episodes.append(ep_idx)
            continue
    
    print("\n" + "=" * 80)
    print("Pipeline Summary")
    print("=" * 80)
    print(f"Successfully processed: {successful_episodes}/{len(episodes_to_process)} episodes")
    if failed_episodes:
        print(f"Failed episodes: {failed_episodes}")
    print("=" * 80)


if __name__ == "__main__":
    main()

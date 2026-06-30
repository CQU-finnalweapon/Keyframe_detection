#!/usr/bin/env python3
"""
Automatic Threshold Optimization for Trajectory Segmentation V3.

This script analyzes trajectory data to compute optimal per-axis velocity thresholds.

Approach:
1. Analyze velocity distributions per axis across multiple episodes
2. Identify "movement" vs "stable" phases using gripper state as ground truth
3. Find optimal thresholds that best separate the two distributions
4. Validate by checking that expected segment structures are detected

The key insight is that:
- During "stable" phases (before grasp, after release), velocities should be near zero
- During "moving" phases, velocities should be higher
- The threshold should be set to distinguish these two modes

Usage:
    python scripts/optimize_thresholds.py --task square --num_episodes 20
    python scripts/optimize_thresholds.py --task square --analyze_episode 10
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

# Add project root to path
script_dir = Path(__file__).parent.parent
sys.path.insert(0, str(script_dir))

from src.trajectory_segmentation_v3 import (
    TASK_THRESHOLDS,
    smooth_trajectory,
    compute_motion_labels_per_axis,
    segment_movement_timeline_v3
)
import config as _cfg


def load_trajectory(dataset_folder: str, episode_idx: int) -> Tuple[np.ndarray, np.ndarray, float]:
    """Load trajectory positions and gripper from HDF5 file."""
    import h5py
    
    # Try different possible file names
    possible_files = ['demo.hdf5', 'low_dim.hdf5', 'low_dim_abs.hdf5', 'image.hdf5', 'image_abs.hdf5']
    demo_file = None
    
    for fname in possible_files:
        fpath = os.path.join(dataset_folder, fname)
        if os.path.exists(fpath):
            demo_file = fpath
            break
    
    if demo_file is None:
        # Try parent folder (dataset_folder might be .../video/ but file is in .../ph/)
        parent = os.path.dirname(dataset_folder.rstrip('/'))
        for fname in possible_files:
            fpath = os.path.join(parent, fname)
            if os.path.exists(fpath):
                demo_file = fpath
                break
    
    if demo_file is None:
        raise FileNotFoundError(f"No HDF5 file found in {dataset_folder} or parent")
    
    with h5py.File(demo_file, 'r') as f:
        demo_key = f"data/demo_{episode_idx}"
        if demo_key not in f:
            raise KeyError(f"Episode {episode_idx} not found in {demo_file}")
        
        demo = f[demo_key]
        
        # Get end-effector position (first 3 columns of obs)
        if 'obs/robot0_eef_pos' in demo:
            positions = demo['obs/robot0_eef_pos'][:]
        else:
            # Fallback: try to extract from states
            states = demo['states'][:]
            positions = states[:, :3]  # Assume first 3 are position
        
        # Get gripper state
        if 'obs/robot0_gripper_qpos' in demo:
            gripper = demo['obs/robot0_gripper_qpos'][:, 0]  # First gripper DOF
        else:
            gripper = np.zeros(len(positions))
        
        # Get attributes
        attrs = dict(demo.attrs)
        fps = 20.0  # Default
    
    return positions, gripper, fps


def compute_velocities(positions: np.ndarray, fps: float = 20.0) -> np.ndarray:
    """Compute velocities from positions."""
    dt = 1.0 / fps
    velocities = np.diff(positions, axis=0) / dt
    # Pad to match length
    velocities = np.vstack([velocities, velocities[-1:]])
    return velocities


def analyze_velocity_distribution(velocities: np.ndarray, axis_name: str = 'x') -> Dict:
    """Analyze velocity distribution for one axis."""
    axis_idx = {'x': 0, 'y': 1, 'z': 2}[axis_name]
    v = velocities[:, axis_idx]
    
    return {
        'mean': float(np.mean(np.abs(v))),
        'std': float(np.std(v)),
        'median': float(np.median(np.abs(v))),
        'p25': float(np.percentile(np.abs(v), 25)),
        'p50': float(np.percentile(np.abs(v), 50)),
        'p75': float(np.percentile(np.abs(v), 75)),
        'p90': float(np.percentile(np.abs(v), 90)),
        'p95': float(np.percentile(np.abs(v), 95)),
        'max': float(np.max(np.abs(v))),
    }


def find_optimal_threshold_per_axis(
    velocities: np.ndarray,
    gripper: np.ndarray,
    axis: int,
    gripper_threshold: float = 0.5
) -> Tuple[float, Dict]:
    """
    Find optimal velocity threshold for an axis using gripper state as reference.
    
    The idea: during gripper closing/opening, the robot is typically stationary or slow.
    We can use this as a proxy for "stable" state.
    
    Args:
        velocities: (N, 3) velocity array
        gripper: (N,) gripper position array (normalized 0-1)
        axis: 0=x, 1=y, 2=z
        gripper_threshold: threshold to determine gripper state
    
    Returns:
        optimal_threshold, analysis_dict
    """
    v = np.abs(velocities[:, axis])
    
    # Compute gripper velocity to detect opening/closing
    gripper_vel = np.abs(np.diff(gripper, prepend=gripper[0]))
    gripper_active = gripper_vel > 0.001  # Gripper is moving
    
    # During gripper activity, robot is usually stable or slow
    stable_velocities = v[gripper_active]
    moving_velocities = v[~gripper_active]
    
    # Also consider very slow velocities as stable
    slow_mask = v < np.percentile(v, 25)  # Bottom quartile
    fast_mask = v > np.percentile(v, 75)  # Top quartile
    
    # Analysis
    analysis = {
        'gripper_active_mean': float(np.mean(stable_velocities)) if len(stable_velocities) > 0 else 0,
        'gripper_inactive_mean': float(np.mean(moving_velocities)) if len(moving_velocities) > 0 else 0,
        'slow_p75': float(np.percentile(v[slow_mask], 75)) if np.sum(slow_mask) > 0 else 0,
        'fast_p25': float(np.percentile(v[fast_mask], 25)) if np.sum(fast_mask) > 0 else 0,
    }
    
    # Optimal threshold: between stable mean and moving mean
    # Or: at the 50th percentile of gripper-active velocities (upper bound of stable)
    if len(stable_velocities) > 0 and len(moving_velocities) > 0:
        stable_upper = np.percentile(stable_velocities, 90)  # 90th percentile of stable
        moving_lower = np.percentile(moving_velocities, 25)  # 25th percentile of moving
        
        # Threshold should be between stable upper and moving lower
        optimal = (stable_upper + moving_lower) / 2
        
        # But also consider: threshold should capture at least the slow velocities
        # and distinguish from fast velocities
        alt_optimal = (analysis['slow_p75'] + analysis['fast_p25']) / 2
        
        # Use the more conservative (lower) threshold
        optimal = min(optimal, alt_optimal) if alt_optimal > 0 else optimal
    else:
        # Fallback: use percentile-based approach
        optimal = np.percentile(v, 50)
    
    analysis['optimal_threshold'] = float(optimal)
    
    return optimal, analysis


def analyze_task_thresholds(
    dataset_folder: str,
    task: str,
    num_episodes: int = 20,
    verbose: bool = True
) -> Dict:
    """
    Analyze multiple episodes to compute optimal thresholds for a task.
    
    Returns:
        Dict with recommended thresholds and analysis
    """
    all_velocities = []
    all_grippers = []
    episode_analyses = []
    
    print(f"\n{'='*60}")
    print(f"Analyzing Task: {task}")
    print(f"Dataset: {dataset_folder}")
    print(f"Episodes to analyze: {num_episodes}")
    print('='*60)
    
    for ep_idx in range(num_episodes):
        try:
            positions, gripper, fps = load_trajectory(dataset_folder, ep_idx)
            velocities = compute_velocities(positions, fps)
            
            all_velocities.append(velocities)
            all_grippers.append(gripper)
            
            if verbose:
                print(f"  Episode {ep_idx}: {len(positions)} frames")
                
        except Exception as e:
            if verbose:
                print(f"  Episode {ep_idx}: Error - {e}")
            continue
    
    if not all_velocities:
        print("No valid episodes found!")
        return {}
    
    # Concatenate all velocities
    combined_velocities = np.vstack(all_velocities)
    combined_gripper = np.hstack(all_grippers)
    
    # Analyze per-axis
    results = {
        'task': task,
        'episodes_analyzed': len(all_velocities),
        'total_frames': len(combined_velocities),
        'current_thresholds': TASK_THRESHOLDS.get(task, {'x': 0.030, 'y': 0.030, 'z': 0.030}),
        'axis_analysis': {},
        'recommended_thresholds': {}
    }
    
    print("\nPer-Axis Velocity Analysis:")
    print("-" * 60)
    
    for axis_idx, axis_name in enumerate(['x', 'y', 'z']):
        dist = analyze_velocity_distribution(combined_velocities, axis_name)
        opt_thresh, opt_analysis = find_optimal_threshold_per_axis(
            combined_velocities, combined_gripper, axis_idx
        )
        
        results['axis_analysis'][axis_name] = {
            'distribution': dist,
            'optimization': opt_analysis
        }
        results['recommended_thresholds'][axis_name] = opt_thresh
        
        current = results['current_thresholds'].get(axis_name, 0.030)
        
        print(f"\n  {axis_name.upper()}-axis:")
        print(f"    Velocity stats: mean={dist['mean']:.4f}, median={dist['median']:.4f}, p75={dist['p75']:.4f}, p95={dist['p95']:.4f}")
        print(f"    Current threshold: {current:.4f} m/s")
        print(f"    Recommended threshold: {opt_thresh:.4f} m/s ({'+' if opt_thresh > current else ''}{(opt_thresh - current)*1000:.1f} mm/s)")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("\nCurrent thresholds:")
    for axis in ['x', 'y', 'z']:
        print(f"  {axis}: {results['current_thresholds'].get(axis, 0.030):.4f} m/s")
    
    print("\nRecommended thresholds:")
    for axis in ['x', 'y', 'z']:
        print(f"  {axis}: {results['recommended_thresholds'][axis]:.4f} m/s")
    
    print("\nPython code to update TASK_THRESHOLDS:")
    print(f"    '{task}': {{'x': {results['recommended_thresholds']['x']:.3f}, 'y': {results['recommended_thresholds']['y']:.3f}, 'z': {results['recommended_thresholds']['z']:.3f}}},")
    
    return results


def analyze_single_episode(
    dataset_folder: str,
    task: str,
    episode_idx: int,
    verbose: bool = True
) -> Dict:
    """
    Analyze a single episode to understand why segmentation might be wrong.
    """
    print(f"\n{'='*60}")
    print(f"Episode Analysis: {task} - Episode {episode_idx}")
    print('='*60)
    
    positions, gripper, fps = load_trajectory(dataset_folder, episode_idx)
    velocities = compute_velocities(positions, fps)
    
    print(f"\nTrajectory: {len(positions)} frames at {fps} fps ({len(positions)/fps:.2f} seconds)")
    
    # Analyze velocity distribution
    print("\nVelocity Distribution:")
    print("-" * 40)
    for axis_idx, axis_name in enumerate(['x', 'y', 'z']):
        v = velocities[:, axis_idx]
        current_thresh = TASK_THRESHOLDS.get(task, {}).get(axis_name, 0.030)
        
        # Count frames above/below threshold
        above = np.sum(np.abs(v) > current_thresh)
        below = len(v) - above
        
        print(f"\n  {axis_name.upper()}-axis:")
        print(f"    Range: [{v.min():.4f}, {v.max():.4f}] m/s")
        print(f"    Mean: {np.mean(v):.4f}, Std: {np.std(v):.4f}")
        print(f"    |v| percentiles: p25={np.percentile(np.abs(v), 25):.4f}, p50={np.percentile(np.abs(v), 50):.4f}, p75={np.percentile(np.abs(v), 75):.4f}")
        print(f"    Current threshold: {current_thresh:.4f}")
        print(f"    Frames above threshold: {above} ({100*above/len(v):.1f}%)")
    
    # Show current segmentation
    print("\nCurrent Segmentation with existing thresholds:")
    print("-" * 40)
    
    current_thresholds = TASK_THRESHOLDS.get(task, {'x': 0.030, 'y': 0.030, 'z': 0.030})
    smoothed = smooth_trajectory(positions)

    intervals = segment_movement_timeline_v3(positions, task=task)
    
    print(f"  Segments: {len(intervals)}")
    for interval in intervals:
        sig = interval.to_string() if hasattr(interval, 'to_string') else interval.label
        print(f"    [{interval.start:3d}, {interval.end:3d}) {interval.label:8s} {sig}")
    
    # Check for the final z-down motion (for insert/hang tasks)
    print("\nChecking for final z-down motion:")
    print("-" * 40)
    
    # Find where z-velocity becomes negative near the end
    z_vel = velocities[:, 2]
    
    # Look at last 40% of trajectory
    end_start = int(len(z_vel) * 0.6)
    z_end = z_vel[end_start:]
    
    # Find frames with negative z velocity (downward motion)
    downward_frames = np.where(z_end < -current_thresholds['z'])[0] + end_start
    
    if len(downward_frames) > 0:
        print(f"  Frames with z↓ motion (z_vel < -{current_thresholds['z']:.3f}): {len(downward_frames)}")
        print(f"  Frame range: [{downward_frames[0]}, {downward_frames[-1]}]")
        
        # Check if they form a continuous segment
        gaps = np.diff(downward_frames)
        if np.max(gaps) > 5 if len(gaps) > 0 else True:
            print(f"  WARNING: Downward motion is fragmented (max gap: {np.max(gaps) if len(gaps) > 0 else 0} frames)")
        
        # Find continuous segments
        segments = []
        start = downward_frames[0]
        for i, gap in enumerate(gaps):
            if gap > 5:
                segments.append((start, downward_frames[i]))
                start = downward_frames[i + 1]
        segments.append((start, downward_frames[-1]))
        
        print(f"  Continuous downward segments: {segments}")
        
        # Suggest threshold
        z_down_velocities = np.abs(z_vel[end_start:][z_end < 0])
        if len(z_down_velocities) > 0:
            suggested_z = np.percentile(z_down_velocities, 25)  # 25th percentile of downward velocities
            print(f"\n  Suggested z threshold to capture final motion: {suggested_z:.4f} m/s")
            print(f"  Current z threshold: {current_thresholds['z']:.4f} m/s")
            if suggested_z < current_thresholds['z']:
                print(f"  → Lower z threshold to {suggested_z:.4f} to detect final insertion")
    else:
        print(f"  No significant z↓ motion detected in final 40% of trajectory")
        print(f"  (threshold = {current_thresholds['z']:.3f} m/s)")
        
        # Check with lower threshold
        for test_thresh in [0.020, 0.015, 0.010]:
            test_frames = np.where(z_end < -test_thresh)[0]
            if len(test_frames) > 0:
                print(f"  With threshold {test_thresh:.3f}: {len(test_frames)} frames detected")
    
    return {
        'positions': positions,
        'velocities': velocities,
        'gripper': gripper,
        'intervals': intervals
    }


def main():
    parser = argparse.ArgumentParser(description='Optimize trajectory segmentation thresholds')
    parser.add_argument('--task', type=str, required=True,
                        help='Task name (can, square, tool_hang)')
    parser.add_argument('--num_episodes', type=int, default=20,
                        help='Number of episodes to analyze for optimization')
    parser.add_argument('--analyze_episode', type=int, default=None,
                        help='Analyze a specific episode in detail')
    parser.add_argument('--dataset_folder', type=str, default=None,
                        help='Path to dataset folder (default: <ROBOMIMIC_DATA_ROOT>/<task>/ph from config.py)')

    args = parser.parse_args()

    # Determine dataset folder
    if args.dataset_folder:
        dataset_folder = os.path.expanduser(args.dataset_folder)
    else:
        dataset_folder = os.path.join(_cfg.ROBOMIMIC_DATA_ROOT, args.task, "ph")
    
    if not os.path.exists(dataset_folder):
        print(f"Dataset folder not found: {dataset_folder}")
        sys.exit(1)
    
    if args.analyze_episode is not None:
        # Detailed analysis of single episode
        analyze_single_episode(dataset_folder, args.task, args.analyze_episode)
    else:
        # Optimize thresholds across multiple episodes
        results = analyze_task_thresholds(dataset_folder, args.task, args.num_episodes)
        
        # Save results
        output_file = os.path.join(
            os.path.dirname(__file__), '..', 'data', 'threshold_analysis',
            f'{args.task}_threshold_analysis.json'
        )
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Convert numpy types to Python types for JSON
        def convert_to_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_json(v) for v in obj]
            return obj
        
        with open(output_file, 'w') as f:
            json.dump(convert_to_json(results), f, indent=2)
        print(f"\nAnalysis saved to: {output_file}")


if __name__ == '__main__':
    main()

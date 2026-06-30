#!/usr/bin/env python3
"""
Window Size Analysis for Trajectory Segmentation.

This script analyzes RAW TRAJECTORY CHARACTERISTICS to recommend window sizes.
It does NOT use segmentation results, as those depend on the window size we're optimizing.

Approach:
1. Compute raw velocity from trajectory positions
2. Detect motion bursts (velocity above threshold)  
3. Analyze burst durations to find shortest meaningful motions
4. Recommend window size ≤ 2× shortest burst duration

Usage:
    python scripts/analyze_window_size.py --task square
    python scripts/analyze_window_size.py --task can --episodes "0,5,10"
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import h5py
from scipy.signal import savgol_filter

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.trajectory_utils import load_trajectory_from_hdf5
import config as _cfg


def load_episodes(task: str, episodes: List[int] = None) -> List[Tuple[int, np.ndarray]]:
    """Load trajectory positions for episodes."""
    hdf5_path = Path(_cfg.ROBOMIMIC_DATA_ROOT) / task / "ph" / "low_dim_abs.hdf5"
    
    if not hdf5_path.exists():
        print(f"ERROR: Dataset not found at {hdf5_path}")
        return []
    
    trajectories = []
    with h5py.File(hdf5_path, 'r') as f:
        # Get all episode indices
        demo_keys = [k for k in f['data'].keys() if k.startswith('demo_')]
        episode_indices = [int(k.split('_')[1]) for k in demo_keys]
        
        for ep_idx in episode_indices:
            if episodes is not None and ep_idx not in episodes:
                continue
            
            try:
                traj, _, _ = load_trajectory_from_hdf5(str(hdf5_path), ep_idx)
                positions = traj[:, :3]  # Extract EE position
                trajectories.append((ep_idx, positions))
            except Exception as e:
                print(f"WARNING: Failed to load episode {ep_idx}: {e}")
    
    return trajectories


def smooth_trajectory(positions: np.ndarray, window_length: int = 15, polyorder: int = 2) -> np.ndarray:
    """Smooth trajectory using Savitzky-Golay filter (same as trajectory_segmentation_v3.py)."""
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


def detect_motion_bursts(positions: np.ndarray, threshold: float, axis_idx: int, 
                        window_length: int = 15, min_duration: int = 3, 
                        sampling_interval: float = 0.2) -> List[Tuple[int, int]]:
    """
    Detect continuous motion bursts (velocity above threshold) for a specific axis.
    
    Returns list of (start, end) tuples.
    """
    # Compute velocity (frame-to-frame difference at 5 Hz = 0.2s per frame)
    velocities = np.diff(positions[:, axis_idx])  # (N-1,)
    velocities = np.append(velocities, velocities[-1])  # Pad to keep length N
    
    # Absolute velocity
    vel_abs = np.abs(velocities)
    
    # DEBUG: Print velocity stats
    print(f"    DEBUG: vel_abs min={vel_abs.min():.6f}, max={vel_abs.max():.6f}, mean={vel_abs.mean():.6f}, threshold={threshold}")
    print(f"    DEBUG: Frames above threshold: {(vel_abs > threshold).sum()}/{len(vel_abs)}")
    
    above_threshold = vel_abs > threshold
    
    # Find continuous runs
    changes = np.diff(np.concatenate([[False], above_threshold, [False]]).astype(int))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    
    bursts = []
    for start, end in zip(starts, ends):
        duration = end - start
        if duration >= min_duration:
            bursts.append((start, end))
    
    return bursts


def analyze_burst_durations(task: str, episodes: List[int] = None) -> Dict:
    """
    Analyze motion burst durations using RAW velocity (no segmentation).
    
    This is the GROUND TRUTH for determining window size - it doesn't
    depend on any threshold or window size parameters.
    """
    # Task-specific velocity thresholds (from TASK_THRESHOLDS in segmentation_v3.py)
    thresholds = {
        'tool_hang': {'x': 0.022, 'y': 0.046, 'z': 0.049},
        'square': {'x': 0.035, 'y': 0.015, 'z': 0.030},
        'can': {'x': 0.030, 'y': 0.070, 'z': 0.040},
    }
    
    trajectories = load_episodes(task, episodes)
    if not trajectories:
        return {}
    
    print(f"\nDEBUG: Loaded {len(trajectories)} episodes")
    
    all_bursts = []
    z_bursts = []  # Track z-axis separately (important for insertion)
    
    for ep_idx, positions in trajectories:
        print(f"\nDEBUG: Episode {ep_idx}, positions shape: {positions.shape}")
        
        # Detect bursts for each axis
        for axis_name, axis_idx in [('x', 0), ('y', 1), ('z', 2)]:
            threshold = thresholds[task][axis_name]
            
            bursts = detect_motion_bursts(positions, threshold, axis_idx, min_duration=3)
            
            print(f"  {axis_name}-axis (threshold={threshold}): {len(bursts)} bursts")
            
            for start, end in bursts:
                duration = end - start
                all_bursts.append(duration)
                if axis_name == 'z':
                    z_bursts.append(duration)
    
    print(f"\nDEBUG: Total bursts collected: {len(all_bursts)} (z-axis: {len(z_bursts)})")
    
    return {
        'all_bursts': {
            'count': len(all_bursts),
            'min': int(np.min(all_bursts)) if all_bursts else 0,
            'max': int(np.max(all_bursts)) if all_bursts else 0,
            'mean': float(np.mean(all_bursts)) if all_bursts else 0,
            'median': float(np.median(all_bursts)) if all_bursts else 0,
            'p5': float(np.percentile(all_bursts, 5)) if all_bursts else 0,
            'p10': float(np.percentile(all_bursts, 10)) if all_bursts else 0,
            'p25': float(np.percentile(all_bursts, 25)) if all_bursts else 0,
        },
        'z_bursts': {
            'count': len(z_bursts),
            'min': int(np.min(z_bursts)) if z_bursts else 0,
            'max': int(np.max(z_bursts)) if z_bursts else 0,
            'mean': float(np.mean(z_bursts)) if z_bursts else 0,
            'median': float(np.median(z_bursts)) if z_bursts else 0,
            'p5': float(np.percentile(z_bursts, 5)) if z_bursts else 0,
            'p10': float(np.percentile(z_bursts, 10)) if z_bursts else 0,
            'p25': float(np.percentile(z_bursts, 25)) if z_bursts else 0,
        }
    }


def print_burst_analysis(stats: Dict):
    """Print burst duration analysis."""
    print("\n" + "="*80)
    print("Motion Burst Duration Analysis (from RAW velocity)")
    print("="*80)
    
    print("\nAll Motion Bursts (any axis above threshold):")
    print(f"  Count:      {stats['all_bursts']['count']}")
    print(f"  Min:        {stats['all_bursts']['min']} frames")
    print(f"  5th %ile:   {stats['all_bursts']['p5']:.1f} frames  ← shortest 5% of motions")
    print(f"  10th %ile:  {stats['all_bursts']['p10']:.1f} frames  ← shortest 10% of motions")
    print(f"  25th %ile:  {stats['all_bursts']['p25']:.1f} frames")
    print(f"  Median:     {stats['all_bursts']['median']:.1f} frames")
    print(f"  Mean:       {stats['all_bursts']['mean']:.1f} frames")
    print(f"  Max:        {stats['all_bursts']['max']} frames")
    
    if stats['z_bursts']['count'] > 0:
        print("\nZ-axis Bursts (critical for insertion/release detection):")
        print(f"  Count:      {stats['z_bursts']['count']}")
        print(f"  Min:        {stats['z_bursts']['min']} frames")
        print(f"  5th %ile:   {stats['z_bursts']['p5']:.1f} frames")
        print(f"  10th %ile:  {stats['z_bursts']['p10']:.1f} frames")
        print(f"  25th %ile:  {stats['z_bursts']['p25']:.1f} frames")
        print(f"  Median:     {stats['z_bursts']['median']:.1f} frames")
        print(f"  Mean:       {stats['z_bursts']['mean']:.1f} frames")
        print(f"  Max:        {stats['z_bursts']['max']} frames")


def recommend_window_size(stats: Dict, task: str) -> Tuple[int, str]:
    """
    Recommend window size based on raw motion burst durations.
    
    Rule: Window should be ≤ 2× the 10th percentile of Z-axis bursts.
    This ensures we can detect even short insertion/release motions.
    """
    z_p10 = stats['z_bursts']['p10']
    all_p10 = stats['all_bursts']['p10']
    
    if z_p10 > 0:
        # Use Z-axis bursts (most critical for keypose detection)
        recommended = int(min(z_p10 * 2, 81))
        # Round to nearest odd number (Savgol filter requirement)
        if recommended % 2 == 0:
            recommended += 1
        
        reason = (f"Based on Z-axis motion bursts: p10={z_p10:.1f} frames\n"
                  f"       Window ≤ 2×p10 = {z_p10*2:.1f} → {recommended}\n"
                  f"       This ensures shortest 10% of Z motions can be detected")
    else:
        # Fallback to all bursts
        recommended = int(min(all_p10 * 2, 81))
        if recommended % 2 == 0:
            recommended += 1
        reason = (f"Based on all motion bursts: p10={all_p10:.1f} frames\n"
                  f"       Window ≤ 2×p10 = {all_p10*2:.1f} → {recommended}")
    
    return recommended, reason


def main():
    parser = argparse.ArgumentParser(
        description="Analyze trajectory motion bursts to determine optimal window size")
    parser.add_argument('--task', required=True, choices=['can', 'square', 'tool_hang'],
                        help='Task to analyze')
    parser.add_argument('--episodes', type=str,
                        help='Comma-separated episode numbers (default: all available)')
    
    args = parser.parse_args()
    
    # Parse episodes filter
    episodes_filter = None
    if args.episodes:
        episodes_filter = [int(x.strip()) for x in args.episodes.split(',')]
    
    print(f"\nAnalyzing task: {args.task}")
    if episodes_filter:
        print(f"Episodes: {episodes_filter}")
    else:
        print("Episodes: all available")
    
    # Analyze raw motion bursts
    stats = analyze_burst_durations(args.task, episodes_filter)
    if not stats:
        print("ERROR: No data found")
        return
    
    print_burst_analysis(stats)
    
    # Recommend window size
    recommended, reason = recommend_window_size(stats, args.task)
    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)
    print(f"Recommended window size: {recommended}")
    print(f"Reasoning:\n       {reason}")
    
    print("\nNOTE: This analysis uses RAW velocity bursts, independent of any")
    print("      segmentation parameters. The window size should allow detection")
    print("      of the shortest meaningful motions in your task.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Test V3 trajectory segmentation."""
import os
import sys
import numpy as np
import h5py
_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ)

from src.trajectory_segmentation_v3 import (
    segment_trajectory_v3,
    format_timelines_for_output,
    format_timelines_readable
)
import config as _cfg

def test_tool_hang():
    """Test V3 segmentation on tool_hang task."""
    hdf5_path = os.path.join(_cfg.ROBOMIMIC_DATA_ROOT, 'tool_hang', 'ph', 'low_dim.hdf5')
    task = 'tool_hang'
    episode_idx = 0
    
    with h5py.File(hdf5_path, 'r') as f:
        demo_key = f'data/demo_{episode_idx}'
        pos = f[f'{demo_key}/obs/robot0_eef_pos'][:]
        quat = f[f'{demo_key}/obs/robot0_eef_quat'][:]
        gripper = f[f'{demo_key}/obs/robot0_gripper_qpos'][:]
        actions = f[f'{demo_key}/actions'][:]
        action_gripper = actions[:, -1]
    
    # Build trajectory
    trajectory = np.hstack([pos, quat, gripper])
    print(f"Loaded {len(trajectory)} frames")
    print()
    
    # Segment trajectory
    timelines = segment_trajectory_v3(
        trajectory,
        action_gripper,
        task=task
    )
    
    # Print results
    print(f"Movement intervals: {len(timelines['movement'])}")
    for iv in timelines['movement']:
        print(f"  {iv.to_string()}")
    
    print(f"\nGripper intervals: {len(timelines['gripper'])}")
    for iv in timelines['gripper']:
        print(f"  {iv.to_string()}")
    
    # Test formatting
    formatted = format_timelines_for_output(timelines)
    readable = format_timelines_readable(timelines, episode=episode_idx)
    
    print("\n" + "=" * 60)
    print("Readable output:")
    print("=" * 60)
    print(readable)
    
    print("\n✓ Test passed!")


if __name__ == "__main__":
    test_tool_hang()

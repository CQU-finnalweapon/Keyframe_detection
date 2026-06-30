"""
Trajectory Utilities

Common functions for loading and processing robot trajectory data from HDF5 files.
"""

import h5py
import numpy as np
from typing import Tuple, Dict, Any, Optional


def load_trajectory_from_hdf5(
    hdf5_path: str, 
    episode_idx: int,
    demo_prefix: str = "demo",
    include_qpos: bool = False
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Load end-effector trajectory data from robomimic HDF5 dataset.
    
    This function loads END-EFFECTOR POSE (epos), NOT joint positions (qpos) by default.
    The 'robot0_eef_' prefix in robomimic datasets indicates end-effector data.
    
    Args:
        hdf5_path: Path to the HDF5 dataset
        episode_idx: Episode index (integer)
        demo_prefix: Prefix for demo keys (default: "demo")
        include_qpos: Whether to also load joint positions (default: False)
        
    Returns:
        trajectory: numpy array (T, 9) containing [pos(3), quat(4), gripper(2)]
                   - pos: end-effector position (x, y, z) in meters
                   - quat: orientation quaternion (w, x, y, z)
                   - gripper: gripper joint positions (left, right)
        action_gripper: numpy array (T,) gripper actions
                       - Values: -1 (open), 1 (close), 0 (none)
        qpos: numpy array (T, n_joints) joint positions, or None if include_qpos=False
              - For robosuite: typically (T, 7) for 7-DOF arm
    
    Note:
        For consistency, we call this "trajectory" but it specifically contains
        end-effector poses (epos), not joint configurations.
    
    Example:
        >>> trajectory, action_gripper, qpos = load_trajectory_from_hdf5(
        ...     "datasets/can/ph/low_dim_abs.hdf5", 
        ...     episode_idx=0,
        ...     include_qpos=True
        ... )
        >>> trajectory.shape
        (400, 9)  # 400 timesteps, 9D pose
        >>> action_gripper.shape
        (400,)    # 400 timesteps
        >>> qpos.shape
        (400, 7)  # 400 timesteps, 7 joints
    """
    with h5py.File(hdf5_path, 'r') as f:
        # Handle both integer episode index and string demo key
        if isinstance(episode_idx, str):
            demo_key = episode_idx
        else:
            demo_key = f'{demo_prefix}_{episode_idx}'
        
        if f'data/{demo_key}' not in f:
            available_demos = list(f['data'].keys())
            raise ValueError(
                f"Episode {demo_key} not found in {hdf5_path}. "
                f"Available demos: {available_demos[:10]}..."
            )
        
        demo = f[f'data/{demo_key}']
        
        # Load end-effector pose components
        robot_pos = demo['obs/robot0_eef_pos'][:]      # (T, 3)
        robot_quat = demo['obs/robot0_eef_quat'][:]    # (T, 4)
        gripper_qpos = demo['obs/robot0_gripper_qpos'][:]  # (T, 2)
        
        # Load gripper actions (last dimension of action vector)
        actions = demo['actions'][:]  # (T, action_dim)
        action_gripper = actions[:, -1]  # (T,)
        
        # Concatenate into single trajectory: [pos, quat, gripper]
        trajectory = np.concatenate([robot_pos, robot_quat, gripper_qpos], axis=1)
        
        # Load joint positions (qpos) if requested
        qpos = None
        if include_qpos and 'obs/robot0_joint_pos' in demo:
            qpos = demo['obs/robot0_joint_pos'][:]  # (T, n_joints)
        
    return trajectory, action_gripper, qpos


def load_full_observation(
    hdf5_path: str, 
    episode_idx: int, 
    demo_prefix: str = "demo",
    load_images: bool = True
) -> Dict[str, Any]:
    """
    Load full observation data including images, joint states, and actions.
    
    Useful for training that needs images or joint positions beyond just
    end-effector pose.
    
    Args:
        hdf5_path: Path to the HDF5 dataset
        episode_idx: Episode index
        demo_prefix: Prefix for demo keys (default: "demo")
        load_images: Whether to load image data (can be memory intensive)
    
    Returns:
        dict with keys:
            - 'trajectory': (T, 9) end-effector trajectory
            - 'action_gripper': (T,) gripper actions
            - 'actions': (T, action_dim) full action vector
            - 'images': dict of camera_name -> (T, H, W, C) images (if load_images=True)
            - 'qpos': (T, n_joints) joint positions (if available in dataset)
            - 'episode_metadata': dict with task info, success flag, etc.
    
    Example:
        >>> data = load_full_observation("datasets/can/ph/low_dim_abs.hdf5", 0)
        >>> data['trajectory'].shape
        (400, 9)
        >>> data['images']['agentview'].shape
        (400, 84, 84, 3)
        >>> data['qpos'].shape if data['qpos'] is not None else None
        (400, 7)  # 7 joint angles
    """
    with h5py.File(hdf5_path, 'r') as f:
        demo_key = f'{demo_prefix}_{episode_idx}'
        demo = f[f'data/{demo_key}']
        
        # Load trajectory using unified function
        trajectory, action_gripper, qpos_data = load_trajectory_from_hdf5(
            hdf5_path, episode_idx, demo_prefix, include_qpos=True
        )
        
        # Load full actions
        actions = demo['actions'][:]
        
        # Load images (if requested and available)
        images = {}
        if load_images and 'obs/images' in demo:
            for cam_name in demo['obs/images'].keys():
                images[cam_name] = demo['obs/images'][cam_name][:].astype(np.uint8)
        
        # Load episode metadata
        metadata = {}
        if 'attr' in demo:
            for key in demo['attr'].keys():
                metadata[key] = demo['attr'][key][()]
        
        return {
            'trajectory': trajectory,
            'action_gripper': action_gripper,
            'actions': actions,
            'images': images,
            'qpos': qpos_data,
            'episode_metadata': metadata
        }


def get_trajectory_statistics(trajectory: np.ndarray) -> Dict[str, Any]:
    """
    Compute statistics of a trajectory for analysis and debugging.
    
    Args:
        trajectory: (T, 9) array [pos(3), quat(4), gripper(2)]
    
    Returns:
        dict with statistics:
            - position_range: (min, max) for each position dimension
            - position_travel: total distance traveled
            - gripper_range: (min, max) for gripper opening
            - num_timesteps: T
    
    Example:
        >>> stats = get_trajectory_statistics(trajectory)
        >>> print(f"Total distance: {stats['position_travel']:.2f}m")
        Total distance: 1.23m
    """
    position = trajectory[:, :3]
    gripper = trajectory[:, 7:9]
    
    # Position statistics
    pos_min = position.min(axis=0)
    pos_max = position.max(axis=0)
    
    # Compute total distance traveled
    position_diffs = np.linalg.norm(position[1:] - position[:-1], axis=1)
    total_distance = position_diffs.sum()
    
    # Gripper statistics
    gripper_distance = gripper[:, 0] - gripper[:, 1]  # Opening measure
    gripper_min = gripper_distance.min()
    gripper_max = gripper_distance.max()
    
    return {
        'position_range': {
            'x': (float(pos_min[0]), float(pos_max[0])),
            'y': (float(pos_min[1]), float(pos_max[1])),
            'z': (float(pos_min[2]), float(pos_max[2]))
        },
        'position_travel': float(total_distance),
        'gripper_range': (float(gripper_min), float(gripper_max)),
        'num_timesteps': len(trajectory)
    }

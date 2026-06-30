
import h5py
import numpy as np
import os
import sys
import click
from tqdm import tqdm
import json
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from src.trajectory_segmentation_v3 import (
    segment_trajectory_v3,
    format_timelines_for_output,
    parse_trajectory_intervals_v3
)
from src.vl_parser import load_vl_json, parse_vl_events
from src.event_corrector_v3 import correct_events_v3
from src.trajectory_utils import load_trajectory_from_hdf5
from config import get_task_paths, get_relative_paths


def extract_keyposes_from_events(trajectory, qpos, corrected_events, key_event_types=['grasp', 'release', 'attach_hang']):
    """
    Extract keypose states from trajectory at corrected event timestamps.
    
    Args:
        trajectory: numpy array (T, 9) containing [pos(3), quat(4), gripper(2)]
                   This is end-effector pose (epos), not joint state (qpos)
        qpos: numpy array (T, n_joints) containing joint positions, or None
        corrected_events: list of corrected VL events with frame indices
        key_event_types: list of event types to extract as keyposes
        
    Returns:
        dict with keys:
            - 'frames': Frame indices of keyposes
            - 'epos': End-effector poses at keyposes (K, 9) [pos, quat, gripper]
            - 'qpos': Joint positions at keyposes (K, n_joints), or None if not available
            - 'event_types': Event type labels for each keypose
    """
    keypose_frames = []
    keypose_epos = []  # End-effector pose
    keypose_qpos = []  # Joint positions
    keypose_types = []
    
    for event in corrected_events:
        event_type = event['type']
        if event_type in key_event_types:
            frame_idx = event['corrected_frame_idx']
            
            # Ensure frame index is within bounds
            if 0 <= frame_idx < len(trajectory):
                keypose_frames.append(frame_idx)
                keypose_epos.append(trajectory[frame_idx])
                if qpos is not None:
                    keypose_qpos.append(qpos[frame_idx])
                keypose_types.append(event_type)
    
    result = {
        'frames': np.array(keypose_frames),
        'epos': np.array(keypose_epos),  # End-effector pose (9D)
        'event_types': keypose_types
    }
    
    if qpos is not None and len(keypose_qpos) > 0:
        result['qpos'] = np.array(keypose_qpos)  # Joint positions (K, n_joints)
    
    return result



def process_single_episode(hdf5_path, vl_json_path, episode_idx, 
                          position_threshold=0.005, rotation_threshold=0.01,
                          gripper_threshold=0.0001, task="square", corrector_version="clean"):
    """
    Process a single episode: load trajectory, correct VL events, extract keyposes.
    
    Args:
        hdf5_path: Path to robomimic HDF5 dataset
        vl_json_path: Path to VL model JSON output
        episode_idx: Episode index
        position_threshold: Threshold for position movement detection
        rotation_threshold: Threshold for rotation movement detection
        gripper_threshold: Threshold for gripper action detection
        task: Task name for task-specific correction strategies ('square', 'tool_hang', 'can')
        corrector_version: Not used (V3 only) - kept for API compatibility
        
    Returns:
        dict containing trajectory, corrected events, and keyposes
    """
    print(f"\n[Episode {episode_idx}] Loading trajectory...")
    trajectory, action_gripper, qpos = load_trajectory_from_hdf5(hdf5_path, episode_idx, include_qpos=True)
    
    print(f"[Episode {episode_idx}] Segmenting trajectory (V3 - task: {task})...")
    # Use V3 segmentation
    timelines = segment_trajectory_v3(
        trajectory,
        action_gripper,
        task=task,
        gripper_threshold=gripper_threshold
    )
    formatted = format_timelines_for_output(timelines)
    trajectory_intervals = parse_trajectory_intervals_v3(formatted)
    
    print(f"[Episode {episode_idx}] Loading VL events...")
    vl_data = load_vl_json(vl_json_path)
    vl_events = parse_vl_events(vl_data)
    
    print(f"[Episode {episode_idx}] Correcting event timestamps (V3)...")
    
    # Use V3 corrector with improved segmentation
    positions = trajectory[:, :3]  # Extract positions (x, y, z)
    quaternions = trajectory[:, 3:7]  # Extract quaternions (qx, qy, qz, qw)
    
    corrected_events = correct_events_v3(
        vl_events,
        timelines,  # Use Interval objects directly
        positions,
        quaternions,
        task=task,
        frame_offset=0
    )
    
    print(f"[Episode {episode_idx}] Extracting keyposes...")
    keyposes = extract_keyposes_from_events(
        trajectory,
        qpos,
        corrected_events,
        key_event_types=['grasp', 'release', 'attach_hang']
    )
    
    return {
        'trajectory': trajectory,
        'qpos': qpos,
        'corrected_events': corrected_events,
        'keyposes': keyposes,
        'trajectory_intervals': trajectory_intervals
    }



def process_single_episode_auto(hdf5_path, episode_idx,
                                gripper_threshold=0.0001, task="square"):
    """
    Process a single episode in auto mode: no VL JSON needed.
    
    Generates pseudo-events from trajectory segmentation (gripper open/close
    events) and corrects them using trajectory intervals.
    
    Args:
        hdf5_path: Path to robomimic HDF5 dataset (must be low_dim_abs.hdf5)
        episode_idx: Episode index
        gripper_threshold: Threshold for gripper action detection
        task: Task name for task-specific correction strategies
        
    Returns:
        dict containing trajectory, corrected events, and keyposes
    """
    print(f"\n[Episode {episode_idx}] Loading trajectory...")
    trajectory, action_gripper, qpos = load_trajectory_from_hdf5(hdf5_path, episode_idx, include_qpos=True)
    
    print(f"[Episode {episode_idx}] Segmenting trajectory (V3 - task: {task})...")
    timelines = segment_trajectory_v3(
        trajectory=trajectory,
        action_gripper=action_gripper,
        task=task,
        gripper_threshold=gripper_threshold
    )
    formatted = format_timelines_for_output(timelines)
    trajectory_intervals = parse_trajectory_intervals_v3(formatted)
    
    # Generate pseudo-events from gripper timeline
    pseudo_events = []
    gripper_timeline = timelines['gripper']
    
    for interval in gripper_timeline:
        if interval.label == 'closing':
            pseudo_events.append({
                'type': 'grasp',
                'frame_idx': interval.start - 1
            })
        elif interval.label == 'opening':
            pseudo_events.append({
                'type': 'release',
                'frame_idx': interval.start - 1
            })
    
    # For tool_hang, add attach_hang pseudo-event at the last release
    if task == 'tool_hang' and pseudo_events:
        release_events = [e for e in pseudo_events if e['type'] == 'release']
        if release_events:
            last_release_idx = release_events[-1]['frame_idx']
            pseudo_events.append({
                'type': 'attach_hang',
                'frame_idx': last_release_idx
            })
    
    print(f"[Episode {episode_idx}] Correcting events (V3)...")
    positions = trajectory[:, :3]
    quaternions = trajectory[:, 3:7]
    
    corrected_events = correct_events_v3(
        pseudo_events,
        timelines,
        positions,
        quaternions,
        task=task,
        frame_offset=0
    )
    
    print(f"[Episode {episode_idx}] Extracting keyposes...")
    keyposes = extract_keyposes_from_events(
        trajectory,
        qpos,
        corrected_events,
        key_event_types=['grasp', 'release', 'attach_hang']
    )
    
    return {
        'trajectory': trajectory,
        'qpos': qpos,
        'corrected_events': corrected_events,
        'keyposes': keyposes,
        'trajectory_intervals': trajectory_intervals
    }



def generate_keypose_training_data(hdf5_path, keyposes, episode_idx):
    """
    Generate training dataset with keypose observations and targets.
    
    For each timestep t:
    - obs: current state + previous keypose
    - target: next keypose + event type
    
    Args:
        hdf5_path: Path to robomimic HDF5 dataset
        keyposes: dict with 'frames', 'qpos', 'event_types'
        episode_idx: Episode index
        
    Returns:
        dict containing training data
    """
    with h5py.File(hdf5_path, 'r') as f:
        demo_key = f'data/demo_{episode_idx}'
        
        # Load full trajectory
        robot_pos = f[f'{demo_key}/obs/robot0_eef_pos'][:]
        robot_quat = f[f'{demo_key}/obs/robot0_eef_quat'][:]
        gripper_qpos = f[f'{demo_key}/obs/robot0_gripper_qpos'][:]
        
        # Load environment observations (object state)
        object_obs = None
        if f'{demo_key}/obs/object' in f:
            object_obs = f[f'{demo_key}/obs/object'][:]  # (T, 14) - object state including grasp success
        
        # Load joint positions (qpos)
        qpos_data = None
        if f'{demo_key}/obs/robot0_joint_pos' in f:
            qpos_data = f[f'{demo_key}/obs/robot0_joint_pos'][:]  # (T, n_joints)
        
        # Load images (if available)
        images = {}
        if f'obs' in f[demo_key] and 'images' in f[f'{demo_key}/obs']:
            for cam_name in f[f'{demo_key}/obs/images'].keys():
                images[cam_name] = f[f'{demo_key}/obs/images/{cam_name}'][:].astype(np.uint8)
        
        # Load actions
        actions = f[f'{demo_key}/actions'][:]
        
        horizon = len(robot_pos)
    
    # Combine into full end-effector trajectory (epos = end-effector pose)
    full_epos = np.concatenate([robot_pos, robot_quat, gripper_qpos], axis=1)
    
    # Note: env_obs (concatenated observation) is no longer computed here.
    # The dataset loader uses separate obs keys (object, robot0_eef_pos, etc.)
    # and concatenates them via _data_to_obs() with rotation conversion.
    
    # Initialize training data arrays
    obs_epos = full_epos  # (T, 9) - current end-effector pose
    obs_prev_keypose = np.zeros((horizon, 9))  # Previous keypose
    target_next_keypose = np.zeros((horizon, 9))  # Next keypose
    target_event_type = np.zeros(horizon, dtype=int)  # Event type (0=none, 1=grasp, 2=release, 3=attach_hang)
    # stage_index: which keypose stage is being pursued at each timestep.
    # 0 = before first keypose, 1 = heading to keypose[0], 2 = heading to keypose[1], ...
    # K = after last keypose (task complete). Range [0, K] where K = num_keyposes.
    target_stage_index = np.zeros(horizon, dtype=np.int32)
    
    event_type_map = {'grasp': 1, 'release': 2, 'attach_hang': 3}
    
    # Fill keypose data for each timestep
    kp_frames = keyposes['frames']
    kp_epos = keyposes['epos']  # End-effector poses at keyposes
    kp_types = keyposes['event_types']
    num_kp = len(kp_frames)
    
    for t in range(horizon):
        # Find previous keypose (most recent keypose before t)
        prev_kp_idx = np.where(kp_frames <= t)[0]
        if len(prev_kp_idx) > 0:
            obs_prev_keypose[t] = kp_epos[prev_kp_idx[-1]]
        else:
            obs_prev_keypose[t] = full_epos[0]  # Use initial state
        
        # Find next keypose (next keypose after t)
        next_kp_idx = np.where(kp_frames > t)[0]
        if len(next_kp_idx) > 0:
            idx = next_kp_idx[0]
            target_next_keypose[t] = kp_epos[idx]
            target_event_type[t] = event_type_map.get(kp_types[idx], 0)
            # Stage index = how many keyposes have been completed before this step
            # (= index of the next keypose to reach)
            target_stage_index[t] = idx
        else:
            target_next_keypose[t] = full_epos[-1]  # Use final state
            target_event_type[t] = 0
            # All keyposes completed: stage_index = num_kp
            target_stage_index[t] = num_kp
    
    training_data = {
        'obs': {
            'epos': obs_epos,  # (T, 9) - end-effector pose [pos, quat, gripper]
            # Separate obs keys (matches raw robomimic format, used by _data_to_obs)
            'object': object_obs if object_obs is not None else np.zeros((horizon, 14), dtype=np.float32),
            'robot0_eef_pos': robot_pos,     # (T, 3)
            'robot0_eef_quat': robot_quat,   # (T, 4)
            'robot0_gripper_qpos': gripper_qpos,  # (T, 2)
            'qpos': qpos_data,  # (T, n_joints) - joint positions
            'prev_keypose': obs_prev_keypose,  # (T, 9)
            'images': images  # dict of {cam_name: (T, H, W, C)}
        },
        'target': {
            'next_keypose': target_next_keypose,  # (T, 9)
            'event_type': target_event_type,  # (T,)
            # stage_index: which keypose is being pursued at each timestep.
            # Range [0, K] where K = num_keyposes.
            # 0 means heading to keypose[0], 1 means heading to keypose[1], etc.
            # K means all keyposes completed.
            'stage_index': target_stage_index,  # (T,) int32
            'keypose_frames': kp_frames,  # (num_kp,)
            'keypose_epos': kp_epos,  # (num_kp, 9) - end-effector poses
            'keypose_types': kp_types  # list of str
        },
        'actions': actions  # (T, action_dim)
    }
    
    # Add qpos data if available
    if 'qpos' in keyposes and keyposes['qpos'] is not None:
        training_data['target']['keypose_qpos'] = keyposes['qpos']  # (num_kp, n_joints)
    
    return training_data



def save_keypose_dataset(output_path, training_data):
    """
    Save keypose training dataset to HDF5 file.
    
    Args:
        output_path: Path to save HDF5 file
        training_data: dict containing training data
    """
    with h5py.File(output_path, 'w') as f:
        # Observation data
        obs_grp = f.create_group('obs')
        obs_grp.create_dataset('epos', data=training_data['obs']['epos'])  # End-effector pose (9D: pos+quat+gripper)
        # Separate obs keys (matches raw robomimic format, used by _data_to_obs)
        obs_grp.create_dataset('object', data=training_data['obs']['object'])  # Object state
        obs_grp.create_dataset('robot0_eef_pos', data=training_data['obs']['robot0_eef_pos'])  # (T, 3)
        obs_grp.create_dataset('robot0_eef_quat', data=training_data['obs']['robot0_eef_quat'])  # (T, 4)
        obs_grp.create_dataset('robot0_gripper_qpos', data=training_data['obs']['robot0_gripper_qpos'])  # (T, 2)
        if training_data['obs']['qpos'] is not None:
            obs_grp.create_dataset('qpos', data=training_data['obs']['qpos'])  # Joint positions (optional, for joint-space robots)
        obs_grp.create_dataset('prev_keypose', data=training_data['obs']['prev_keypose'])
        
        # Images (if available)
        if training_data['obs']['images']:
            img_grp = obs_grp.create_group('images')
            for cam_name, img_data in training_data['obs']['images'].items():
                img_grp.create_dataset(cam_name, data=img_data, compression='gzip')
        
        # Target data
        tgt_grp = f.create_group('target')
        tgt_grp.create_dataset('next_keypose', data=training_data['target']['next_keypose'])  # (T, 9) - epos target
        tgt_grp.create_dataset('event_type', data=training_data['target']['event_type'])  # (T,) - 0=none, 1=grasp, 2=release, 3=attach_hang
        tgt_grp.create_dataset('stage_index', data=training_data['target']['stage_index'])  # (T,) int32 - which keypose stage is active
        tgt_grp.create_dataset('keypose_frames', data=training_data['target']['keypose_frames'])  # (K,) - frame indices of keyposes
        tgt_grp.create_dataset('keypose_epos', data=training_data['target']['keypose_epos'])  # (K, 9) - epos at keyposes
        
        # Save joint positions (qpos) if available (optional, for joint-space robots)
        if 'keypose_qpos' in training_data['target']:
            tgt_grp.create_dataset('keypose_qpos', data=training_data['target']['keypose_qpos'])  # (K, n_joints)
        
        # Store event types as strings
        dt = h5py.string_dtype(encoding='utf-8')
        tgt_grp.create_dataset('keypose_types', data=training_data['target']['keypose_types'], dtype=dt)
        
        # Actions
        f.create_dataset('actions', data=training_data['actions'])
    
    print(f"  Saved keypose dataset to {output_path}")






@click.command()
@click.option('--hdf5_path', type=str, required=True, help='Path to robomimic HDF5 dataset (must be low_dim_abs.hdf5 for absolute actions)')
@click.option('--vl_json_dir', type=str, default=None, help='Directory containing VL JSON outputs (required unless --auto)')
@click.option('--output_dir', type=str, required=True, help='Directory to save keypose training datasets')
@click.option('--num_episodes', type=int, default=1, help='Number of episodes to process from 0 to N-1')
@click.option('--episode_list', type=str, default=None, help='Comma-separated list of episode indices (e.g., "5,7,9")')
@click.option('--task', type=str, default='square', help='Task name (square, tool_hang, can) for task-specific correction')
@click.option('--auto', 'auto_mode', is_flag=True, default=False, help='Auto mode: generate pseudo-events from trajectory segmentation (no VL JSON needed)')
@click.option('--position_threshold', type=float, default=0.005, help='Position movement threshold (meters)')
@click.option('--rotation_threshold', type=float, default=0.01, help='Rotation movement threshold (radians)')
@click.option('--gripper_threshold', type=float, default=0.0001, help='Gripper change threshold')
def main(hdf5_path, vl_json_dir, output_dir, num_episodes, episode_list, task,
         auto_mode, position_threshold, rotation_threshold, gripper_threshold):
    """
    Generate keypose training dataset from robomimic trajectories.
    
    Two modes:
      VL mode (default): Uses VL-corrected events from Steps 1-2 of the pipeline.
      Auto mode (--auto): Generates pseudo-events directly from trajectory
        segmentation (gripper open/close), no VL JSON needed.
    
    IMPORTANT: --hdf5_path must point to low_dim_abs.hdf5 (absolute actions),
    NOT low_dim.hdf5 (delta actions). The trajectory policy expects absolute
    actions which get converted to rotation_6d by the dataset loader.
    
    This script:
    1. Loads trajectory from robomimic HDF5 dataset
    2. Segments trajectory into movement/gripper intervals
    3. Corrects event timestamps using trajectory analysis
    4. Extracts keyposes at key events (grasp, release, attach_hang)
    5. Generates training dataset with obs, keyposes, and actions
    """
    
    # Expand user paths
    hdf5_path = os.path.expanduser(hdf5_path)
    if vl_json_dir:
        vl_json_dir = os.path.expanduser(vl_json_dir)
    output_dir = os.path.expanduser(output_dir)
    
    # Validate mode
    if not auto_mode and not vl_json_dir:
        raise click.UsageError("Either --vl_json_dir or --auto is required.")
    
    mode_str = "Auto (trajectory segmentation)" if auto_mode else "VL (vision-language events)"
    
    print("\n" + "="*80)
    print("Keypose Training Dataset Generation")
    print("="*80)
    print(f"Mode: {mode_str}")
    print(f"HDF5 Dataset: {hdf5_path}")
    if not auto_mode:
        print(f"VL JSON Directory: {vl_json_dir}")
    print(f"Output Directory: {output_dir}")
    
    # Determine which episodes to process
    if episode_list is not None:
        episodes_to_process = [int(ep.strip()) for ep in episode_list.split(',')]
        print(f"Episodes: {episode_list} (specific episodes)")
    else:
        episodes_to_process = list(range(num_episodes))
        print(f"Episodes: 0-{num_episodes-1}")
    
    print(f"Task: {task}")
    print("="*80 + "\n")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each episode
    for episode_idx in tqdm(episodes_to_process, desc=f"Processing {task}"):
        try:
            if auto_mode:
                # Auto mode: generate pseudo-events from trajectory segmentation
                result = process_single_episode_auto(
                    hdf5_path, episode_idx,
                    gripper_threshold=gripper_threshold,
                    task=task
                )
            else:
                # VL mode: load VL JSON and correct events
                # Find VL JSON file for this episode in subfolder structure
                # Structure: vl_json_dir/000_019/episode_N/corrected_events.json
                range_start = (episode_idx // 20) * 20
                range_end = range_start + 19
                range_folder = f"{range_start:03d}_{range_end:03d}"
                episode_folder = f"episode_{episode_idx}"
                vl_json_path = os.path.join(vl_json_dir, range_folder, episode_folder, 'corrected_events.json')
                
                if not os.path.exists(vl_json_path):
                    print(f"  [Episode {episode_idx}] WARNING: No VL JSON found at {vl_json_path}, skipping")
                    continue
                
                result = process_single_episode(
                    hdf5_path, vl_json_path, episode_idx,
                    position_threshold, rotation_threshold, gripper_threshold,
                    task=task
                )
            
            # Generate training data
            training_data = generate_keypose_training_data(
                hdf5_path,
                result['keyposes'],
                episode_idx
            )
            
            # Save to file
            output_path = os.path.join(output_dir, f'kp_episode_{episode_idx}.hdf5')
            save_keypose_dataset(output_path, training_data)
            
            # Print summary
            num_kp = len(result['keyposes']['frames'])
            if episode_idx % 50 == 0:
                print(f"  [Episode {episode_idx}] ✓ {num_kp} keyposes")
            
        except Exception as e:
            print(f"  [Episode {episode_idx}] ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "="*80)
    print("🎉 All episodes processed successfully!")
    print("="*80 + "\n")


def test_single_episode():
    """
    Test function to process a single episode using the new single-arm pipeline.
    """
    # Configuration
    episode_idx = 0
    task = 'can'  # Change to 'square' or 'tool_hang' for other tasks
    
    # Get paths from config
    paths = get_task_paths(task)
    rel_paths = get_relative_paths(task)
    
    hdf5_path = paths['hdf5_path']
    vl_json_path = os.path.join(paths['corrected_events_dir'], f'corrected_episode_{episode_idx}.json')
    output_dir = paths['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'kp_episode_{episode_idx}.hdf5')
    
    print("\n" + "="*80)
    print(f"Testing Single Episode: {task} - Episode {episode_idx}")
    print("="*80)
    print(f"HDF5: {hdf5_path}")
    print(f"VL JSON: {vl_json_path}")
    print(f"Output: {output_path}")
    print(f"Corrector: V3")
    print("="*80 + "\n")
    
    try:
        # Process episode: trajectory + VL → keyposes
        print("Processing episode...")
        result = process_single_episode(
            hdf5_path, vl_json_path, episode_idx,
            position_threshold=0.005,
            rotation_threshold=0.01,
            gripper_threshold=0.0001,
            task=task
        )
        
        # Generate training data
        print("\nGenerating training data...")
        training_data = generate_keypose_training_data(
            hdf5_path,
            result['keyposes'],
            episode_idx
        )
        
        # Save to file
        print(f"\nSaving to {output_path}...")
        save_keypose_dataset(output_path, training_data)
        
        # Print summary
        print("\n" + "="*80)
        print("✓ Test completed successfully!")
        print(f"  Keyposes extracted: {len(result['keyposes']['frames'])}")
        print(f"  Training samples: {len(training_data['obs']['epos'])}")
        print(f"  Event types: {', '.join(result['keyposes']['event_types'])}")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        print("\n")


if __name__ == "__main__":
    import sys
    
    # Check if running with command line arguments
    if len(sys.argv) > 1:
        # Run with click command line interface
        main()
    else:
        # Run test function with default parameters
        print("No command line arguments provided. Running test with default parameters...")
        test_single_episode()

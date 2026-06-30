#!/usr/bin/env python3
"""
Reorganize corrected events into episode-based folders and extract event frames.

New Structure:
data/corrected_events/
  {task}/
    000_019/              (subfolder for episodes 0-19)
      episode_0/
        events.json              (renamed from corrected_episode_0.json)
        intervals.json           (renamed from trajectory_intervals_episode_0.json)
        frames/
          1_grasp_frame_53.jpg
          2_detach_frame_61.jpg
          3_release_frame_106.jpg
          4_attach_drop_frame_118.jpg
        correction_flow.txt      (visualization of correction process)
"""

import os
import sys
import json
import shutil
import h5py
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import config as _cfg


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


def extract_events_from_json(corrected_json: Dict) -> List[Tuple[str, int]]:
    """Extract all event primitives and their frame indices from corrected JSON.
    
    Returns:
        List of (event_type, frame_index) tuples
    """
    events = []
    
    for subtask_key, subtask_data in corrected_json.items():
        if not subtask_key.startswith('subtask'):
            continue
            
        for phase in ['interaction', 'result']:
            if phase not in subtask_data:
                continue
                
            connections = subtask_data[phase].get('connections', [])
            for conn in connections:
                if len(conn) >= 3:
                    event_type = conn[1]  # 'grasp', 'release', etc.
                    frame_idx = conn[2]   # frame number
                    events.append((event_type, frame_idx))
    
    return sorted(events, key=lambda x: x[1])  # Sort by frame index


def extract_frame_from_hdf5(hdf5_path: str, demo_idx: int, frame_idx: int, 
                            camera: str = 'agentview_image') -> np.ndarray:
    """Extract a specific frame from HDF5 dataset.
    
    Args:
        hdf5_path: Path to image_abs.hdf5
        demo_idx: Demo index (0-199)
        frame_idx: Frame index within the demo
        camera: Camera view to extract ('agentview_image' or 'robot0_eye_in_hand_image')
    
    Returns:
        RGB image as numpy array (H, W, 3)
    """
    with h5py.File(hdf5_path, 'r') as f:
        demo_key = f'demo_{demo_idx}'
        if demo_key not in f['data']:
            raise ValueError(f"Demo {demo_key} not found in {hdf5_path}")
        
        obs = f['data'][demo_key]['obs']
        if camera not in obs:
            raise ValueError(f"Camera {camera} not found in demo {demo_key}")
        
        images = obs[camera]
        if frame_idx >= len(images):
            raise ValueError(f"Frame {frame_idx} out of range (max: {len(images)-1})")
        
        return images[frame_idx]


def create_correction_flow_text(corrected_json: Dict, intervals_json: Dict, 
                                episode_idx: int) -> str:
    """Create a text visualization showing the correction flow.
    
    Shows:
    - VL template (from prompt)
    - Trajectory intervals (moving/stable)
    - Final event assignment
    - Physical interpretation
    """
    lines = []
    lines.append("=" * 80)
    lines.append(f"CORRECTION FLOW VISUALIZATION - Episode {episode_idx}")
    lines.append("=" * 80)
    lines.append("")
    
    # Extract task info
    task = corrected_json.get('args', {}).get('task', 'unknown')
    lines.append(f"Task: {task}")
    lines.append("")
    
    # Show VL template structure (summarized)
    prompt = corrected_json.get('prompt', '')
    if 'Task:' in prompt:
        task_line = [line for line in prompt.split('\n') if 'Task:' in line][0].strip()
        lines.append("VL Template Input:")
        lines.append(f"  {task_line}")
    
    objects = corrected_json.get('args', {}).get('task', '')
    lines.append(f"  Objects: {objects}")
    lines.append("")
    
    # Show trajectory intervals summary
    lines.append("Trajectory Intervals Detected:")
    if intervals_json:
        # Check format - could be old subtask-based or new flat format
        if 'intervals' in intervals_json and isinstance(intervals_json['intervals'], list):
            # New flat format
            lines.append(f"  Total intervals: {len(intervals_json['intervals'])}")
            for interval_str in intervals_json['intervals']:
                lines.append(f"    {interval_str}")
        else:
            # Old subtask-based format
            for subtask_key in sorted(k for k in intervals_json.keys() if k.startswith('subtask')):
                lines.append(f"  {subtask_key}:")
                subtask = intervals_json[subtask_key]
                
                moving = subtask.get('moving_intervals', [])
                stable = subtask.get('stable_intervals', [])
                
                lines.append(f"    Moving: {len(moving)} intervals")
                for i, interval in enumerate(moving[:3]):  # Show first 3
                    lines.append(f"      [{interval['start_step']}, {interval['end_step']})")
                if len(moving) > 3:
                    lines.append(f"      ... ({len(moving) - 3} more)")
                
                lines.append(f"    Stable: {len(stable)} intervals")
                for i, interval in enumerate(stable[:3]):
                    lines.append(f"      [{interval['start_step']}, {interval['end_step']})")
                if len(stable) > 3:
                    lines.append(f"      ... ({len(stable) - 3} more)")
    lines.append("")
    
    # Show final event assignment with physical interpretation
    lines.append("Final Event Assignment:")
    lines.append("")
    
    for subtask_key in sorted(k for k in corrected_json.keys() if k.startswith('subtask')):
        subtask_data = corrected_json[subtask_key]
        lines.append(f"{subtask_key}:")
        
        # Targeting
        targeting = subtask_data.get('targeting', {})
        lines.append(f"  Targeting: frames {targeting.get('start_frame', '?')}-{targeting.get('end_frame', '?')}")
        lines.append(f"    → Robot moving to target position")
        
        # Interaction
        interaction = subtask_data.get('interaction', {})
        lines.append(f"  Interaction: frames {interaction.get('start_frame', '?')}-{interaction.get('end_frame', '?')}")
        
        connections = interaction.get('connections', [])
        for conn in connections:
            if len(conn) >= 3:
                edge, primitive, frame = conn[0], conn[1], conn[2]
                interpretation = get_physical_interpretation(primitive, edge)
                lines.append(f"    {primitive:<12} @ frame {frame:>3} → {interpretation}")
        
        # Result
        if 'result' in subtask_data:
            result = subtask_data['result']
            lines.append(f"  Result: frames {result.get('start_frame', '?')}-{result.get('end_frame', '?')}")
            
            connections = result.get('connections', [])
            for conn in connections:
                if len(conn) >= 3:
                    edge, primitive, frame = conn[0], conn[1], conn[2]
                    interpretation = get_physical_interpretation(primitive, edge)
                    lines.append(f"    {primitive:<12} @ frame {frame:>3} → {interpretation}")
        
        lines.append("")
    
    # Show chronological timeline
    all_events = extract_events_from_json(corrected_json)
    lines.append("Chronological Timeline:")
    for event_type, frame_idx in all_events:
        lines.append(f"  Frame {frame_idx:>3}: {event_type}")
    lines.append("")
    
    lines.append("=" * 80)
    return "\n".join(lines)


def get_physical_interpretation(primitive: str, edge: List[str]) -> str:
    """Get physical interpretation of an event primitive."""
    interpretations = {
        'grasp': f"Robot closes gripper on {edge[1]}",
        'detach': f"{edge[0]} lifts off from {edge[1]}",
        'release': f"Robot opens gripper, releasing {edge[1]}",
        'attach_hang': f"{edge[0]} contacts/inserts into {edge[1]}",
        'attach_drop': f"{edge[0]} lands/settles on {edge[1]}"
    }
    return interpretations.get(primitive, f"Unknown event: {primitive}")


def reorganize_task(task: str, corrected_events_dir: str, hdf5_path: str, 
                    output_dir: str, extract_frames: bool = True, camera: str = 'agentview_image',
                    video_base_dir: str = None):
    """Reorganize one task's corrected events into episode-based folders.
    
    Args:
        task: Task name ('can', 'square', 'tool_hang')
        corrected_events_dir: Path to data/corrected_events/{task}/
        hdf5_path: Path to image_abs.hdf5 for this task
        output_dir: Output directory for reorganized structure
        extract_frames: Whether to extract frames from HDF5
        camera: Camera view to use ('agentview_image' or 'sideview_image')
        video_base_dir: Base directory containing video files (e.g., .../datasets/{task}/ph/video/)
    """
    task_input_dir = Path(corrected_events_dir) / task
    task_output_dir = Path(output_dir) / task
    
    if not task_input_dir.exists():
        print(f"Warning: {task_input_dir} does not exist, skipping")
        return
    
    # Find all episode files (check both root and subfolders)
    corrected_files = []
    
    # Check subfolders first (new structure: 000_019, 020_039, etc.)
    for subfolder in sorted(task_input_dir.glob('[0-9][0-9][0-9]_[0-9][0-9][0-9]')):
        if subfolder.is_dir():
            # Check for newest structure: 000_019/episode_XXX/corrected_events.json
            for episode_folder in sorted(subfolder.glob('episode_*')):
                if episode_folder.is_dir():
                    episode_file = episode_folder / 'corrected_events.json'
                    if episode_file.exists():
                        corrected_files.append(episode_file)
            
            # Fallback to: episode_XXX_corrected_events.json (files directly in range folder)
            new_files = sorted(subfolder.glob('episode_*_corrected_events.json'))
            if new_files:
                corrected_files.extend(new_files)
            else:
                # Fallback to old naming: corrected_episode_XXX.json
                corrected_files.extend(sorted(subfolder.glob('corrected_episode_*.json')))
    
    # Fallback to root directory (old structure)
    if not corrected_files:
        # Try new naming first
        corrected_files = sorted(task_input_dir.glob('episode_*_corrected_events.json'))
        if not corrected_files:
            # Fallback to old naming
            corrected_files = sorted(task_input_dir.glob('corrected_episode_*.json'))
    
    print(f"\n{'='*80}")
    print(f"Processing task: {task}")
    print(f"Found {len(corrected_files)} episodes")
    print(f"{'='*80}")
    
    for corrected_file in corrected_files:
        # Also look for trajectory_intervals.txt in the same directory
        intervals_txt = corrected_file.parent / 'trajectory_intervals.txt'
        
        # Extract episode number from either naming convention
        parent_name = corrected_file.parent.name
        
        # Check if file is in episode_XX folder (newest structure)
        if parent_name.startswith('episode_'):
            episode_idx = int(parent_name.replace('episode_', ''))
        else:
            # Extract from filename
            stem = corrected_file.stem
            if '_corrected_events' in stem:
                # New format: episode_XXX_corrected_events
                episode_str = stem.replace('episode_', '').replace('_corrected_events', '')
            elif stem == 'corrected_events':
                # Newest format in episode folder: episode_XX/corrected_events.json
                # Already handled above
                continue
            else:
                # Old format: corrected_episode_XXX
                episode_str = stem.replace('corrected_episode_', '')
            episode_idx = int(episode_str)
        
        # Determine subfolder for output
        subfolder = get_episode_subfolder(episode_idx)
        
        # Create episode folder with subfolder
        episode_dir = task_output_dir / subfolder / f'episode_{episode_idx}'
        episode_dir.mkdir(parents=True, exist_ok=True)
        frames_dir = episode_dir / 'frames'
        frames_dir.mkdir(exist_ok=True)
        
        # Load corrected events
        with open(corrected_file, 'r') as f:
            corrected_json = json.load(f)
        
        # Load trajectory intervals (check multiple naming conventions)
        # 1. Newest: episode_XX/trajectory_intervals.json (same folder as corrected_events.json)
        intervals_file = corrected_file.parent / 'trajectory_intervals.json'
        if not intervals_file.exists():
            # 2. New: episode_XXX_trajectory_intervals.json (with prefix)
            intervals_file = corrected_file.parent / f'episode_{episode_idx}_trajectory_intervals.json'
        if not intervals_file.exists():
            # 3. Old: trajectory_intervals_episode_XXX.json
            intervals_file = corrected_file.parent / f'trajectory_intervals_episode_{episode_idx}.json'
        intervals_json = {}
        if intervals_file.exists():
            with open(intervals_file, 'r') as f:
                intervals_json = json.load(f)
        
        # Save renamed files
        with open(episode_dir / 'events.json', 'w') as f:
            json.dump(corrected_json, f, indent=2)
        
        if intervals_json:
            with open(episode_dir / 'intervals.json', 'w') as f:
                json.dump(intervals_json, f, indent=2)
        
        # Copy trajectory_intervals.txt if it exists
        if intervals_txt.exists():
            import shutil
            shutil.copy2(intervals_txt, episode_dir / 'trajectory_intervals.txt')
        
        # Copy video file if available
        if video_base_dir:
            subfolder = get_episode_subfolder(episode_idx)
            video_file = Path(video_base_dir) / subfolder / f'episode_{episode_idx}_video.mp4'
            if video_file.exists():
                import shutil
                shutil.copy2(video_file, episode_dir / f'episode_{episode_idx}_video.mp4')
        
        # Create correction flow visualization
        flow_text = create_correction_flow_text(corrected_json, intervals_json, episode_idx)
        with open(episode_dir / 'correction_flow.txt', 'w') as f:
            f.write(flow_text)
        
        # Extract event frames
        if extract_frames and os.path.exists(hdf5_path):
            events = extract_events_from_json(corrected_json)
            
            # Get max frame index from HDF5
            max_frame_idx = None
            try:
                with h5py.File(hdf5_path, 'r') as f:
                    demo_key = f'demo_{episode_idx}'
                    if demo_key in f['data']:
                        obs = f['data'][demo_key]['obs']
                        if camera in obs:
                            max_frame_idx = len(obs[camera]) - 1
            except Exception as e:
                print(f"  Warning: Could not determine max frame for episode {episode_idx}: {e}")
            
            for seq_id, (event_type, frame_idx) in enumerate(events, start=1):
                try:
                    # Handle frame beyond video length (typically last frame)
                    actual_frame_idx = frame_idx
                    if max_frame_idx is not None and frame_idx > max_frame_idx:
                        actual_frame_idx = max_frame_idx
                        print(f"  Note: {event_type} at frame {frame_idx} is beyond video "
                              f"(max: {max_frame_idx}), using last frame")
                    
                    frame = extract_frame_from_hdf5(hdf5_path, episode_idx, actual_frame_idx, camera=camera)
                    
                    # Save as JPG with sequence ID prefix
                    img = Image.fromarray(frame)
                    output_path = frames_dir / f'{seq_id}_{event_type}_frame_{frame_idx}.jpg'
                    img.save(output_path, quality=95)
                    
                except Exception as e:
                    print(f"  Warning: Failed to extract {event_type} frame {frame_idx} "
                          f"for episode {episode_idx}: {e}")
        
        if (episode_idx + 1) % 20 == 0:
            print(f"  Processed {episode_idx + 1} episodes...")
    
    print(f"✓ Completed task {task}: {len(corrected_files)} episodes reorganized")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Reorganize corrected events and extract event frames'
    )
    parser.add_argument(
        '--corrected-events-dir',
        default=_cfg.CORRECTED_EVENTS_DIR,
        help='Path to corrected_events directory'
    )
    parser.add_argument(
        '--data-dir',
        default=_cfg.ROBOMIMIC_DATA_ROOT,
        help='Path to robomimic datasets directory'
    )
    parser.add_argument(
        '--output-dir',
        default=_cfg.CORRECTED_EVENTS_ORGANIZED_DIR,
        help='Output directory for reorganized structure'
    )
    parser.add_argument(
        '--tasks',
        nargs='+',
        default=['can', 'square', 'tool_hang'],
        help='Tasks to process'
    )
    parser.add_argument(
        '--no-frames',
        action='store_true',
        help='Skip frame extraction (only reorganize JSON files)'
    )
    
    args = parser.parse_args()
    
    # Task name mapping and camera views
    task_to_dataset = {
        'can': 'can',
        'square': 'square',
        'tool_hang': 'tool_hang'
    }
    
    task_to_camera = {
        'can': 'agentview_image',
        'square': 'agentview_image',
        'tool_hang': 'sideview_image'
    }
    
    print("Starting reorganization and visualization...")
    print(f"Input: {args.corrected_events_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Tasks: {args.tasks}")
    print(f"Extract frames: {not args.no_frames}")
    
    for task in args.tasks:
        dataset_name = task_to_dataset.get(task, task)
        camera = task_to_camera.get(task, 'agentview_image')
        hdf5_path = os.path.join(args.data_dir, dataset_name, 'ph', 'image_abs.hdf5')
        video_base_dir = os.path.join(args.data_dir, dataset_name, 'ph', 'video')
        
        reorganize_task(
            task=task,
            corrected_events_dir=args.corrected_events_dir,
            hdf5_path=hdf5_path,
            output_dir=args.output_dir,
            extract_frames=not args.no_frames,
            camera=camera,
            video_base_dir=video_base_dir
        )
    
    print(f"\n{'='*80}")
    print("✓ All tasks completed!")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()

"""
Vision-Language (VL) Model Event Parser

This module parses JSON output from VL models (like Qwen3-VL) that analyze robot videos
and extract interaction events with timestamps.
"""

import json
from typing import Dict, List, Any, Optional


def load_vl_json(json_path: str) -> Dict[str, Any]:
    """
    Load VL model JSON output from file.
    
    Args:
        json_path: Path to JSON file
        
    Returns:
        Dictionary containing VL model output
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def parse_vl_events(vl_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Parse VL events into a chronological list with extracted event types.
    
    Args:
        vl_data: VL model JSON data (can be dict or string)
        
    Returns:
        List of parsed events with metadata
    """
    # Handle both dict and string input
    if isinstance(vl_data, str):
        vl_data = json.loads(vl_data)
    
    events = []
    
    # Handle both old format (ModeChangeDetection) and new format (subtasks)
    if "ModeChangeDetection" in vl_data:
        # Old format: flat list of interaction changes
        for event in vl_data["ModeChangeDetection"]:
            parsed_event = _parse_mode_change_event(event)
            events.append(parsed_event)
    elif "subtask1" in vl_data or "subtask2" in vl_data:
        # New format: hierarchical subtasks with targeting/interaction/result
        events = _parse_subtask_format(vl_data)
    else:
        raise ValueError("Unknown VL JSON format. Expected 'ModeChangeDetection' or 'subtaskN' keys.")
    
    # Ensure events are in chronological order
    events.sort(key=lambda x: x.get("frame_idx", x.get("start_time", 0)))
    
    return events


def _parse_mode_change_event(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse a single ModeChangeDetection event.
    
    Args:
        event: Raw event from VL model
        
    Returns:
        Parsed event with extracted metadata
    """
    # Extract event type from interaction_change string
    interaction_str = event.get("interaction_change", "")
    event_type = _extract_event_type(interaction_str)
    
    parsed_event = {
        "original": event.copy(),
        "type": event_type,
        "graph_number": event.get("graph_number"),
        "timestamp": event.get("timestep_orig", event.get("time", 0)),
        "frame_idx": event.get("frame_idx", 0),
        "interaction_change": interaction_str,
        "assigned_interval": None,  # Will be filled by corrector
        "corrected_frame_idx": None,
        "corrected_timestep": None
    }
    
    return parsed_event


def _parse_connection(connection: list, subtask_idx: int, phase: str) -> Optional[Dict[str, Any]]:
    """
    Parse a connection from subtask format into an event.
    
    Handles multiple formats:
    - Format A (old): [[obj1, obj2], primitive, time] (3 elements)
    - Format B (corrected): [[obj1, obj2], event_type, frame_idx, timestamp] (4 elements, first is list)
    - Format C (alternative): [obj1, obj2, event_type, time] (4 elements, first two are strings)
    
    Args:
        connection: Connection list from VL JSON
        subtask_idx: Index of the subtask
        phase: "interaction" or "result"
        
    Returns:
        Parsed event dict or None
    """
    if len(connection) == 3:
        # Format A: [[obj1, obj2], primitive, time]
        edge = connection[0]
        event_type = connection[1]
        frame_idx = connection[2]
        timestamp = None
    elif len(connection) == 4:
        # Check if first element is a list (Format B) or string (Format C)
        if isinstance(connection[0], list):
            # Format B: [[obj1, obj2], event_type, frame_idx, timestamp]
            edge = connection[0]
            event_type = connection[1]
            frame_idx = connection[2]
            timestamp = connection[3]
        else:
            # Format C: [obj1, obj2, event_type, time]
            edge = [connection[0], connection[1]]
            event_type = connection[2]
            frame_idx = connection[3]
            timestamp = None
    else:
        return None
    
    # Ensure frame_idx is an integer
    if isinstance(frame_idx, str):
        frame_idx = int(frame_idx)
    
    # Calculate timestamp if not provided (assuming 20fps for robomimic)
    if timestamp is None:
        timestamp = frame_idx / 20.0
    
    event = {
        "original": {
            "subtask": subtask_idx,
            "phase": phase,
            "edge": edge,
            "primitive": event_type,
            "time": timestamp
        },
        "type": event_type,  # NOW CORRECTLY SET TO EVENT TYPE NAME
        "subtask_number": subtask_idx,
        "timestamp": timestamp,
        "frame_idx": frame_idx,
        "interaction_change": f"[{edge}, {event_type}]",
        "assigned_interval": None,
        "corrected_frame_idx": None,
        "corrected_timestep": None
    }
    return event


def _parse_subtask_format(vl_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Parse new subtask-based format into events.
    
    Args:
        vl_data: VL data with subtask structure
        
    Returns:
        List of parsed events
    """
    events = []
    
    # Iterate through all subtasks
    subtask_idx = 1
    while f"subtask{subtask_idx}" in vl_data:
        subtask = vl_data[f"subtask{subtask_idx}"]
        
        # Extract events from interaction phase
        if "interaction" in subtask:
            interaction = subtask["interaction"]
            for connection in interaction.get("connections", []):
                event = _parse_connection(connection, subtask_idx, "interaction")
                if event:
                    events.append(event)
        
        # Extract events from result phase
        if "result" in subtask:
            result = subtask["result"]
            for connection in result.get("connections", []):
                event = _parse_connection(connection, subtask_idx, "result")
                if event:
                    events.append(event)
        
        subtask_idx += 1
    
    return events


def _extract_event_type(interaction_str: str) -> str:
    """
    Extract event type from interaction_change string.
    
    Args:
        interaction_str: Interaction change string like "[robot, obj], grasp, add"
        
    Returns:
        Event type (grasp, release, attach, detach, etc.)
    """
    # Handle variations in format
    interaction_str = interaction_str.strip()
    
    # Try to extract primitive (second element after splitting by comma)
    parts = [p.strip() for p in interaction_str.split(",")]
    
    if len(parts) >= 2:
        # Remove quotes and brackets
        event_type = parts[-2].strip("'\"[] ")
        return event_type
    
    # Fallback: check for known primitives in the string
    known_primitives = ["grasp", "release", "attach", "detach", "attach/intersect"]
    for primitive in known_primitives:
        if primitive in interaction_str.lower():
            return primitive
    
    return "unknown"


def save_corrected_vl_json(
    events: List[Dict[str, Any]],
    original_vl_data: Dict[str, Any],
    output_path: str,
    trajectory_fps: float = 20.0
) -> None:
    """
    Save corrected VL events to JSON file.
    Updates both connection timestamps and phase start_time/end_time.
    Includes both frame indices and timestamps in seconds.
    
    Args:
        events: List of corrected events
        original_vl_data: Original VL data structure
        output_path: Path to save corrected JSON
        trajectory_fps: FPS of the trajectory data (default: 20.0 for robomimic)
    """
    import copy
    # Deep copy to avoid modifying original
    corrected_data = copy.deepcopy(original_vl_data)
    
    # Helper to convert frame to seconds
    def frame_to_sec(frame_idx):
        return round(frame_idx / trajectory_fps, 3)
    
    if "ModeChangeDetection" in corrected_data:
        # Old format: update ModeChangeDetection list
        corrected_events = []
        for event in events:
            corrected_event = event["original"].copy()
            if event["corrected_frame_idx"] is not None:
                corrected_event["frame_idx"] = event["corrected_frame_idx"]
                corrected_event["time_sec"] = frame_to_sec(event["corrected_frame_idx"])
            if event["corrected_timestep"] is not None:
                corrected_event["timestep_orig"] = event["corrected_timestep"]
            corrected_events.append(corrected_event)
        
        corrected_data["ModeChangeDetection"] = corrected_events
    
    elif any(k.startswith("subtask") for k in corrected_data.keys()):
        # New format: update subtask connections and phase times
        # First, rename start_time/end_time to start_frame/end_frame in all phases
        subtask_idx = 1
        while f"subtask{subtask_idx}" in corrected_data:
            subtask_key = f"subtask{subtask_idx}"
            subtask = corrected_data[subtask_key]
            for phase_name in ["targeting", "interaction", "result"]:
                if phase_name in subtask:
                    phase = subtask[phase_name]
                    if "start_time" in phase:
                        phase["start_frame"] = phase.pop("start_time")
                    if "end_time" in phase:
                        phase["end_frame"] = phase.pop("end_time")
            subtask_idx += 1
        
        # Group events by subtask and phase
        events_by_subtask_phase = {}
        for event in events:
            subtask_num = event.get("subtask_number", 1)
            phase = event["original"].get("phase", "interaction")
            key = (subtask_num, phase)
            if key not in events_by_subtask_phase:
                events_by_subtask_phase[key] = []
            events_by_subtask_phase[key].append(event)
        
        # Update each subtask
        subtask_idx = 1
        prev_end_frame = 0
        
        while f"subtask{subtask_idx}" in corrected_data:
            subtask_key = f"subtask{subtask_idx}"
            subtask = corrected_data[subtask_key]
            
            for phase_name in ["targeting", "interaction", "result"]:
                if phase_name not in subtask:
                    continue
                    
                phase = subtask[phase_name]
                key = (subtask_idx, phase_name)
                
                # Update connection timestamps
                if key in events_by_subtask_phase:
                    phase_events = events_by_subtask_phase[key]
                    corrected_times = []
                    
                    for event in phase_events:
                        event_edge = event["original"]["edge"]
                        event_primitive = event["original"]["primitive"]
                        
                        for conn in phase.get("connections", []):
                            # Handle both connection formats
                            if isinstance(conn[0], list):
                                conn_edge = conn[0]
                                conn_primitive = conn[1]
                                time_idx = 2
                            else:
                                conn_edge = [conn[0], conn[1]]
                                conn_primitive = conn[2]
                                time_idx = 3
                            
                            if conn_edge == event_edge and conn_primitive == event_primitive:
                                if event["corrected_frame_idx"] is not None:
                                    conn[time_idx] = event["corrected_frame_idx"]
                                    corrected_times.append(event["corrected_frame_idx"])
                    
                    # Update phase start_frame and end_frame based on corrected event times
                    if corrected_times:
                        min_time = min(corrected_times)
                        max_time = max(corrected_times)
                        
                        # For phases with events, set times around the events
                        if phase_name == "targeting":
                            # targeting ends when interaction starts
                            phase["end_frame"] = min_time
                        elif phase_name == "interaction":
                            phase["start_frame"] = min_time
                            phase["end_frame"] = max_time
                        elif phase_name == "result":
                            phase["start_frame"] = min_time
                            phase["end_frame"] = max_time
                
                # Ensure phase continuity: this phase starts where previous ended
                if phase_name == "targeting":
                    phase["start_frame"] = prev_end_frame
                
                prev_end_frame = phase.get("end_frame", prev_end_frame)
            
            subtask_idx += 1
        
        # Second pass: ensure consecutive subtasks are continuous
        _ensure_subtask_continuity(corrected_data)
        
        # Third pass: add time_sec fields for all times
        _add_time_seconds(corrected_data, trajectory_fps)
    
    # Add metadata
    corrected_data["_correction_info"] = {
        "trajectory_fps": trajectory_fps,
        "note": "frame indices are trajectory frames at {:.0f}fps, time_sec is in seconds".format(trajectory_fps)
    }
    
    # Save to file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(corrected_data, f, indent=2)


def _ensure_subtask_continuity(corrected_data: Dict[str, Any]) -> None:
    """
    Ensure subtasks and phases are continuous (no gaps or overlaps).
    Each phase's end_frame should equal the next phase's start_frame.
    """
    subtask_idx = 1
    prev_end_frame = 0
    
    while f"subtask{subtask_idx}" in corrected_data:
        subtask_key = f"subtask{subtask_idx}"
        subtask = corrected_data[subtask_key]
        
        # Process phases in order
        for phase_name in ["targeting", "interaction", "result"]:
            if phase_name not in subtask:
                continue
            
            phase = subtask[phase_name]
            
            # Handle both old (start_time) and new (start_frame) naming
            start_key = "start_frame" if "start_frame" in phase else "start_time"
            end_key = "end_frame" if "end_frame" in phase else "end_time"
            
            # Set start to previous end for continuity
            phase[start_key] = prev_end_frame
            
            # Ensure end >= start
            if phase.get(end_key, 0) < phase[start_key]:
                phase[end_key] = phase[start_key]
            
            prev_end_frame = phase[end_key]
        
        subtask_idx += 1


def _add_time_seconds(corrected_data: Dict[str, Any], fps: float) -> None:
    """
    Add time_sec fields (in seconds) alongside frame indices and reorder fields.
    Final order: start_frame, end_frame, start_sec, end_sec, connections
    
    Args:
        corrected_data: The corrected VL data structure
        fps: Frames per second for conversion
    """
    def frame_to_sec(frame_idx):
        return round(frame_idx / fps, 3)
    
    subtask_idx = 1
    while f"subtask{subtask_idx}" in corrected_data:
        subtask_key = f"subtask{subtask_idx}"
        subtask = corrected_data[subtask_key]
        
        for phase_name in ["targeting", "interaction", "result"]:
            if phase_name not in subtask:
                continue
            
            phase = subtask[phase_name]
            
            # Rename start_time/end_time to start_frame/end_frame if they exist
            if "start_time" in phase:
                phase["start_frame"] = phase.pop("start_time")
            if "end_time" in phase:
                phase["end_frame"] = phase.pop("end_time")
            
            # Add time_sec for start_frame and end_frame
            start_frame = phase.get("start_frame", 0)
            end_frame = phase.get("end_frame", 0)
            start_sec = frame_to_sec(start_frame)
            end_sec = frame_to_sec(end_frame)
            
            # Get connections
            connections = phase.get("connections", [])
            
            # Add time_sec for each connection
            for conn in connections:
                if isinstance(conn[0], list):
                    # Format: [[obj1, obj2], primitive, frame, time_sec]
                    frame_idx = conn[2]
                    if len(conn) == 3:
                        conn.append(frame_to_sec(frame_idx))
                    else:
                        conn[3] = frame_to_sec(frame_idx)
                else:
                    # Format: [obj1, obj2, primitive, frame, time_sec]
                    frame_idx = conn[3]
                    if len(conn) == 4:
                        conn.append(frame_to_sec(frame_idx))
                    else:
                        conn[4] = frame_to_sec(frame_idx)
            
            # Rebuild phase with correct field order
            subtask[phase_name] = {
                "start_frame": start_frame,
                "end_frame": end_frame,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "connections": connections
            }
        
        subtask_idx += 1


def get_vl_fps(vl_data: Dict[str, Any]) -> float:
    """
    Extract FPS from VL data args if available.
    
    Args:
        vl_data: VL model output data
        
    Returns:
        FPS value (default: 5.0)
    """
    if "args" in vl_data and "target_fps" in vl_data["args"]:
        return float(vl_data["args"]["target_fps"])
    return 5.0  # Default FPS

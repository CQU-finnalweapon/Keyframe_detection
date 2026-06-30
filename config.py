"""
Configuration file for the keypose-labelling pipeline.

Paths are resolved relative to this file so the project is self-contained.
Override the dataset / data roots with environment variables if your robomimic
data or generated outputs live elsewhere:

    KEYPOSE_LABELLING_ROBOMIMIC_ROOT   robomimic datasets dir (default: shared manip store)
    KEYPOSE_LABELLING_DATA_ROOT        where generated labels/events are written (default: ./data)
"""
import os

# ============================================================================
# Base Directories  (resolved from this file — no machine-specific hard-coding)
# ============================================================================

# Keypose-labelling project root (the directory containing this config.py)
KEYPOSE_ROOT = os.path.dirname(os.path.abspath(__file__))

# ============================================================================
# Data Directories
# ============================================================================

# Robomimic dataset root.  Defaults to the shared manip dataset store
# (…/manip/datasets/robomimic); override with KEYPOSE_LABELLING_ROBOMIMIC_ROOT.
_default_robomimic = os.path.normpath(
    os.path.join(KEYPOSE_ROOT, "..", "..", "datasets", "robomimic")
)
ROBOMIMIC_DATA_ROOT = os.environ.get(
    "KEYPOSE_LABELLING_ROBOMIMIC_ROOT", _default_robomimic
)

# Keypose data root (all generated data goes here); override with
# KEYPOSE_LABELLING_DATA_ROOT.
KEYPOSE_DATA_ROOT = os.environ.get(
    "KEYPOSE_LABELLING_DATA_ROOT", os.path.join(KEYPOSE_ROOT, "data")
)

# VL model analysis output (raw JSON from VL model) - renamed from data_divide
DIVIDED_EVENTS_DIR = os.path.join(KEYPOSE_DATA_ROOT, "divided_events")

# Corrected events output (trajectory-corrected JSON)
CORRECTED_EVENTS_DIR = os.path.join(KEYPOSE_DATA_ROOT, "corrected_events")

# Organized corrected events (per-episode structure)
CORRECTED_EVENTS_ORGANIZED_DIR = os.path.join(KEYPOSE_DATA_ROOT, "corrected_events_organized")

# ============================================================================
# Task-Specific Paths
# ============================================================================

def get_task_paths(task_name):
    """
    Get all paths for a specific task.
    
    Args:
        task_name: Task name (e.g., 'can', 'square', 'tool_hang')
    
    Returns:
        Dictionary containing all relevant paths for the task
    """
    task_root = os.path.join(ROBOMIMIC_DATA_ROOT, task_name, "ph")
    
    # Map task names to VL JSON subdirectories
    # Note: folder names now match robomimic conventions (can, square, tool_hang)
    vl_json_task_map = {
        'can': 'can',
        'square': 'square',
        'tool_hang': 'tool_hang'
    }
    
    vl_task_dir = vl_json_task_map.get(task_name, task_name)
    
    return {
        'task_name': task_name,
        'hdf5_path': os.path.join(task_root, "low_dim_abs.hdf5"),
        'image_hdf5_path': os.path.join(task_root, "image_abs.hdf5"),
        'video_dir': os.path.join(task_root, "video"),
        'divided_events_dir': os.path.join(DIVIDED_EVENTS_DIR, vl_task_dir, "json_sg"),
        'divided_events_image_dir': os.path.join(DIVIDED_EVENTS_DIR, vl_task_dir),
        'corrected_events_dir': os.path.join(CORRECTED_EVENTS_DIR, task_name),
        'corrected_events_organized_dir': os.path.join(CORRECTED_EVENTS_ORGANIZED_DIR, task_name),
        'output_dir': os.path.join(task_root, "kp_training"),
        'dataset_root': task_root
    }

# ============================================================================
# Relative Paths (for scripts running from keypose directory)
# ============================================================================

def get_relative_paths(task_name):
    """
    Get relative paths from keypose directory.
    Useful for scripts that cd into keypose directory before running.
    
    Args:
        task_name: Task name (e.g., 'can', 'square', 'tool_hang')
    
    Returns:
        Dictionary containing relative paths
    """
    return {
        'divided_events_dir': f"./data/divided_events",
        'corrected_events_dir': f"./data/corrected_events/{task_name}",
        'corrected_events_organized_dir': f"./data/corrected_events_organized/{task_name}"
    }

# ============================================================================
# API Configuration
# ============================================================================

# ============================================================================
# Environment Configuration
# ============================================================================

# Conda environment name
CONDA_ENV = "tase"

# ============================================================================
# API Configuration
# ============================================================================

# DashScope API Key for the VL model (Qwen3-VL).
# Provide it via the DASHSCOPE_API_KEY environment variable; no key is bundled.
DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY')

# ============================================================================
# Helper Functions
# ============================================================================

def ensure_dir(path):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)
    return path

def print_config(task_name=None):
    """Print current configuration for debugging."""
    print("=" * 80)
    print("Keypose Pipeline Configuration")
    print("=" * 80)
    print(f"KEYPOSE_ROOT: {KEYPOSE_ROOT}")
    print(f"KEYPOSE_DATA_ROOT: {KEYPOSE_DATA_ROOT}")
    print(f"ROBOMIMIC_DATA_ROOT: {ROBOMIMIC_DATA_ROOT}")
    print(f"DIVIDED_EVENTS_DIR: {DIVIDED_EVENTS_DIR}")
    print(f"CORRECTED_EVENTS_DIR: {CORRECTED_EVENTS_DIR}")
    print(f"CORRECTED_EVENTS_ORGANIZED_DIR: {CORRECTED_EVENTS_ORGANIZED_DIR}")
    
    if task_name:
        print("\n" + "-" * 80)
        print(f"Task: {task_name}")
        print("-" * 80)
        paths = get_task_paths(task_name)
        for key, value in paths.items():
            print(f"{key}: {value}")
    print("=" * 80)

if __name__ == "__main__":
    # Test configuration
    import sys
    task = sys.argv[1] if len(sys.argv) > 1 else "can"
    print_config(task)

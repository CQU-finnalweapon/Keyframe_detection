#!/bin/bash
# Configuration file for demo scripts
# 
# This script reads configuration from ../config.py (single source of truth)
# Only edit config.py to change paths!

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CONFIG_PY="$SCRIPT_DIR/../config.py"

# ============================================================================
# Load Configuration from Python config.py
# ============================================================================

# Read base directories from config.py
export KEYPOSE_ROOT=$(python3 -c "import sys; sys.path.insert(0, '$SCRIPT_DIR/..'); from config import KEYPOSE_ROOT; print(KEYPOSE_ROOT)")

# Read data directories
export ROBOMIMIC_DATA_ROOT=$(python3 -c "import sys; sys.path.insert(0, '$SCRIPT_DIR/..'); from config import ROBOMIMIC_DATA_ROOT; print(ROBOMIMIC_DATA_ROOT)")
export DIVIDED_EVENTS_DIR=$(python3 -c "import sys; sys.path.insert(0, '$SCRIPT_DIR/..'); from config import DIVIDED_EVENTS_DIR; print(DIVIDED_EVENTS_DIR)")
export CORRECTED_EVENTS_DIR=$(python3 -c "import sys; sys.path.insert(0, '$SCRIPT_DIR/..'); from config import CORRECTED_EVENTS_DIR; print(CORRECTED_EVENTS_DIR)")

# ============================================================================
# Conda Environment (edit in config.py if needed)
# ============================================================================

# Default conda environment - can be overridden in config.py
export CONDA_ENV="tase"

# ============================================================================
# Helper Functions
# ============================================================================

# Get task-specific paths
get_task_hdf5_path() {
    local task=$1
    echo "$ROBOMIMIC_DATA_ROOT/$task/ph/low_dim_abs.hdf5"
}

get_task_dataset_root() {
    local task=$1
    echo "$ROBOMIMIC_DATA_ROOT/$task/ph"
}

get_task_vl_json_dir() {
    local task=$1
    # Note: divided_events folders now match robomimic dataset folder names
    echo "$DIVIDED_EVENTS_DIR/${task}/json_sg"
}

get_task_output_dir() {
    local task=$1
    echo "$ROBOMIMIC_DATA_ROOT/$task/ph/kp_training"
}

get_task_corrected_events_dir() {
    local task=$1
    echo "$CORRECTED_EVENTS_DIR/$task"
}

#!/bin/bash
# Step 3: Generate keypose training dataset
#
# Two modes:
#   VL mode (default): Uses VL-corrected events (requires Steps 1-2)
#   Auto mode (--auto): Uses automatic trajectory segmentation (no VL needed)
#
# Usage:
#   ./03_generate_dataset.sh <task_name> <num_episodes> [episode_list]
#   ./03_generate_dataset.sh --auto <task_name> [num_episodes]
#
# Examples:
#   ./03_generate_dataset.sh can 200              # VL mode, episodes 0-199
#   ./03_generate_dataset.sh can 1 5              # VL mode, only episode 5
#   ./03_generate_dataset.sh can 3 "5,7,9"        # VL mode, episodes 5,7,9
#   ./03_generate_dataset.sh --auto tool_hang 200  # Auto mode, episodes 0-199
#   ./03_generate_dataset.sh --auto all            # Auto mode, all tasks (200 eps each)
#
# IMPORTANT: Both modes use low_dim_abs.hdf5 (absolute actions).
#   The trajectory policy expects absolute actions (pos + axis-angle)
#   which get converted to rotation_6d by the dataset loader.

# Load configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/config.sh"

# Check for --auto flag
if [[ "$1" == "--auto" ]]; then
    shift
    TASK=${1:-can}
    NUM_EPISODES=${2:-200}

    # Handle "all" task
    if [[ "$TASK" == "all" ]]; then
        echo "============================================"
        echo "Step 3: Generating Keypose Training Dataset"
        echo "Mode: Auto (all tasks)"
        echo "============================================"
        for T in can square tool_hang; do
            HDF5_PATH=$(get_task_hdf5_path "$T")
            OUTPUT_DIR=$(get_task_output_dir "$T")
            cd "$KEYPOSE_ROOT"
            conda run -n $CONDA_ENV python gen_dataset.py \
                --hdf5_path "$HDF5_PATH" \
                --output_dir "$OUTPUT_DIR" \
                --num_episodes "$NUM_EPISODES" \
                --task "$T" \
                --auto
        done
        exit 0
    fi

    HDF5_PATH=$(get_task_hdf5_path "$TASK")
    OUTPUT_DIR=$(get_task_output_dir "$TASK")

    echo "============================================"
    echo "Step 3: Generating Keypose Training Dataset"
    echo "Mode: Auto (trajectory segmentation)"
    echo "Task: $TASK"
    echo "Episodes: $NUM_EPISODES"
    echo "Source: $HDF5_PATH"
    echo "============================================"
    echo ""

    cd "$KEYPOSE_ROOT"
    conda run -n $CONDA_ENV python gen_dataset.py \
        --hdf5_path "$HDF5_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --num_episodes "$NUM_EPISODES" \
        --task "$TASK" \
        --auto

    echo ""
    echo "============================================"
    echo "Auto dataset generation completed!"
    echo "Output saved to: $OUTPUT_DIR"
    echo "============================================"
    exit 0
fi

TASK=${1:-can}
NUM_EPISODES=${2:-200}
EPISODE_LIST=${3:-}

# Get task-specific paths
HDF5_PATH=$(get_task_hdf5_path "$TASK")
VL_JSON_DIR="$KEYPOSE_ROOT/data/corrected_events/$TASK"
OUTPUT_DIR=$(get_task_output_dir "$TASK")

echo "============================================"
echo "Step 3: Generating Keypose Training Dataset"
echo "Task: $TASK"
if [[ -n "$EPISODE_LIST" ]]; then
    echo "Episodes: $EPISODE_LIST (specific episodes)"
else
    echo "Episodes: 0-$((NUM_EPISODES-1))"
fi
echo "Corrector: V3"
echo "============================================"
echo ""

# Change to keypose directory for relative paths
cd "$KEYPOSE_ROOT"

# Build command with episode list if provided
EPISODE_FLAG="--num_episodes $NUM_EPISODES"
if [[ -n "$EPISODE_LIST" ]]; then
    EPISODE_FLAG="--episode_list $EPISODE_LIST"
fi

conda run -n $CONDA_ENV python gen_dataset.py \
    --hdf5_path "$HDF5_PATH" \
    --vl_json_dir "$VL_JSON_DIR" \
    --output_dir "$OUTPUT_DIR" \
    $EPISODE_FLAG \
    --task $TASK

echo ""
echo "============================================"
echo "Dataset generation completed!"
echo "Output saved to: $OUTPUT_DIR"
echo "============================================"

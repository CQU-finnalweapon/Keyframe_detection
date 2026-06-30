#!/bin/bash
# Step 1: Convert HDF5 to videos and divide them with VL model
# Usage: ./01_prepare_videos.sh <task_name> <num_episodes>

# Load configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/config.sh"

TASK=${1:-can}
NUM_EPISODES=${2:-1}

# Get task-specific paths
DATASET_ROOT=$(get_task_dataset_root "$TASK")
OUTPUT_DIR="$KEYPOSE_ROOT/data/divided_events"

echo "============================================"
echo "Step 1.1: Converting HDF5 to Videos"
echo "Task: $TASK"
echo "Episodes: $NUM_EPISODES"
echo "============================================"
echo ""

# 1.1 Convert HDF5 to videos
cd "$KEYPOSE_ROOT"

conda run -n $CONDA_ENV python hdf5_to_video.py \
    --task "$TASK" \
    --dataset_root "$DATASET_ROOT" \
    --num_episodes $NUM_EPISODES \
    --dataset_name "image_abs" \
    --delta_time 0.05

echo ""
echo "============================================"
echo "Step 1.2: Analyzing Videos with VL Model"
echo "============================================"
echo ""

# 1.2 Run VL model to divide videos into subtasks
conda run -n $CONDA_ENV python divide_video.py \
    --dataset_folder "$DATASET_ROOT/video/" \
    --task "$TASK" \
    --model "3-vl-235b-t" \
    --num_video $NUM_EPISODES \
    --out_dir "$OUTPUT_DIR" \
    --target_fps 5 \
    --frame_exist True \
    --scale 1 \
    --version 10

echo ""
echo "============================================"
echo "Videos and VL analysis completed!"
echo "Output saved to: $OUTPUT_DIR"
echo "============================================"

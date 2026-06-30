#!/bin/bash
# Step 2: Correct VL events with trajectory data (V3)
# Usage: ./02_correct_events.sh <task_name> <num_episodes_or_episode_idx> <episode_list_or_no_gap_filling> [vl_model]
# Examples:
#   ./02_correct_events.sh can 10 "" qwen3-vl-235b-a22b-thinking       # Process episodes 0-9
#   ./02_correct_events.sh can 1 "5" qwen3-vl-235b-a22b-thinking        # Process only episode 5
#   ./02_correct_events.sh can 3 "5,7,9" qwen3-vl-235b-a22b-thinking    # Process episodes 5,7,9
# 
# To customize direction threshold, edit DIRECTION_THRESHOLD variable in this script
# Default is 90 degrees (splits when direction changes > 90°)
#
# To enable stage-by-stage segmentation breakdown in trajectory_intervals.txt,
# set DEBUG_SEGMENTATION="" in this script

# Load configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/config.sh"

TASK=${1:-can}
NUM_EPISODES=${2:-1}
EPISODE_ARG=${3:-}  # Can be episode list (e.g., "5,7,9") or --no-gap-filling
VL_MODEL=${4:-qwen3-vl-235b-a22b-thinking}  # Default model

# Parse episode argument - check if it's a list of episodes or gap filling flag
NO_GAP_FILLING=""
EPISODE_LIST=""
if [[ "$EPISODE_ARG" == "--no-gap-filling" ]]; then
    NO_GAP_FILLING="$EPISODE_ARG"
elif [[ -n "$EPISODE_ARG" ]]; then
    EPISODE_LIST="$EPISODE_ARG"
fi

# Get task-specific paths
HDF5_PATH=$(get_task_hdf5_path "$TASK")
VL_JSON_DIR=$(get_task_vl_json_dir "$TASK")
OUTPUT_DIR="$KEYPOSE_ROOT/data/corrected_events/$TASK"

echo "============================================"
echo "Step 2: Correcting VL Events with Trajectory"
echo "Task: $TASK"
echo "Model: $VL_MODEL"
if [[ -n "$EPISODE_LIST" ]]; then
    echo "Episodes: $EPISODE_LIST (specific episodes)"
else
    echo "Episodes: 0-$((NUM_EPISODES-1))"
fi
echo "Using: V3 segmentation and event corrector"
echo "============================================"
echo ""

cd "$KEYPOSE_ROOT"

# Find VL JSON file (template) for episode 0 with specified model
# Extract version numbers, sort numerically, and pick the newest
VL_JSON=$(ls -1 $VL_JSON_DIR/imglist_episode_0_${VL_MODEL}_*.json 2>/dev/null | \
          sed 's/.*_v\([0-9]\+\)\.json$/\1 &/' | sort -n -k1,1 | tail -1 | cut -d' ' -f2)

if [ -z "$VL_JSON" ]; then
    echo "Error: No VL JSON template found for model $VL_MODEL"
    exit 1
fi

echo "Using VL JSON template: $(basename $VL_JSON)"
if [[ -n "$EPISODE_LIST" ]]; then
    echo "Processing episodes: $EPISODE_LIST"
else
    echo "Processing episodes 0-$((NUM_EPISODES-1))"
fi
echo ""

# Create log directory
LOG_DIR="$KEYPOSE_ROOT/logs/$TASK"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/correction_$(date +%Y%m%d_%H%M%S).log"

echo "Logging to: $LOG_FILE"
echo ""

# Default thresholds
SAMPLING_INTERVAL=0.04
POS_THRESH=5e-3
ROT_THRESH=1e-2
STABLE_WIN=5
GRIPPER_THRESHOLD=2e-4
DIRECTION_THRESHOLD=90  # Angle threshold in degrees for direction-based splitting
DIRECTION_SIMILARITY_THRESHOLD=0.5  # Cosine similarity threshold for gap filling
MAX_GAP=5
DEBUG_SEGMENTATION="--debug_segmentation"  # Set to --debug_segmentation to enable stage-by-stage output

# Use quiet mode for large batch processing (>5 episodes)
QUIET_FLAG=""
if [ "$NUM_EPISODES" -gt 5 ]; then
    QUIET_FLAG="--quiet"
fi

# Task-specific thresholds
# if [ "$TASK" == "tool_hang" ]; then
#     POS_THRESH=2e-3
#     ROT_THRESH=5e-3
#     STABLE_WIN=5
#     echo "Using stricter thresholds for tool_hang: POS=$POS_THRESH, ROT=$ROT_THRESH"
# fi

# Build command with episode list if provided
EPISODE_FLAG="--num_episodes $NUM_EPISODES"
if [[ -n "$EPISODE_LIST" ]]; then
    EPISODE_FLAG="--episode_list $EPISODE_LIST"
fi

# Run merge pipeline for all episodes at once with detailed logging
conda run -n $CONDA_ENV python merge_trajectory_vl.py \
    --hdf5_path "$HDF5_PATH" \
    --vl_json "$VL_JSON" \
    $EPISODE_FLAG \
    --output_dir "$OUTPUT_DIR" \
    --sampling_interval $SAMPLING_INTERVAL \
    --position_threshold $POS_THRESH \
    --rotation_threshold $ROT_THRESH \
    --gripper_threshold $GRIPPER_THRESHOLD \
    --stable_window $STABLE_WIN \
    --max_gap $MAX_GAP \
    --direction_threshold_deg $DIRECTION_THRESHOLD \
    --direction_similarity_threshold $DIRECTION_SIMILARITY_THRESHOLD \
    --task "$TASK" \
    $QUIET_FLAG $NO_GAP_FILLING $DEBUG_SEGMENTATION 2>&1 | tee "$LOG_FILE"

echo ""
echo "============================================"
echo "Event correction completed!"
echo "Output saved to: $OUTPUT_DIR"
echo "Log saved to: $LOG_FILE"
echo "============================================"

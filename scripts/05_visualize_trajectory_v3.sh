#!/bin/bash
# Visualize trajectory with V3 per-axis motion analysis
#
# Usage:
#   ./05_visualize_trajectory_v3.sh <task> <episode> [options]
#
# Examples:
#   # Basic - square task, episode 0  (auto-enables correction rows if data exists)
#   ./05_visualize_trajectory_v3.sh square 0
#
#   # Without slow-motion row
#   ./05_visualize_trajectory_v3.sh square 0 --no-slow-motion
#
#   # Without VL correction rows
#   ./05_visualize_trajectory_v3.sh square 0 --no-correction
#
#   # Custom output path
#   ./05_visualize_trajectory_v3.sh square 0 --output /tmp/my_plot.png
#
#   # Custom smoothing window and velocity thresholds
#   ./05_visualize_trajectory_v3.sh tool_hang 3 --window 21 --thresh_z 0.005
#
#   # Specific frame range
#   ./05_visualize_trajectory_v3.sh can 5 --frame_start 10 --frame_end 80
#
#   # Multi-scale window analysis
#   ./05_visualize_trajectory_v3.sh tool_hang 0 --multi-scale
#
#   # Manually supply VL correction files
#   ./05_visualize_trajectory_v3.sh square 0 --show-correction \
#       --vl_imglist data/divided_events/square/json_sg/imglist_episode_0_*.json \
#       --events_json data/corrected_events_organized/square/000_019/episode_0/events.json \
#       --intervals_json data/corrected_events_organized/square/000_019/episode_0/intervals.json
#
# Supported tasks: tool_hang, square, can

# Load configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/config.sh"

TASK=${1:-tool_hang}
EPISODE=${2:-3}
shift 2 2>/dev/null || true  # Shift past task and episode, ignore error if not enough args

# Get task-specific HDF5 path
HDF5_PATH=$(get_task_hdf5_path "$TASK")

echo "============================================"
echo "Trajectory Visualization V3"
echo "Task: $TASK"
echo "Episode: $EPISODE"
echo "HDF5: $HDF5_PATH"
echo "============================================"
echo ""

cd "$KEYPOSE_ROOT"

# Default output path
OUTPUT_DIR="$KEYPOSE_ROOT/data/visualizations/$TASK"
mkdir -p "$OUTPUT_DIR"
DEFAULT_OUTPUT="$OUTPUT_DIR/${TASK}_ep${EPISODE}_v3.png"

# Check if --output is provided in remaining args
HAS_OUTPUT=false
for arg in "$@"; do
    if [[ "$arg" == "--output" ]]; then
        HAS_OUTPUT=true
        break
    fi
done

if [ "$HAS_OUTPUT" = false ]; then
    EXTRA_ARGS="--output $DEFAULT_OUTPUT"
else
    EXTRA_ARGS=""
fi

# Auto-enable --show-correction if corrected events data exists for this episode
BATCH_START=$(( (EPISODE / 20) * 20 ))
BATCH_END=$(( BATCH_START + 19 ))
BATCH_DIR=$(printf "%03d_%03d" $BATCH_START $BATCH_END)
CORR_DIR="$KEYPOSE_ROOT/data/corrected_events_organized/$TASK/$BATCH_DIR/episode_$EPISODE"
if [ -f "$CORR_DIR/events.json" ] && [ -f "$CORR_DIR/intervals.json" ]; then
    # Only add if the user hasn't explicitly opted out
    if [[ ! " $@ " =~ "--no-correction" ]]; then
        EXTRA_ARGS="$EXTRA_ARGS --show-correction"
    fi
fi

conda run -n $CONDA_ENV python viz/visualize_trajectory_v3.py \
    --hdf5_path "$HDF5_PATH" \
    --episode $EPISODE \
    --task "$TASK" \
    $EXTRA_ARGS \
    "$@"

echo ""
echo "============================================"
echo "Visualization completed!"
if [ "$HAS_OUTPUT" = false ]; then
    echo "Output: $DEFAULT_OUTPUT"
fi
echo "============================================"

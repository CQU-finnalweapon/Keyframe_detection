#!/bin/bash
# Complete keypose pipeline
#
# Two modes:
#   VL mode (default):  Steps 1→2→3 (video extraction → VL analysis → event correction → dataset)
#   Auto mode (--auto): Step 3 only  (trajectory segmentation, no VL needed — fast & free)
#
# Usage:
#   ./run_full_pipeline.sh <task_name> <num_episodes> [vl_model]    # VL mode
#   ./run_full_pipeline.sh --auto <task_name> [num_episodes]        # Auto mode
#   ./run_full_pipeline.sh --auto all                               # Auto mode, all tasks
#
# Examples:
#   ./run_full_pipeline.sh can 50                                   # VL mode, 50 episodes
#   ./run_full_pipeline.sh tool_hang 200 qwen3-vl-235b-a22b-thinking
#   ./run_full_pipeline.sh --auto can 200                           # Auto mode, 200 episodes
#   ./run_full_pipeline.sh --auto all                               # Auto mode, all 3 tasks

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# ============================================================================
# Auto mode: skip VL (Steps 1-2), run Step 3 with --auto
# ============================================================================
if [[ "$1" == "--auto" ]]; then
    shift
    TASK=${1:-can}
    NUM_EPISODES=${2:-200}

    if [[ "$TASK" == "all" ]]; then
        echo "========================================"
        echo "Running Auto Pipeline (all tasks)"
        echo "Episodes per task: $NUM_EPISODES"
        echo "========================================"
        for T in can square tool_hang; do
            echo ""
            echo ">>> [$T] Step 3: Generating dataset (auto mode)"
            bash "$SCRIPT_DIR/03_generate_dataset.sh" --auto "$T" "$NUM_EPISODES"
            if [ $? -ne 0 ]; then
                echo "Error processing $T. Exiting."
                exit 1
            fi
        done
    else
        echo "========================================"
        echo "Running Auto Pipeline"
        echo "Task: $TASK"
        echo "Episodes: $NUM_EPISODES"
        echo "========================================"
        echo ""
        echo ">>> Step 3: Generating dataset (auto mode)"
        bash "$SCRIPT_DIR/03_generate_dataset.sh" --auto "$TASK" "$NUM_EPISODES"
        if [ $? -ne 0 ]; then
            echo "Error in Step 3. Exiting."
            exit 1
        fi
    fi

    echo ""
    echo "========================================"
    echo "✓ Auto pipeline completed successfully!"
    echo "========================================"
    exit 0
fi

# ============================================================================
# VL mode (default): run full Steps 1→2→3
# ============================================================================
TASK=${1:-can}
NUM_EPISODES=${2:-1}
VL_MODEL=${3:-qwen3-vl-235b-a22b-thinking}

echo "========================================"
echo "Running Full VL Pipeline"
echo "Task: $TASK"
echo "Episodes: $NUM_EPISODES"
echo "VL Model: $VL_MODEL"
echo "========================================"
echo ""

# Step 1: Prepare videos and VL analysis
echo ">>> STEP 1: Preparing videos and VL analysis"
bash "$SCRIPT_DIR/01_prepare_videos.sh" "$TASK" "$NUM_EPISODES"

if [ $? -ne 0 ]; then
    echo "Error in Step 1. Exiting."
    exit 1
fi

echo ""
echo ">>> STEP 2: Correcting VL events with trajectory (V3)"
bash "$SCRIPT_DIR/02_correct_events.sh" "$TASK" "$NUM_EPISODES" "" "$VL_MODEL"

if [ $? -ne 0 ]; then
    echo "Error in Step 2. Exiting."
    exit 1
fi

echo ""
echo ">>> STEP 3: Generating training dataset (VL mode)"
bash "$SCRIPT_DIR/03_generate_dataset.sh" "$TASK" "$NUM_EPISODES"

if [ $? -ne 0 ]; then
    echo "Error in Step 3. Exiting."
    exit 1
fi

echo ""
echo "========================================"
echo "✓ Full VL pipeline completed successfully!"
echo "========================================"

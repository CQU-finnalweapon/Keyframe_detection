#!/bin/bash
# Process all three tasks: event correction → dataset generation
#
# Two modes:
#   VL mode (default):  Steps 2→3 for each task (requires Step 1 already done)
#   Auto mode (--auto): Step 3 only for each task (no VL needed)
#
# Usage:
#   ./correct_all_tasks.sh                # VL mode, all tasks
#   ./correct_all_tasks.sh --auto         # Auto mode, all tasks

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/config.sh"

# ============================================================================
# Auto mode: skip Step 2, run Step 3 --auto for all tasks
# ============================================================================
if [[ "$1" == "--auto" ]]; then
    echo "=========================================="
    echo "Auto mode: all tasks (trajectory only)"
    echo "=========================================="

    for task in can square tool_hang; do
        echo ""
        echo "=========================================="
        echo "Processing task: $task (auto)"
        echo "=========================================="

        # Step 3 only: Generate keypose dataset (auto)
        echo "[Step 3/3] Generating keypose datasets (auto)..."
        bash "$SCRIPT_DIR/03_generate_dataset.sh" --auto $task 200

        # Show results
        dataset_dir=$(get_task_output_dir "$task")
        dataset_count=$(ls -1 $dataset_dir/kp_episode_*.hdf5 2>/dev/null | wc -l)

        echo ""
        echo "Results for $task:"
        echo "  Keypose datasets: $dataset_count"
        echo ""
    done

    echo "=========================================="
    echo "✓ ALL TASKS COMPLETED (auto mode)"
    echo "=========================================="
    exit 0
fi

# ============================================================================
# VL mode (default): Steps 2→3 for each task
# ============================================================================
for task in can square tool_hang; do
  echo ""
  echo "=========================================="
  echo "Processing task: $task"
  echo "=========================================="
  
  # Step 2: Correct VL events for all episodes (V3)
  echo "[Step 2/3] Correcting VL events..."
  bash "$SCRIPT_DIR/02_correct_events.sh" $task 200 "" qwen3-vl-235b-a22b-thinking
  
  # Step 3: Generate keypose dataset for all episodes
  echo "[Step 3/3] Generating keypose datasets..."
  bash "$SCRIPT_DIR/03_generate_dataset.sh" $task 200
  
  # Show results
  corrected_count=$(ls -1 $KEYPOSE_ROOT/data/corrected_events/$task/corrected_episode_*.json 2>/dev/null | wc -l)
  dataset_dir=$(get_task_output_dir "$task")
  dataset_count=$(ls -1 $dataset_dir/kp_episode_*.hdf5 2>/dev/null | wc -l)
  
  echo ""
  echo "Results for $task:"
  echo "  Corrected JSONs: $corrected_count"
  echo "  Keypose datasets: $dataset_count"
  echo ""
done

echo "=========================================="
echo "✓ ALL TASKS COMPLETED"
echo "=========================================="

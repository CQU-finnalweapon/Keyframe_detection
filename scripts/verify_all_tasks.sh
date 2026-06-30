#!/bin/bash
# Verify correction results for all tasks against defined rules
# Usage: ./verify_all_tasks.sh [num_episodes_per_task]
#        ./verify_all_tasks.sh          # Verify all episodes
#        ./verify_all_tasks.sh 20       # Verify first 20 episodes per task

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/config.sh"

NUM_EPISODES=${1:-}  # Optional: limit episodes per task

echo "============================================"
echo "Verifying Correction Results for All Tasks"
echo "============================================"
echo ""

cd "$KEYPOSE_ROOT"

FAILED_TASKS=()
PASSED_TASKS=()

for task in can square tool_hang; do
    echo ""
    echo "=========================================="
    echo "Verifying task: $task"
    echo "=========================================="
    
    if [[ -n "$NUM_EPISODES" ]]; then
        # Build episode list 0,1,2,...,N-1
        EPISODE_LIST=$(seq -s, 0 $((NUM_EPISODES - 1)))
        conda run -n $CONDA_ENV python tests/test_correction_rules.py \
            --task $task \
            --episode_list "$EPISODE_LIST" \
            --quiet
    else
        conda run -n $CONDA_ENV python tests/test_correction_rules.py \
            --task $task \
            --quiet
    fi
    
    if [ $? -eq 0 ]; then
        PASSED_TASKS+=($task)
        echo "✓ $task: All episodes passed"
    else
        FAILED_TASKS+=($task)
        echo "✗ $task: Some episodes failed"
    fi
done

echo ""
echo "=========================================="
echo "Verification Summary"
echo "=========================================="
echo "Passed tasks: ${PASSED_TASKS[*]:-none}"
echo "Failed tasks: ${FAILED_TASKS[*]:-none}"
echo "=========================================="

# Exit with failure if any task failed
if [ ${#FAILED_TASKS[@]} -gt 0 ]; then
    exit 1
else
    echo "✓ All tasks passed verification!"
    exit 0
fi

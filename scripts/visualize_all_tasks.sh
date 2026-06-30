#!/bin/bash
# Visualize trajectories for all three tasks (V3)
# Usage: ./visualize_all_tasks.sh [num_episodes_per_task]

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/config.sh"

NUM_EPISODES=${1:-5}  # Default: 5 episodes per task

for task in can square tool_hang; do
  echo ""
  echo "=========================================="
  echo "Visualizing task: $task"
  echo "=========================================="
  
  for ep in $(seq 0 $((NUM_EPISODES - 1))); do
    echo ""
    echo "[Episode $ep] Visualizing trajectory..."
    bash "$SCRIPT_DIR/05_visualize_trajectory_v3.sh" $task $ep
  done
  
  echo ""
  echo "Completed visualization for $task (Episodes 0-$((NUM_EPISODES - 1)))"
  echo ""
done

echo ""
echo "=========================================="
echo "All visualizations completed!"
echo "=========================================="

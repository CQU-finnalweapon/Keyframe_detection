#!/bin/bash
# Step 4: Reorganize and visualize corrected events
# Usage: ./04_visualize_events.sh [tasks...]
# Example: ./04_visualize_events.sh can square tool_hang

# Load configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/config.sh"

# Default to all tasks if none specified
TASKS=${@:-can square tool_hang}

echo "============================================"
echo "Step 4: Reorganizing and Visualizing Events"
echo "Tasks: $TASKS"
echo "============================================"
echo ""

cd "$KEYPOSE_ROOT"

conda run -n $CONDA_ENV python viz/reorganize_and_visualize.py \
    --corrected-events-dir "$KEYPOSE_ROOT/data/corrected_events" \
    --data-dir "$ROBOMIMIC_DATA_ROOT" \
    --output-dir "$KEYPOSE_ROOT/data/corrected_events_organized" \
    --tasks $TASKS

echo ""
echo "============================================"
echo "Visualization completed!"
echo "Output: $KEYPOSE_ROOT/data/corrected_events_organized"
echo "============================================"

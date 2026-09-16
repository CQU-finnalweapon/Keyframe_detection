#!/usr/bin/env bash
# Step 6 (LeRobot): annotation-driven keypose extraction -- no VLM needed.
#
# LeRobot datasets recorded on Moz1 already carry hand-annotated subtasks, so the
# semantic stage ("what happens, roughly when") is free and the whole VLM half of
# the pipeline is skipped.  Only the deterministic trajectory/gripper analysis
# runs, turning each annotated window into an exact keypose frame.
#
# Usage:
#   ./06_generate_lerobot_dataset.sh <dataset_root> [output_dir] [num_episodes]
#
# Examples:
#   ./06_generate_lerobot_dataset.sh /mnt/vepfs01/output/klay.zhou/datasets/PickPlaceONLY_Moz1_cjb
#   ./06_generate_lerobot_dataset.sh /path/to/dataset ./data/lerobot_kp 500
#   ./06_generate_lerobot_dataset.sh /path/to/dataset ./data/lerobot_kp all
#
# Environment:
#   PYTHON   python interpreter to use (default: python)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
PYTHON="${PYTHON:-python}"

DATASET_ROOT="${1:-}"
OUTPUT_DIR="${2:-$ROOT_DIR/data/lerobot_kp}"
NUM_EPISODES="${3:-10}"

usage() {
    sed -n '2,18p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 1
}

if [[ -z "$DATASET_ROOT" ]]; then
    echo "ERROR: dataset_root is required." >&2
    usage
fi

if [[ ! -f "$DATASET_ROOT/dataset.json" ]]; then
    echo "ERROR: $DATASET_ROOT/dataset.json not found -- is this a LeRobot dataset root?" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# "all" processes every episode present in dataset.json
if [[ "$NUM_EPISODES" == "all" ]]; then
    NUM_EPISODES="$("$PYTHON" -c "
import json,sys
print(len(json.load(open('$DATASET_ROOT/dataset.json')).get('episodes') or []))
")"
fi

echo "======================================================================"
echo "LeRobot keypose extraction (annotation-driven, no VLM)"
echo "======================================================================"
echo "Dataset  : $DATASET_ROOT"
echo "Output   : $OUTPUT_DIR"
echo "Episodes : $NUM_EPISODES"
echo "Python   : $PYTHON"
echo "======================================================================"

cd "$ROOT_DIR"
"$PYTHON" gen_dataset_lerobot.py \
    --dataset_root "$DATASET_ROOT" \
    --output_dir "$OUTPUT_DIR" \
    --num_episodes "$NUM_EPISODES"

echo
echo "Validate the result with:"
echo "  $PYTHON tests/evaluate_lerobot_keyposes.py --dataset_root $DATASET_ROOT --num_episodes $NUM_EPISODES"
echo "  $PYTHON viz/visualize_lerobot_episode.py --dataset_root $DATASET_ROOT --episode 0"

# Keypose Dataset Generation Pipeline — Scripts

Shell scripts for the keypose dataset generation pipeline.

## ⚙️ Configuration

Paths resolve from `../config.py` (relative to the project; no edits needed for
the default layout). Override input/output roots and the VL key via environment
variables:

```bash
export KEYPOSE_LABELLING_ROBOMIMIC_ROOT=/path/to/robomimic   # input datasets
export KEYPOSE_LABELLING_DATA_ROOT=/path/to/generated        # outputs (default: ../data)
export DASHSCOPE_API_KEY=sk-...                              # only for VL mode
```

`config.sh` reads from `config.py` automatically.

## Overview

Two pipeline modes are available:

| Mode | Steps | Requires | Speed | Quality |
|------|-------|----------|-------|---------|
| **VL mode** (default) | 1 → 2 → 3 | VL API key, network | Slow (API calls) | Higher (semantic events) |
| **Auto mode** (`--auto`) | 3 only | Nothing extra | Fast | Good (gripper-based events) |

### Steps

1. **Prepare Videos** (`01_prepare_videos.sh`) — HDF5 → video + VL model analysis (**expensive**)
2. **Correct Events** (`02_correct_events.sh`) — Match VL events with trajectory data
3. **Generate Dataset** (`03_generate_dataset.sh`) — Create keypose training HDF5 files

## Quick Start

### Auto Mode (recommended for fast iteration)

```bash
# Single task
./run_full_pipeline.sh --auto can 200

# All 3 tasks at once
./run_full_pipeline.sh --auto all

# Or use the batch helper
./correct_all_tasks.sh --auto
```

### VL Mode (full quality)

```bash
# Complete pipeline: videos → VL → correction → dataset
./run_full_pipeline.sh can 50

# Or run all tasks (Steps 2-3, assumes Step 1 done)
./correct_all_tasks.sh
```

### Run Individual Steps

```bash
# Step 1: Extract videos + VL analysis (slow, API calls)
./01_prepare_videos.sh can 5

# Step 2: Correct VL events with trajectory
./02_correct_events.sh can 5

# Step 3: Generate training dataset
./03_generate_dataset.sh can 50               # VL mode
./03_generate_dataset.sh --auto can 200        # Auto mode
./03_generate_dataset.sh --auto all            # Auto mode, all tasks
./03_generate_dataset.sh can 1 "5"             # VL mode, episode 5 only
./03_generate_dataset.sh can 3 "5,7,9"         # VL mode, episodes 5,7,9
```

## Script Details

### run_full_pipeline.sh

**Top-level orchestrator.** Supports both modes:

```
./run_full_pipeline.sh <task> <num_episodes> [vl_model]    # VL mode
./run_full_pipeline.sh --auto <task> [num_episodes]        # Auto mode
./run_full_pipeline.sh --auto all                          # Auto mode, all tasks
```

- VL mode: runs Steps 1 → 2 → 3 sequentially, exits on any error
- Auto mode: runs Step 3 only with `--auto` flag (no VL needed)

### correct_all_tasks.sh

**Batch helper for all 3 tasks** (can, square, tool_hang, 200 episodes each):

```
./correct_all_tasks.sh           # VL mode: Steps 2→3 (assumes Step 1 done)
./correct_all_tasks.sh --auto    # Auto mode: Step 3 only
```

### 01_prepare_videos.sh

Converts HDF5 to MP4 videos, then runs VL model to identify subtask events.

| Setting | Value |
|---------|-------|
| Video FPS | 20 (robomimic default) |
| VL sampling FPS | 5 |
| VL model | qwen3-vl-235b-a22b-thinking |

**Output:** `data/divided_events/<task>/json_sg/imglist_episode_*.json`

### 02_correct_events.sh

Matches VL events with trajectory data using V3 per-axis segmentation.

| Setting | Value |
|---------|-------|
| Sampling interval | 0.04s (20fps) |
| Position threshold | 5e-3 |
| Rotation threshold | 1e-2 |
| Gripper threshold | 2e-4 |
| Stable window | 5 frames |
| Max gap | 5 frames |

**Output:** `data/corrected_events/<task>/corrected_episode_*.json`

### 03_generate_dataset.sh

Creates keypose training HDF5 files from trajectories.

```
./03_generate_dataset.sh <task> <num_episodes> [episode_list]   # VL mode
./03_generate_dataset.sh --auto <task> [num_episodes]           # Auto mode
./03_generate_dataset.sh --auto all                             # Auto mode, all tasks
```

**IMPORTANT:** Both modes use `low_dim_abs.hdf5` (absolute actions).

**Output format** (`kp_episode_<N>.hdf5`):

| Key | Shape | Description |
|-----|-------|-------------|
| `obs/epos` | (T, 9) | End-effector pose |
| `obs/env_obs` | (T, D) | Environment observation [object, eef_pos, eef_quat, gripper] |
| `obs/qpos` | (T, 7) | Joint positions |
| `obs/prev_keypose` | (T, 9) | Previous keypose |
| `target/next_keypose` | (T, 9) | Target keypose |
| `target/event_type` | (T,) | Event type indices |
| `actions` | (T, 7) | Absolute actions |

**Output dir:** `<robomimic_data>/<task>/ph/kp_training/`

## Directory Structure

```
scripts/                           # This folder
├── run_full_pipeline.sh           # Top-level orchestrator (--auto supported)
├── correct_all_tasks.sh           # Batch all tasks (--auto supported)
├── verify_all_tasks.sh            # Validate correction results
├── 01_prepare_videos.sh           # Step 1: HDF5 → video + VL
├── 02_correct_events.sh           # Step 2: Event correction
├── 03_generate_dataset.sh         # Step 3: Dataset generation (--auto supported)
├── 04_visualize_events.sh         # Organize + visualize events
├── 05_visualize_trajectory_v3.sh  # V3 trajectory analysis
└── config.sh                      # Shell config (reads config.py)

<robomimic_data>/<task>/ph/
├── low_dim_abs.hdf5               # Source trajectory data (absolute actions)
├── video/                         # Generated videos (Step 1)
└── kp_training/                   # Output: keypose datasets (Step 3)
    └── kp_episode_*.hdf5

data/
├── divided_events/<task>/json_sg/ # VL event JSONs (Step 1)
├── corrected_events/<task>/       # Corrected event JSONs (Step 2)
└── corrected_events_organized/    # Organized with frames (Step 4)
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| No VL JSON found | Run Step 1 first, check API key |
| Import errors | Activate conda env `tase` |
| Path errors | Check `config.py` paths |
| Slow Step 1 | Use `--auto` to skip VL entirely |

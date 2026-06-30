# Keypose Exploration: Efficient Automatic Trajectory Labelling

Accepted by IROS2026.

Automatic trajectory labelling for grasp-related robot manipulation. A
vision-language model (VLM) identifies *what* interaction events occur
(`grasp`, `release`, `detach`, `interact`, `land`) on a single representative
demonstration, and deterministic trajectory/gripper analysis pins down *when*
each event happens, yielding keypose labels without per-event hand-coded rules
or manual annotation.

## Citation

```bibtex
@article{lu2026Keypose,
  title={Keypose Exploration: Efficient Automatic Trajectory Labelling and Cross-Embodiment Policy Transfer},
  author={Lu, Yupu and Xu, Hang and Chen, Yizhou and Pan, Jia},
  journal={arXiv preprint arXiv:2606.29028},
  year={2026}
}
```

## Configuration

Paths resolve relative to this directory — set `KEYPOSE_LABELLING_ROBOMIMIC_ROOT`
to your robomimic dataset (the default looks for `../../datasets/robomimic`).
Override with environment variables if your data lives elsewhere:

```bash
export KEYPOSE_LABELLING_ROBOMIMIC_ROOT=/path/to/robomimic   # input datasets
export KEYPOSE_LABELLING_DATA_ROOT=/path/to/generated        # outputs (default: ./data)
export DASHSCOPE_API_KEY=sk-...                              # only for VL mode
python config.py square                                       # print resolved paths
```

No API key is bundled; VL mode requires `DASHSCOPE_API_KEY` in the environment.

## Quick Start

```bash
# Auto mode — trajectory-only, no VL needed (fast, recommended for iteration)
cd scripts
./run_full_pipeline.sh --auto can 200           # Single task
./run_full_pipeline.sh --auto all               # All 3 tasks at once
./correct_all_tasks.sh --auto                   # Same as above (batch helper)

# VL mode — full quality (requires DASHSCOPE_API_KEY, slow)
./run_full_pipeline.sh can 50                   # Full pipeline: Steps 1→2→3
./correct_all_tasks.sh                          # All tasks: Steps 2→3 (assumes Step 1 done)

# Test single episode or specific episodes
./02_correct_events.sh square 1 "16" qwen3-vl-235b-a22b-thinking       # Episode 16 only
./03_generate_dataset.sh square 1 "16"      # Generate dataset for episode 16
./03_generate_dataset.sh --auto can 200     # Auto mode, Step 3 directly

# Visualize trajectories
./visualize_all_tasks.sh 5  # Visualize first 5 episodes of each task
./05_visualize_trajectory_v3.sh square 16   # Visualize episode 16 only
```

## 📂 Directory Structure

```
keypose_labelling/
├── config.py                       # Central configuration (paths resolve from here)
├── divide_video.py                 # VL analysis: HDF5 → video → events JSON
├── gen_dataset.py                  # Generate keypose dataset (VL mode or --auto)
├── merge_trajectory_vl.py          # Debug/validate a single episode
├── hdf5_to_video.py                # Render demo videos from an HDF5 dataset
├── src/                            # Core labelling modules
│   ├── trajectory_segmentation_v3.py  # Per-axis motion segmentation (single source)
│   ├── event_corrector_v3.py       # Semantic-motion event correction
│   ├── trajectory_utils.py         # Trajectory loading utilities
│   └── vl_parser.py                # VL JSON parser
├── task/                           # Dataset/task adapters
│   ├── robomimic.py                # robomimic task configs
│   └── utils.py
├── viz/                            # Visualization utilities
│   ├── visualize_trajectory_v3.py  # Segmentation + correction timeline figure
│   └── reorganize_and_visualize.py # Reorganize events + extract event frames
├── scripts/                        # Shell pipeline drivers (*.sh)
│   ├── run_full_pipeline.sh        # Top-level orchestrator (--auto supported)
│   ├── correct_all_tasks.sh        # Batch all tasks (--auto supported)
│   ├── 01_prepare_videos.sh        # Step 1: HDF5 → video + VL analysis
│   ├── 02_correct_events.sh        # Step 2: trajectory event correction
│   ├── 03_generate_dataset.sh      # Step 3: generate training HDF5 (--auto)
│   ├── 04_visualize_events.sh      # Visualize events with frames
│   ├── 05_visualize_trajectory_v3.sh  # Visualize trajectory analysis
│   └── config.sh                   # Shell config (reads config.py)
├── tests/                          # Tests + parameter-analysis tools
│   ├── test_v3_segmentation.py
│   ├── test_correction_rules.py
│   ├── optimize_thresholds.py      # velocity-distribution analysis for theta^k
│   └── analyze_window_size.py      # motion-scale analysis for window w
└── data/                           # Generated outputs (git-ignored; created at runtime)
    ├── divided_events/{task}/json_sg/       # VL model analysis outputs
    ├── corrected_events/{task}/             # Trajectory-corrected events
    └── corrected_events_organized/{task}/episode_N/
        ├── events.json             # Corrected events
        ├── intervals.json          # Trajectory intervals
        ├── correction_flow.txt     # Human-readable log
        └── frames/                 # Event frame images
```

## 📋 Pipeline Overview

```
Raw HDF5 Dataset (robomimic, low_dim_abs.hdf5)
         │
    ┌────┴───────────────────────────────────────────┐
    │ VL Mode (default)                              │ Auto Mode (--auto)
    │                                                │
    │ run_full_pipeline.sh <task> <N>                │ run_full_pipeline.sh --auto <task> [N]
    │                                                │
    │ [1] 01_prepare_videos.sh                       │ (skipped — no VL needed)
    │      → HDF5 → video → VL analysis             │
    │      → data/divided_events/                    │
    │ [2] 02_correct_events.sh                       │ (skipped — no VL needed)
    │      → Match VL events to trajectory           │
    │      → data/corrected_events/                  │
    │ [3] 03_generate_dataset.sh                     │ [3] 03_generate_dataset.sh --auto
    │      → gen_dataset.py (uses VL events)         │     → gen_dataset.py --auto
    │                                                │       (gripper-based pseudo-events)
    └────┬───────────────────────────────────────────┘
         │
    Keypose Training Dataset (kp_episode_*.hdf5)
         │
    [Optional] 04/05 Visualization scripts
```

**When to use which mode:**
- **Auto mode**: Fast iteration, no API costs, good for initial experiments
- **VL mode**: Production quality, semantic event understanding, better for complex tasks

## 🔑 Key Features

- **VL-Guided**: Uses vision-language models (Qwen, GPT-4V) for event detection
- **Robust Matching**: Handles multi-piece tasks, attach scenarios (hang/drop)
- **Dual Pose Extraction**: Captures both end-effector pose (epos) and joint positions (qpos)
- **Gripper Smoothing**: Prevents fragmentation of continuous gripper actions
- **Production Ready**: Batch processing with comprehensive error handling

## 📂 Core Files

### Configuration
- `config.py` - Single source of truth for all paths and settings
- `scripts/config.sh` - Shell config (reads from config.py)

### Shell Scripts (`scripts/`)
- `run_full_pipeline.sh` - **Top-level entry point** (`--auto` to skip VL Steps 1-2)
- `correct_all_tasks.sh` - Batch all 3 tasks (`--auto` supported)
- `01_prepare_videos.sh` - Step 1: HDF5 → video + VL analysis (expensive)
- `02_correct_events.sh` - Step 2: VL event correction
- `03_generate_dataset.sh` - Step 3: Generate training HDF5 (`--auto` supported)
- `config.sh` - Shell config (reads from config.py)

### Main Python Scripts (Root)
- `divide_video.py` - VL analysis (video → events JSON)
- `gen_dataset.py` - Generate keypose datasets (VL mode or `--auto` mode)
- `merge_trajectory_vl.py` - Debug/validate single episode
- `hdf5_to_video.py` - Render demo videos from an HDF5 dataset

### Core Modules (`src/`)
- `trajectory_segmentation_v3.py` - Per-axis motion analysis (single source of truth)
- `event_corrector_v3.py` - Context-aware event matching
- `trajectory_utils.py` - Load trajectory from HDF5 (epos + qpos)
- `vl_parser.py` - Parse VL JSON output

### Visualization (`viz/`)
- `visualize_trajectory_v3.py` - Segmentation + correction timeline figure (imports from `src/`)
- `reorganize_and_visualize.py` - Reorganize corrected events into per-episode folders + extract event frames

### Tests (`tests/`)
- `test_v3_segmentation.py` - Segmentation smoke test
- `test_correction_rules.py` - Validation tests for event correction
- `optimize_thresholds.py` - Analyze per-axis velocity distributions to inform the per-task thresholds $\theta^k$
- `analyze_window_size.py` - Analyze trajectory motion scales to inform the smoothing window $w$

## 📊 Data Structures

### Trajectory (End-Effector Pose)
```python
epos: (T, 9) array
    [pos_x, pos_y, pos_z,           # Position (3)
     quat_w, quat_x, quat_y, quat_z, # Orientation (4)
     gripper_left, gripper_right]    # Gripper joints (2)

qpos: (T, 7) array
    [joint_1, joint_2, ..., joint_7] # Robot joint positions
```

### Keyposes Dictionary
```python
keyposes = {
    'frames': [10, 45, 89],              # Frame indices
    'epos': [[x,y,z,qw,qx,qy,qz,g1,g2],  # End-effector poses (K, 9)
             [...],
             [...]],
    'qpos': [[j1,j2,j3,j4,j5,j6,j7],     # Joint positions (K, 7)
             [...],
             [...]],
    'event_types': ['grasp', 'release', 'attach_hang']
}
```

### Action Gripper
```python
action_gripper: (T,) array
    -1 = open gripper
     1 = close gripper
     0 = no action (stationary)
```

## 🎯 Event Types

### Interaction Events
- **`grasp`**: Robot closes gripper on object
- **`release`**: Robot opens gripper to release object
- **`detach`**: Object separates from surface/fixture
- **`attach_hang`**: Object contacts target while held (hang scenario)
- **`attach_drop`**: Object contacts surface after release (drop scenario)

### Attach Scenarios

#### attach_hang (before_release)
```
Timeline: grasp → move → [ATTACH] → release
Example: Hang tool on pegboard
Match to: LAST moving interval before release
```

#### attach_drop (after_release)
```
Timeline: grasp → move → release → [ATTACH]
Example: Drop can into bin
Match to: STABLE interval after opening
```

## 🔧 Common Tasks

### Generate Keypose Dataset
```bash
# Preferred: use the top-level pipeline script
cd scripts
./run_full_pipeline.sh --auto can 200          # Auto mode (fast, no VL)
./run_full_pipeline.sh can 50                  # VL mode (full quality)

# Or call gen_dataset.py directly
python gen_dataset.py \
    --hdf5_path ./data/robomimic/datasets/can/ph/low_dim_abs.hdf5 \
    --output_dir ./data/robomimic/datasets/can/ph/kp_training \
    --num_episodes 200 --task can --auto

# VL mode (requires corrected events from Steps 1-2)
python gen_dataset.py \
    --hdf5_path ./data/robomimic/datasets/can/ph/low_dim_abs.hdf5 \
    --vl_json_dir ./data/corrected_events/can \
    --output_dir ./data/robomimic/datasets/can/ph/kp_training \
    --num_episodes 50 --task can
```

### Debug Single Episode
```bash
python merge_trajectory_vl.py \
    --hdf5_path ./data/robomimic/datasets/can/ph/low_dim_abs.hdf5 \
    --vl_json ./keypose/corrected_events/can/json_sg/imglist_episode_0_*.json \
    --episode 0 \
    --output_dir ./debug/
```

### VL Analysis on Videos
```bash
python divide_video.py \
    --dataset_folder ./data/robomimic/datasets/can/ph/ \
    --task can \
    --model max \
    --target_fps 10
```

## 🛠️ Tuning Parameters

### Trajectory Segmentation
```python
segment_trajectory(
    trajectory,
    action_gripper,
    position_threshold=5e-3,    # 5mm position change
    rotation_threshold=1e-2,    # ~0.6° rotation change
    gripper_threshold=1e-4,     # Gripper joint change
    stable_window=5,            # 5 timesteps stable
    max_gap=5                   # Fill gaps ≤5 timesteps
)
```

**Key Parameter:** `max_gap`
- **Larger (10)**: More aggressive smoothing, connects distant intervals
- **Smaller (1)**: Conservative smoothing, only fill single gaps
- **Default (5)**: Good balance for most tasks

## 📦 Output Dataset

```
keypose_dataset.hdf5
  ├── obs/
  │   ├── env_obs            # (T, 23) environment obs [object(14), robot(9)]
  │   ├── epos               # (T, 9) end-effector pose
  │   ├── qpos               # (T, 7) joint positions (optional)
  │   ├── prev_keypose       # (T, 9) previous keypose epos
  │   └── images/            # Camera views (if enabled)
  ├── target/
  │   ├── next_keypose       # (T, 9) next keypose epos
  │   ├── event_type         # (T,) event type indices
  │   ├── keypose_frames     # (K,) keypose frame indices
  │   ├── keypose_epos       # (K, 9) keypose end-effector poses
  │   ├── keypose_qpos       # (K, 7) keypose joint positions (optional)
  │   └── keypose_types      # (K,) event type strings
  └── actions                # (T, action_dim) robot actions [INCLUDED]
```

## ❓ FAQ

### Why did qpos become epos?
Clarity! In robomimic datasets:
- `qpos` = joint positions (7D for 7-DOF arm)
- `epos` = end-effector pose (9D: position + quaternion + gripper)

We use end-effector data for keyposes, so `epos` is the correct term.

### Do we smooth gripper actions?
YES! Via `_fill_gripper_gaps()` in `trajectory_segmentation.py`. This prevents breaking continuous gripper actions into fragments.

### What if multiple closing intervals exist?
Handled chronologically. First grasp → first closing, second grasp → second closing, etc.

### How is VL video analysis run?
Use `divide_video.py` (subtask-based, uses the Qwen3-VL model) — it produces the per-episode event JSON consumed by the correction step.

### Object slipping from gripper?
NOT handled. Assumption: clean grasp → release → grasp sequences. VL should output detach events for slips.

## 📚 Documentation

- `scripts/README.md` - Demo scripts usage guide

---

*Last Updated: June 27, 2026*

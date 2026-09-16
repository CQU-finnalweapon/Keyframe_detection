# Keypose Exploration: Efficient Automatic Trajectory Labelling

Accepted by IROS2026.

Automatic trajectory labelling for grasp-related robot manipulation. A
vision-language model (VLM) identifies *what* interaction events occur
(`grasp`, `release`, `detach`, `interact`, `land`) on a single representative
demonstration, and deterministic trajectory/gripper analysis pins down *when*
each event happens, yielding keypose labels without per-event hand-coded rules
or manual annotation.

> **Repository status.** This is a working copy of the IROS 2026 *Keypose
> Exploration* codebase (MIT, © 2026 Yupu Lu), extended with an
> annotation-driven LeRobot pipeline, review tooling (figures + annotated videos)
> and a synthetic-signal test suite.  The original LICENSE and citation are kept
> intact; the additions are described under [Two workflows](#two-workflows) below.

## Two workflows

The *what / when* split is the same in both; only the source of the semantic
stage differs.

| | **robomimic mode** | **LeRobot mode** |
|---|---|---|
| Semantic stage ("what, roughly when") | VLM analyses the demo video | hand-annotated subtasks in `dataset.json` |
| Motion data | robomimic HDF5 | LeRobot parquet (per-arm gripper + cartesian state) |
| API key | needed for VL mode (`DASHSCOPE_API_KEY`) | **none** |
| Entry point | `scripts/run_full_pipeline.sh <task> <N>` | `scripts/06_generate_lerobot_dataset.sh <dataset> <out> <N>` |
| Detail | below | **[LEROBOT_PIPELINE.md](LEROBOT_PIPELINE.md)** |

Both modes share the deterministic core (`src/trajectory_segmentation_v3.py`):
savitzky-golay smoothing, per-axis velocity thresholding, per-axis motion
signatures, direction-reversal aware phase merging, and gripper close/open
interval extraction.  New parameters added for LeRobot all default to the
original robomimic behaviour, so the two paths cannot drift apart.

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

### LeRobot datasets — annotated subtasks, no VLM at all

Use this when the dataset already carries subtask annotations (LeRobot `v2.1`
layout with `dataset.json` + per-episode parquet).  No API key, no video
analysis; the gripper signal comes from the parquet.

```bash
cd scripts

# Full run over N episodes
./06_generate_lerobot_dataset.sh /path/to/PickPlaceONLY_Moz1_cjb ./data/lerobot_kp 200

# Review: figures (per episode + overview sheet) and annotated videos
python ../viz/visualize_lerobot_episode.py --dataset_root /path/to/dataset --num_episodes 20
python ../viz/render_lerobot_video.py     --dataset_root /path/to/dataset --num_episodes 20

# Validate: 7 physical checks + failure-mode breakdown
python ../tests/evaluate_lerobot_keyposes.py   --dataset_root /path/to/dataset --num_episodes 600
python ../tests/diagnose_lerobot_matching.py   --dataset_root /path/to/dataset --num_episodes 200
```

### robomimic datasets — VLM mode or auto mode

```bash
cd scripts
# Auto mode — trajectory-only, no VL needed (fast, recommended for iteration)
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
├── gen_dataset_lerobot.py          # LeRobot mode: subtask annotations → keyposes (no VLM)
├── merge_trajectory_vl.py          # Debug/validate a single episode
├── hdf5_to_video.py                # Render demo videos from an HDF5 dataset
├── LEROBOT_PIPELINE.md             # LeRobot mode: data contract, rules, metrics
├── src/                            # Core labelling modules
│   ├── trajectory_segmentation_v3.py  # Per-axis motion + gripper segmentation
│   ├── event_corrector_v3.py       # Semantic-motion event correction
│   ├── lerobot_utils.py            # LeRobot adapter (parquet, annotations, profiles)
│   ├── subtask_keypose.py          # Annotation-driven keypose extraction
│   ├── trajectory_utils.py         # Trajectory loading utilities
│   └── vl_parser.py                # VL JSON parser
├── task/                           # Dataset/task adapters
│   ├── robomimic.py                # robomimic task configs
│   └── utils.py
├── viz/                            # Visualization utilities
│   ├── visualize_trajectory_v3.py  # Segmentation + correction timeline figure
│   ├── visualize_lerobot_episode.py   # LeRobot: diagnostic figure per episode (+ overview sheet)
│   ├── render_lerobot_video.py        # LeRobot: review video (camera + live keypose overlay)
│   └── reorganize_and_visualize.py # Reorganize events + extract event frames
├── scripts/                        # Shell pipeline drivers (*.sh)
│   ├── run_full_pipeline.sh        # Top-level orchestrator (--auto supported)
│   ├── correct_all_tasks.sh        # Batch all tasks (--auto supported)
│   ├── 01_prepare_videos.sh        # Step 1: HDF5 → video + VL analysis
│   ├── 02_correct_events.sh        # Step 2: trajectory event correction
│   ├── 03_generate_dataset.sh      # Step 3: generate training HDF5 (--auto)
│   ├── 04_visualize_events.sh      # Visualize events with frames
│   ├── 05_visualize_trajectory_v3.sh  # Visualize trajectory analysis
│   ├── 06_generate_lerobot_dataset.sh # LeRobot mode: annotation-driven keyposes
│   └── config.sh                   # Shell config (reads config.py)
├── tests/                          # Tests + parameter-analysis tools
│   ├── test_v3_segmentation.py
│   ├── test_correction_rules.py
│   ├── test_lerobot_gripper_segmentation.py  # Synthetic-signal unit tests (no dataset)
│   ├── evaluate_lerobot_keyposes.py          # 7-point physical validation
│   ├── diagnose_lerobot_matching.py          # Failure-mode breakdown
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
                    which dataset?
        ┌───────────────────┴────────────────────┐
        │                                        │
  robomimic HDF5                     LeRobot parquet + subtask annotations
        │                                        │
  ┌─────┴──────────────┐              gen_dataset_lerobot.py
  │ VL mode   │ --auto │              (annotation-driven, no VLM, no API key)
  └─────┬──────┴───┬───┘                         │
        │          │                             │
  [1] video → VL   │ (skipped)                   │
  [2] event        │ (skipped)                   │
      correction   │                             │
  [3] gen_dataset.py ───────────────────────────┘
        │                                        │
        └───────────────┬────────────────────────┘
                        ▼
          Keypose Training Dataset (kp_episode_*.hdf5)
                        ▼
        [Optional] figures + annotated review videos
```

robomimic branch in detail:

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
- **LeRobot mode**: the dataset already ships annotated subtasks — skip the VLM entirely
- **Auto mode**: robomimic data, fast iteration, no API costs, good for initial experiments
- **VL mode**: robomimic data, production quality, semantic event understanding, better for complex tasks

### LeRobot mode (Moz1 datasets, no VLM at all)

LeRobot datasets recorded on Moz1 ship with **hand-annotated subtasks** in
`dataset.json`, which play exactly the role of the VL output ("what happens,
roughly when").  The semantic stage is therefore free and the whole VLM half of
the pipeline is skipped (see Quick Start for the commands).

The gripper signal lives in the episode **parquet** files
(`<arm>_gripper_state_pos` / `<arm>_gripper_cmd_pos`), not in the videos, and the
deterministic analysis turns each annotated window into an exact keypose frame.
On `PickPlaceONLY_Moz1_cjb` (600 episodes) this reaches a 100% subtask match rate
with every keypose landing inside its annotated window and on a real gripper
transition.

See **[LEROBOT_PIPELINE.md](LEROBOT_PIPELINE.md)** for the data contract, the
matching rules, the tuning knobs and the measured metrics.

## 🔑 Key Features

- **Annotation-driven (no VLM)**: LeRobot datasets with annotated subtasks skip the
  whole VLM half of the pipeline — deterministic, reproducible, no API key
- **VL-Guided**: Uses vision-language models (Qwen, GPT-4V) for event detection on
  datasets without annotations
- **Physics-checked matching**: a subtask can only consume one gripper action, so the
  gripper signal overrules contradictory annotation text (wrong hand, pick labelled as
  place, collapsed pick+place spans)
- **Robust Matching**: Handles multi-piece tasks, attach scenarios (hang/drop)
- **Dual Pose Extraction**: Captures both end-effector pose (epos) and joint positions (qpos)
- **Gripper Smoothing**: Prevents fragmentation of continuous gripper actions, merges
  two-stage closures and slow sub-threshold squeezes
- **Review tooling**: per-episode diagnostic figures, overview sheets, and annotated
  review videos with a live keypose dashboard
- **Production Ready**: Batch processing with comprehensive error handling

## 📂 Core Files

### Configuration
- `config.py` - Single source of truth for all paths and settings
- `scripts/config.sh` - Shell config (reads from config.py)

### Shell Scripts (`scripts/`)
- `run_full_pipeline.sh` - **Top-level entry point** for robomimic (`--auto` to skip VL Steps 1-2)
- `correct_all_tasks.sh` - Batch all 3 tasks (`--auto` supported)
- `01_prepare_videos.sh` - Step 1: HDF5 → video + VL analysis (expensive)
- `02_correct_events.sh` - Step 2: VL event correction
- `03_generate_dataset.sh` - Step 3: Generate training HDF5 (`--auto` supported)
- `06_generate_lerobot_dataset.sh` - **LeRobot entry point**: annotation-driven keyposes
- `config.sh` - Shell config (reads from config.py)

### Main Python Scripts (Root)
- `divide_video.py` - VL analysis (video → events JSON)
- `gen_dataset.py` - Generate keypose datasets (VL mode or `--auto` mode)
- `gen_dataset_lerobot.py` - Generate keypose datasets from LeRobot annotations (no VLM)
- `merge_trajectory_vl.py` - Debug/validate single episode
- `hdf5_to_video.py` - Render demo videos from an HDF5 dataset

### Core Modules (`src/`)
- `trajectory_segmentation_v3.py` - Per-axis motion + gripper segmentation (single source of truth)
- `event_corrector_v3.py` - Context-aware event matching
- `lerobot_utils.py` - LeRobot adapter: parquet reader, subtask annotations, tuned profiles
- `subtask_keypose.py` - Annotation-driven keypose extraction (replaces VLM + corrector)
- `trajectory_utils.py` - Load trajectory from HDF5 (epos + qpos)
- `vl_parser.py` - Parse VL JSON output

### Visualization (`viz/`)
- `visualize_trajectory_v3.py` - Segmentation + correction timeline figure (imports from `src/`)
- `visualize_lerobot_episode.py` - LeRobot diagnostic figure per episode, plus a batch overview sheet
- `render_lerobot_video.py` - LeRobot review video: camera feed + live keypose dashboard, gripper zoom, timeline
- `reorganize_and_visualize.py` - Reorganize corrected events into per-episode folders + extract event frames

### Tests (`tests/`)
- `test_v3_segmentation.py` - Segmentation smoke test
- `test_correction_rules.py` - Validation tests for event correction
- `test_lerobot_gripper_segmentation.py` - 34 synthetic-signal unit tests (no dataset needed)
- `evaluate_lerobot_keyposes.py` - 7-point physical validation over a batch
- `diagnose_lerobot_matching.py` - Buckets every mismatch by failure mode
- `optimize_thresholds.py` - Analyze per-axis velocity distributions to inform the per-task thresholds $\theta^k$
- `analyze_window_size.py` - Analyze trajectory motion scales to inform the smoothing window $w$

`test_lerobot_gripper_segmentation.py` also runs standalone (`python tests/test_lerobot_gripper_segmentation.py`),
which is handy because plain `pytest` walks up to `/mnt` on this cluster and trips over
unreadable mounts; use `pytest -c /dev/null --rootdir=. <file>` if you prefer pytest.

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

### Generate Keyposes from a LeRobot Dataset
```bash
# Whole dataset / a number of episodes / a list
./scripts/06_generate_lerobot_dataset.sh /path/to/PickPlaceONLY_Moz1_cjb ./data/lerobot_kp 200

python gen_dataset_lerobot.py \
    --dataset_root /path/to/PickPlaceONLY_Moz1_cjb \
    --output_dir ./data/lerobot_kp \
    --episode_list 0,3,7 --verbose

# Tuning overrides (see LEROBOT_PIPELINE.md for the full list)
python gen_dataset_lerobot.py --dataset_root ... --output_dir ... \
    --threshold_x 0.025 --threshold_y 0.025 --threshold_z 0.018 \
    --window_size 41 --search_radius 15
```

### Review a LeRobot Run
```bash
# one figure per episode + a combined overview sheet
python viz/visualize_lerobot_episode.py --dataset_root ... --num_episodes 20

# annotated review video: camera feed + live keypose dashboard
python viz/render_lerobot_video.py --dataset_root ... --num_episodes 20
#   --cameras cam_high,cam_left_wrist,cam_right_wrist   multiple views side by side
#   --hold_frames 10 --speed 0.5                        pause on each keypose

# batch validation and failure breakdown
python tests/evaluate_lerobot_keyposes.py --dataset_root ... --num_episodes 600 --report eval.csv
python tests/diagnose_lerobot_matching.py --dataset_root ... --num_episodes 200
python tests/test_lerobot_gripper_segmentation.py      # 34 unit tests, no dataset
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

### LeRobot Profiles

LeRobot runs are configured by a named profile in `src/lerobot_utils.py`
(`LEROBOT_PROFILES['moz1']`) rather than by the robomimic defaults above, because
real-hardware grippers need different handling:

| key | meaning |
|---|---|
| `thresholds` | per-axis velocity threshold (m/s) for the moving/stable decision |
| `window_size` | savitzky-golay smoothing window (frames, odd) |
| `merge_hiccup_gaps` | merge `close → brief open → close` two-stage closures |
| `interrupted_action_gap` | merge a same-direction action split by a slow, sub-threshold plateau |
| `min_short_travel` | travel (m) that rescues a too-short interval (a one-frame snap release is real) |
| `mask_whole_command_run` | `False` here: the gripper command is a continuous setpoint that lags the state |
| `min_movement_between` | `0` disables the "correction motion" heuristic, which would delete real grasps that happen without arm travel |

All of these default to the original robomimic behaviour when not set, so the shared
`segment_trajectory_v3()` was verified to produce byte-identical results on 360
random synthetic robomimic-style inputs.  Full table in
**[LEROBOT_PIPELINE.md](LEROBOT_PIPELINE.md#5-调参入口)**.

## 📦 Output Dataset

### robomimic mode

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

### LeRobot mode

```
<output_dir>/
  ├── kp_episode_<i>.hdf5            # training tensors (one file per episode)
  ├── events_episode_<i>.json        # every event + per-subtask diagnostics
  ├── timelines_episode_<i>.txt      # human-readable segmentation timelines
  └── summary.csv                    # per-episode match rate

kp_episode_<i>.hdf5
  ├── obs/
  │   ├── epos              # (T, 9) end-effector pose of the *active* arm
  │   ├── left_epos         # (T, 9)
  │   ├── right_epos        # (T, 9)
  │   ├── prev_keypose      # (T, 9) last reached keypose
  │   ├── active_arm        # (T,)  0 = left, 1 = right
  │   └── subtask_index     # (T,)  annotated subtask active at t
  ├── target/
  │   ├── next_keypose      # (T, 9) next keypose to reach
  │   ├── event_type        # (T,) 0 none / 1 grasp / 2 release / 3 detach / 4 attach_drop
  │   ├── stage_index       # (T,) which keypose is being pursued, in [0, K]
  │   ├── keypose_frames    # (K,)
  │   ├── keypose_epos      # (K, 9)
  │   ├── keypose_qpos      # (K, 7)
  │   ├── keypose_types     # (K,) strings
  │   └── keypose_arm       # (K,) 'left' / 'right' per keypose
  ├── actions/              # <arm>_cmd_cart_pos, <arm>_cmd_joint_pos,
  │                         # <arm>_gripper_cmd_pos, vector (T, 28)
  └── meta/                 # attrs: fps, num_frames, num_keyposes, arms, arm_ids
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

### Where does the gripper signal come from in LeRobot mode?
From the episode **parquet**, not the video: `<arm>_gripper_state_pos` for the
measurement and `<arm>_gripper_cmd_pos` to detect frames where the command
contradicts the measurement.  Video frame *i* and parquet row *i* are aligned
(both at the dataset fps), so no resampling is involved.

### What if the annotation text contradicts the gripper?
The gripper wins.  A subtask can physically consume only one gripper action, so a
`pick` window holding only an `opening` becomes a `release`, a window naming the
wrong hand switches arms, and an unannotated span keeps every action it contains.
Every such correction is recorded in `events_episode_*.json`
(`kind_source` / `arm_source` / `message`) so it stays auditable.

## 📚 Documentation

- `LEROBOT_PIPELINE.md` - LeRobot mode: data contract, matching rules, tuning knobs, measured metrics
- `scripts/README.md` - Demo scripts usage guide

---

*Last Updated: September 16, 2026*

# LeRobot 数据集关键帧提取（跳过 VLM）

本目录的原始流程面向 robomimic HDF5 + VLM 输出：VLM 负责回答"发生了什么事件、大致在哪一帧"，
轨迹分析负责把时间点精确到帧。

Moz1 采集的 LeRobot 数据集**已经带有人工子任务标注**（`dataset.json` 里
`episodes[i].annotation.annotation_data.data`），语义阶段相当于已经做完了，
因此可以**完全跳过 `divide_video.py`（VLM）和 `event_corrector_v3.py`**，
直接进入确定性的动作分析阶段。

```
robomimic 流程:  HDF5 ──► 视频 ──► VLM 事件(粗) ──┐
                                                  ├──► 事件校正 ──► 关键帧
                 轨迹分段 ─────────────────────────┘

LeRobot 流程:    parquet 夹爪/位姿 ──► 轨迹分段 ──┐
                 子任务标注(已有) ────────────────┴──► 关键帧      ← 无 API 依赖
```

---

## 1. 数据集格式要求

```
<dataset_root>/
├── dataset.json                        # 必需：元信息 + 每个 episode 的内联子任务标注
├── annotations.json                    # 可选：按 recording_id 索引的同一份标注
├── data/chunk-000/episode_000000.parquet
└── videos/chunk-000/cam_high/episode_000000.mp4   # 本流程不需要
```

### 从 parquet 读取的信号

每个 episode 一个 parquet，按手臂前缀 `leftarm_*` / `rightarm_*`：

| 列 | 形状 | 用途 |
|---|---|---|
| `<arm>_state_cart_pos` | (T, 6) | 末端位姿 `x,y,z,rx,ry,rz` → `pos(3)` + 由欧拉角转出的 `quat(4)` |
| `<arm>_gripper_state_pos` | (T, 1) | **夹爪开合信号，关键帧定位的主依据**（0.098=全开，≈0.001=全闭） |
| `<arm>_gripper_cmd_pos` | (T, 1) | 夹爪指令，用于剔除"指令与实测矛盾"的帧 |
| `<arm>_state_joint_pos` | (T, 7) | 关节角 → `keypose_qpos` |
| `<arm>_cmd_cart_pos` / `<arm>_cmd_joint_pos` | (T, 6/7) | 动作标签 |

`src/lerobot_utils.py` 把它们拼成项目内部约定的 `epos (T, 9) = [pos(3), quat(4), gripper(1), 0]`
以及 `action_gripper (T,)`，这样 `trajectory_segmentation_v3` 完全不需要改动。

### 子任务标注

```jsonc
{
  "episode_index": 0,
  "length": 1413,
  "annotation": {
    "annotation_data": {
      "data": [
        { "start_time": 0.0, "end_time": 5.06,
          "action": "右手拿起插头 在桌子上",
          "action_english": "Pick up the plug on the table with right hand" },
        { "start_time": 5.06, "end_time": 11.59,
          "action_english": "Place the plug on the glass" }
      ]
    }
  }
}
```

标注只提供**文本语义 + 时间窗口**（秒），脚本按 `fps` 换算成帧窗口
（`start_frame = round(start_time * fps)`，窗口左闭右开）。

---

## 2. 提取逻辑

### 2.1 意图与手臂解析

`classify_subtask()` 用中英文关键词把文本判为 `pick` / `place` / `other`
（`拿起/pick/grasp` 与 `放在/put/place`）。
`resolve_arms()` 按优先级决定每条子任务由哪只手执行：

1. 文本里明确写了手 → `text`；
2. 继承上一条已知手臂 → `prev`（"Place X on Y" 通常不重复写手）；
3. 继承下一条已知手臂 → `next`；
4. 窗口内运动/夹爪活跃度 → `activity`。

### 2.2 夹爪时间线（关键帧的物理依据）

`segment_trajectory_v3()` 对每条手臂独立跑一遍：

1. **差分 + 阈值**：`gripper_threshold = 1e-3`，把逐帧变化分成 `closing` / `opening` / `none`；
2. **指令校验**：`_filter_gripper_labels()` 只屏蔽**确实与指令矛盾**的帧
   （实测在开、指令说闭），不会像旧实现那样把整段指令周期都丢掉
   （由 `mask_whole_command_run=False` 开启；robomimic 仍用旧语义）；
3. **填补短洞**：`_fill_gripper_gaps(max_gap=5)`；
4. **`merge_gripper_hiccups`**：合并 `闭→短暂开→闭` 的两段式夹取；
5. **`merge_interrupted_actions`**：合并被"低于阈值的平台期"打断的同一方向动作。
   真实夹爪一次缓慢闭合常花几十帧低于逐帧阈值，不合并会被切成 3 次假动作；
   判据是**跨越空隙时信号方向未反转**（这是安全的：真正的反转会产生一个反标签区间，
   而本函数只合并相邻的同标签区间）；
6. **`filter_gripper_intervals`**：丢弃过短区间与"修正动作"对。
   短区间若**幅度足够大**（`min_short_travel=5e-3`）则保留 —— 单帧从全闭弹到全开是真实释放；
7. 丢掉 robomimic 用的末尾哨兵区间（`sentinel_tail=False`）。

同时产出运动时间线（逐轴速度 + 每轴阈值 → 逐段运动指纹），用于 `detach` 定位。

### 2.3 子任务 → 关键帧匹配

`assign_intervals()`：**全局**把每个夹爪区间分配给"起点落在其窗口内"的那条子任务。
按归属而不是"谁最近谁抢"，避免放置子任务偷走下一个子任务的释放动作。

`extract_key_events()` 对每条子任务：

| 情况 | 行为 |
|---|---|
| `pick` | 取窗口内**第一个 `closing`** → `grasp`（帧 = 区间起点 − 1，即闭合动作开始的那一帧） |
| `place` | 取窗口内**第一个 `opening`** → `release` |
| 文本与物理矛盾 | `resolve_intent()` **相信夹爪**：`pick` 窗口只有 `opening` → 判为 `release` |
| 窗口内是完整开合循环 | 可选 `--split_cycles` 拆成两个事件（默认关闭，见 §4） |
| 文本为空（`无法标注`） | 保留窗口内**全部**夹爪动作 |
| 指定手臂窗口为空 | `find_other_arm_evidence()` 换到另一只手臂并**重新推断意图** |
| 附加事件 | `detach`（grasp 之后的第一次运动）/ `attach_drop`（release 结尾） |

最后强制关键帧严格递增，并输出 `grasp` / `release` 作为关键帧（`detach` 为次要事件）。

---

## 3. 使用

```bash
cd keypose_labelling

# 单集，带逐子任务明细
python gen_dataset_lerobot.py \
    --dataset_root /path/to/PickPlaceONLY_Moz1_cjb \
    --output_dir ./data/lerobot_kp \
    --episode 0 --verbose

# 批量（0..N-1）
python gen_dataset_lerobot.py --dataset_root ... --output_dir ... --num_episodes 200

# 指定若干集
python gen_dataset_lerobot.py --dataset_root ... --output_dir ... --episode_list 0,3,7

# 或使用封装脚本
./scripts/06_generate_lerobot_dataset.sh /path/to/dataset ./data/lerobot_kp 200
```

### 评测与可视化

```bash
# 7 项物理校验（批处理）
python tests/evaluate_lerobot_keyposes.py \
    --dataset_root /path/to/dataset --num_episodes 600 \
    --report ./data/eval.csv

# 失败模式分桶诊断
python tests/diagnose_lerobot_matching.py \
    --dataset_root /path/to/dataset --num_episodes 200 --top 5

# 单集四点图（位姿 / 速度+阈值 / 夹爪 / 时间线）
python viz/visualize_lerobot_episode.py --dataset_root /path/to/dataset --episode 0

# 批量：每集一张详细图，外加一张总览长图
python viz/visualize_lerobot_episode.py --dataset_root /path/to/dataset \
    --num_episodes 20 --output_dir ./data/lerobot_viz
python viz/visualize_lerobot_episode.py --dataset_root /path/to/dataset \
    --episode_list 0,71,408,495,545 --output_dir ./data/lerobot_viz

# 合成信号单元测试（不需要数据集）
python tests/test_lerobot_gripper_segmentation.py
```

### 可视化输出

* `episode_<i>.png` —— 四层子图：
  1. 双臂末端位置 + 子任务窗口 + 关键帧标记（▼grasp / ▲release）；
  2. 逐轴速度 vs 阈值（虚线）+ 分段出的 moving 区间（灰底）；
  3. **夹爪状态/指令**（实线=实测，点线=指令）+ 检测到的 closing（绿底）/ opening（红底）；
  4. 时间线长条：pick/place 子任务、检测到的事件、关键帧刻度。
* `overview_<N>episodes.png` —— 批量时的总览长图，每集一行，只画双臂夹爪曲线 +
  子任务底色 + 关键帧竖线，用于快速扫一遍整批（先看总览定位可疑集，再打开单集详图）。
  可用 `--overview_cols` 调列数，`--no_overview` 关闭。

### 审阅视频（边放画面边显示关键帧）

```bash
# 单集，cam_high
python viz/render_lerobot_video.py --dataset_root /path/to/dataset --episode 0

# 批量
python viz/render_lerobot_video.py --dataset_root /path/to/dataset --num_episodes 20

# 多机位并排（检查接触/抓取最有用）
python viz/render_lerobot_video.py --dataset_root /path/to/dataset --episode 0 \
    --cameras cam_high,cam_left_wrist,cam_right_wrist

# 关键帧多停留 10 帧便于观察；半速回放
python viz/render_lerobot_video.py --dataset_root /path/to/dataset --episode 0 \
    --hold_frames 10 --speed 0.5
```

输出 `episode_<i>_review.mp4`，画面分四块：

| 区域 | 内容 |
|---|---|
| 顶部横幅 | `KEYPOSE: GRASP (right arm)` + `frame 67 (+0) subtask #0 detected here`。接近关键帧时显示 `approaching` 与帧距，无事件时显示 `no keypose nearby`；命中时整条变实色并把画面描边 |
| 左侧画面 | 相机原生分辨率（**不放大，避免浪费码率**）。多个 `--cameras` 时画布自动加宽成网格 |
| 左下「GRIPPER ZOOM」| 当前帧 ±3 s 的夹爪曲线（逐帧精度），叠加检测到的区间底色与关键帧竖线，用来直接判断"关键帧是否真的落在夹爪动作上" |
| 右侧面板 | 集号/帧号/时间、当前子任务（编号、pick/place、手臂、窗口、英文文本）、关键帧状态、双臂夹爪开合条（白线=指令）、**全部关键帧列表**、**全部子任务列表**（当前项高亮） |
| 底部时间线 | 子任务色带、逐臂夹爪区间、关键帧刻度、白色播放头、图例 |

实现要点：

* 视频帧与 parquet 行 **按帧索引 1:1 对齐**（都是数据集 fps），无需重采样；
* 用 `ffmpeg -c:v libx264` 管道编码（本机 OpenCV 的 `avc1` 缺硬件设备、`mp4v` 体积约为其 2 倍），
  默认 `--crf 22`，不可用时自动回退 `cv2.VideoWriter`；
* 相机画面**按原生分辨率渲染、不放大**，画布随相机数自适应（1–3 路单行、4 路 2×2），
  避免把码率浪费在无信息的上采样上；
* `--hold_frames` 的停顿发生在**横幅变为实色**的每一帧，即主关键帧（grasp/release）
  与次级事件（`detach`，由 `--add_detach` 控制）都会停顿；`--hold_frames 0` 即实时播放；
* 标注文本用 `action_english` 显示 —— OpenCV 的 `putText` 无法渲染中文，界面上有说明。

实测（`PickPlaceONLY_Moz1_cjb` 前 20 集，`cam_high`，`--hold_frames 10 --crf 22`）：
20 个 `episode_XXXXXX_review.mp4`，h264 / 1280×926 / 30 fps，共 35088 帧（≈19.5 分钟）、330 MB，
单集约 13–22 MB，渲染约 41 s/集（单进程）。

### 输出

```
<output_dir>/
├── kp_episode_<i>.hdf5     # 训练数据
├── events_episode_<i>.json # 事件 + 每条子任务的诊断信息
├── timelines_episode_<i>.txt
└── summary.csv             # 每集匹配率
```

HDF5 结构：

```
obs/    epos (T,9)  left_epos  right_epos  prev_keypose (T,9)
        active_arm (T,)  0=左 1=右
        subtask_index (T,)
target/ next_keypose (T,9)  event_type (T,)  0=无 1=grasp 2=release 3=detach 4=attach_drop
        stage_index (T,)  ∈[0,K] 当前在追第几个关键帧
        keypose_frames (K,)  keypose_epos (K,9)  keypose_qpos (K,7)
        keypose_types (K,)  keypose_arm (K,)
actions/ <arm>_cmd_cart_pos  <arm>_cmd_joint_pos  <arm>_gripper_cmd_pos  vector (T,28)
meta/   fps  num_frames  num_keyposes  arms  arm_ids  key_event_types
```

---

## 4. 实测结果

`PickPlaceONLY_Moz1_cjb`（1609 集，30 FPS），前 600 集：

| 指标 | 结果 |
|---|---|
| 子任务匹配率 | 5313 / 5315 = **100.0%** |
| 关键帧落在其子任务窗口内 | **100%** (5328/5328) |
| 该区间夹爪确实运动 | **100%** |
| 闭合/张开方向正确 | **100%** |
| 关键帧严格递增 | **100%** (600/600) |
| grasp/release 交替 | **99.7%** (598/600) |
| grasp 早于对应 release | **100%** |
| 关键帧距窗口起点 | 中位数 2.96 s（子任务均值 ~5.5 s，即 ~54% 处开始交互） |

### 已知边界情况

* **交替性 2 例失败**：标注窗口与夹爪事件错位（例如 "Pick up X" 的窗口里其实是放置）。
  逐条 `events_episode_*.json` 里有 `kind_source` / `arm_source` / `message` 可审计。
* **`split_cycle_windows` 默认关闭**：仅在"整集子任务全被标成 `place`"这种折叠标注下有用。
  实测 600 集中它修好 1 集、弄坏 3 集，故默认关闭；这类数据集可用 `--split_cycles` 打开。
* **`detach` 依赖运动时间线**，若该子任务内机器人几乎不动则回退到闭合区间末尾。

---

## 5. 调参入口

参数集中在 `src/lerobot_utils.py` 的 `LEROBOT_PROFILES['moz1']`：

| 键 | 值 | 含义 |
|---|---|---|
| `thresholds` | `{x:0.025, y:0.025, z:0.018}` | 逐轴速度阈值 (m/s)，用于运动/静止判定 |
| `window_size` | 41 | Savitzky-Golay 平滑窗口（帧，奇数） |
| `gripper_threshold` | 1e-3 | 夹爪逐帧变化阈值 (m) |
| `max_gap` | 5 | 夹爪标签短洞填补上限 |
| `merge_hiccup_gaps` | 6 | 两段式夹取的短暂反向合并上限 |
| `interrupted_action_gap` | 60 | 跨越平台期合并同一动作的最大空隙 |
| `min_short_travel` | 5e-3 | 救回过短区间所需的最小幅度 (m)。**函数默认 0 = 关闭** |
| `mask_whole_command_run` | False | False 只屏蔽真正矛盾的帧。**函数默认 True = 旧行为** |
| `min_movement_between` | 0 | 0 = 关闭"修正动作"启发式（该启发式面向遥操作 robomimic 数据） |
| `min_closing_duration` / `min_opening_duration` | 2 / 2 | 区间最短帧数 |
| `split_cycle_windows` | False | 是否拆分折叠的 pick+place 窗口 |

命令行可临时覆盖：`--threshold_x/y/z`、`--window_size`、`--search_radius`、`--split_cycles`。

### 与 robomimic 流程的兼容性

`trajectory_segmentation_v3.segment_trajectory_v3()` 是两条流程共用的函数，新增参数全部
**以 robomimic 行为为默认值**，只有 LeRobot profile 才打开新行为：

| 参数 | 函数默认（robomimic） | `moz1` profile |
|---|---|---|
| `sentinel_tail` | `True` | `False` |
| `merge_hiccup_gaps` | `0` | `6` |
| `interrupted_action_gap` | `0` | `60` |
| `min_short_travel` | `0.0` | `5e-3` |
| `mask_whole_command_run` | `True` | `False` |
| `min_closing_duration` / `min_opening_duration` | `4` / `1` | `2` / `2` |
| `min_movement_between` / `max_close_open_gap` | `10` / `30` | `0` / `30` |

等价性已用随机合成信号验证：80 组输入 × 3 个 task × (gripper, movement) 两条时间线，
共 240 例，`segment_trajectory_v3` 的默认参数输出与原实现**逐条相同**
（`tests/test_lerobot_gripper_segmentation.py` 中的
`test_segment_timeline_legacy_masking_swallows_the_rest_of_the_run`
与 `test_filter_gripper_labels_legacy_mode_masks_whole_run` 把这两条默认行为钉住）。

# VLN评测系统 - Benchmark配置格式设计

## 1. 概述

本文档定义了VLN（Vision-Language Navigation）评测系统的Benchmark配置格式规范。配置采用YAML格式，支持继承机制，用于定义评测任务、数据集、指标等核心配置。

### 1.1 设计原则

- **YAML格式**：易读易写，支持注释，兼容Hydra风格
- **核心配置**：任务+数据集+指标+评测，其他配置独立
- **支持继承**：可定义base benchmark，子benchmark覆盖部分配置
- **单场景单split**：一个配置对应一个数据集的一个split

---

## 2. 配置与领域概念的映射

本配置格式设计基于领域概念定义（参考vln_evaluation_architecture.md 1.4节）：

### 2.1 配置与概念对应表

| 配置项 | 对应概念 | 说明 |
|--------|----------|------|
| configs/tasks/vln.yaml | Task | 定义Task概念的动作空间、传感器套件、指标集合 |
| configs/benchmarks/*.yaml | Benchmark | 定义Benchmark概念的评测配置 |
| dataset配置段 | Dataset | 指定Dataset概念的路径和split |
| evaluation配置段 | - | 评测参数（非独立概念）|
| output配置段 | - | 输出配置（非独立概念）|

### 2.2 配置文件的职责划分

- **Task配置** (configs/tasks/): 定义"做什么任务" - 对应Task概念
- **Benchmark配置** (configs/benchmarks/): 定义"如何评测" - 对应Benchmark概念
- **Simulator配置** (configs/simulator/): 定义"仿真环境" - 对应Simulator概念

### 2.3 概念在配置中的体现

| 领域概念 | 配置体现 | 位置 |
|----------|----------|------|
| Task | task.actions, task.sensors, task.metrics | configs/tasks/*.yaml |
| Action | task.actions列表 | configs/tasks/*.yaml |
| Sensor | task.sensors配置 | configs/tasks/*.yaml |
| Metric | task.metrics列表 | configs/tasks/*.yaml |
| Observation | Sensors输出格式定义 | 文档第7.2节 |
| Action | Action格式定义 | 文档第8.1节 |
| Dataset | dataset配置段 | configs/benchmarks/*.yaml |
| Episode | Dataset中包含的Episode集合 | 运行时加载 |
| Benchmark | 整个benchmark配置 | configs/benchmarks/*.yaml |

---

## 3. 目录结构

```
configs/
├── tasks/
│   └── vln.yaml              # VLN任务基础配置（动作空间、传感器、指标）
│
├── benchmarks/
│   ├── vln_base.yaml         # VLN基准模板（抽象基类）
│   ├── vln_val_seen.yaml     # val_seen split具体实现
│   ├── vln_val_unseen.yaml   # val_unseen split具体实现
│   └── vln_test.yaml         # test split具体实现
│
└── simulator/
    └── default.yaml          # 仿真器默认配置（独立于benchmark）
```

---

## 4. 配置文件规范

### 4.1 任务配置 (configs/tasks/vln.yaml)

定义VLN任务的动作空间、传感器配置和评测指标。

```yaml
task:
  # 任务类型
  type: "vln"

  # 动作空间配置
  actions:
    - name: "stop"
      type: "discrete"

    - name: "move_forward"
      type: "discrete"
      params:
        step_size: 0.25  # meters

    - name: "turn_left"
      type: "discrete"
      params:
        turn_angle: 15  # degrees

    - name: "turn_right"
      type: "discrete"
      params:
        turn_angle: 15

    - name: "look_up"
      type: "discrete"
      params:
        tilt_angle: 15

    - name: "look_down"
      type: "discrete"
      params:
        tilt_angle: 15

  # 传感器配置
  sensors:
    rgb:
      width: 640
      height: 480
      hfov: 90  # horizontal field of view

    depth:
      width: 640
      height: 480
      hfov: 90
      min_depth: 0.0
      max_depth: 10.0

    instruction:
      type: "text"
      include_tokens: true  # 是否包含分词结果

    gps:
      coordinate_system: "cartesian"  # cartesian / spherical

    compass:
      type: "angle"  # angle / vector

  # 评测指标
  metrics:
    - success
    - spl
    - soft_spl
    - navigation_error
    - dtw
```

### 4.2 基础Benchmark模板 (configs/benchmarks/vln_base.yaml)

定义VLN benchmark的通用配置，作为具体split的父类。

```yaml
benchmark:
  # Benchmark元数据
  name: "VLN Base Benchmark"
  description: "VLN任务基准模板"
  version: "1.0"

  # 引用任务配置
  task: "vln"

  # 数据集配置
  dataset:
    type: "vln"
    format: "r2r"  # r2r / rxr / speaker_change
    data_path: null  # 子类必须覆盖
    scene_path: "data/scene_datasets/"
    split: null  # 子类必须覆盖
    episodes: null  # null=全部episode，可指定数量

  # 评测配置
  evaluation:
    max_steps: 500
    success_distance: 0.2  # meters
    stop_threshold: 0.2
    timeout: 30  # seconds per episode

  # 输出配置
  output:
    log_dir: "logs/evaluations"
    save_trajectories: true
    save_observations: false
    save_video: false
```

### 4.3 具体Benchmark配置 (configs/benchmarks/vln_val_seen.yaml)

基于vln_base的具体实现，覆盖必要字段。

```yaml
benchmark:
  # 继承基础配置
  extends: "vln_base"

  # Benchmark元数据
  name: "VLN Challenge 2024 - val_seen"
  description: "VLN任务验证集seen场景"
  version: "1.0"
  tags: ["vln", "r2r", "val_seen"]

  # 覆盖数据集配置
  dataset:
    type: "vln"
    format: "r2r"
    data_path: "data/datasets/vln/R2R/val_seen.json.gz"
    scene_path: "data/scene_datasets/"
    split: "val_seen"
    episodes: null  # 使用全部episode

  # 覆盖输出配置
  output:
    log_dir: "logs/evaluations/vln_val_seen"
    save_trajectories: true
    save_observations: false
    save_video: false
```

---

## 5. 配置字段说明

### 5.1 Benchmark元数据

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `extends` | string | 否 | 继承的父benchmark名称 |
| `name` | string | 是 | Benchmark名称 |
| `description` | string | 否 | Benchmark描述 |
| `version` | string | 是 | 配置版本号 |
| `tags` | list[string] | 否 | 标签列表，用于分类和搜索 |

### 5.2 任务配置

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `task` | string | 是 | 引用的任务配置名称或直接配置 |

### 5.3 数据集配置

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `type` | string | 是 | 数据集类型（vln/objectnav等） |
| `format` | string | 是 | 数据格式（r2r/rxr/speaker_change） |
| `data_path` | string | 是 | Episode数据文件路径 |
| `scene_path` | string | 是 | 场景文件目录路径 |
| `split` | string | 是 | 数据集split名称 |
| `episodes` | int\|null | 否 | 评测episode数量，null表示全部 |

### 5.4 评测配置

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `max_steps` | int | 是 | 单episode最大步数 |
| `success_distance` | float | 是 | 成功到达目标距离阈值（米） |
| `stop_threshold` | float | 是 | 停止动作的阈值 |
| `timeout` | int | 是 | 单episode超时时间（秒） |

### 5.5 输出配置

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `log_dir` | string | 是 | 日志输出目录 |
| `save_trajectories` | bool | 否 | 是否保存轨迹数据 |
| `save_observations` | bool | 否 | 是否保存观测数据 |
| `save_video` | bool | 否 | 是否保存视频 |

---

## 6. 动作空间规范

### 6.1 标准VLN离散动作

| 动作名称 | 参数 | 说明 |
|----------|------|------|
| `stop` | - | 停止导航，结束episode |
| `move_forward` | `step_size: 0.25` | 向前移动指定距离 |
| `turn_left` | `turn_angle: 15` | 左转指定角度（度） |
| `turn_right` | `turn_angle: 15` | 右转指定角度（度） |
| `look_up` | `tilt_angle: 15` | 向上看指定角度（度） |
| `look_down` | `tilt_angle: 15` | 向下看指定角度（度） |

### 6.2 动作参数

- `step_size`: 前进距离，单位米，默认0.25m
- `turn_angle`: 转向角度，单位度，默认15°
- `tilt_angle`: 俯仰角度，单位度，默认15°

---

## 7. 传感器配置规范

### 7.1 标准VLN传感器

| 传感器 | 配置项 | 说明 |
|--------|--------|------|
| `rgb` | width, height, hfov | RGB图像观测 |
| `depth` | width, height, hfov, min_depth, max_depth | 深度图像观测 |
| `instruction` | type, include_tokens | 文本指令观测 |
| `gps` | coordinate_system | 位置信息 |
| `compass` | type | 朝向信息 |

### 7.2 Observation（观测）格式

根据领域概念定义，**Observation**是某时刻环境的完整状态表示，由各Sensor的输出集合组成：

**完整Observation结构**:
```json
{
  "rgb": "<base64_encoded>",
  "depth": "<base64_encoded>",
  "instruction": {
    "text": "Walk down the hallway and enter the kitchen",
    "tokens": ["walk", "down", "the", "hallway", ...]
  },
  "gps": [x, y, z],
  "compass": angle
}
```

**各Sensor输出**：

- **RGB**: [H, W, 3] uint8 array (0-255)，base64编码
- **Depth**: [H, W] float32 array (meters)，base64编码
- **Instruction**: {...}
- **GPS**: [x, y, z] 相对坐标（米）
- **Compass**: angle (弧度) 或 [x, y, z] 向量

---

## 8. Action（动作）格式规范

根据领域概念定义，**Action**是Agent对环境的控制指令，用于改变Agent状态或位置。

### 8.1 Action结构

**标准格式**:
```json
{
  "action": "move_forward",
  "action_args": {
    "step_size": 0.25
  }
}
```

- `action`: 动作名称，必须是task.actions中定义的动作
- `action_args`: 可选参数，覆盖默认配置

### 8.2 标准VLN动作示例

| action | action_args | 说明 |
|--------|-------------|------|
| `"stop"` | `{}` | 停止导航，结束episode |
| `"move_forward"` | `{"step_size": 0.25}` | 向前移动指定距离 |
| `"turn_left"` | `{"turn_angle": 15}` | 左转指定角度（度） |
| `"turn_right"` | `{"turn_angle": 15}` | 右转指定角度（度） |
| `"look_up"` | `{"tilt_angle": 15}` | 向上看指定角度（度） |
| `"look_down"` | `{"tilt_angle": 15}` | 向下看指定角度（度） |

### 8.3 与配置的关系

Action在配置中的定义位置：
- **动作列表**: `configs/tasks/vln.yaml` → `task.actions`
- **默认参数**: `configs/tasks/vln.yaml` → `task.actions[].params`
- **运行时覆盖**: Agent可通过`action_args`覆盖默认参数

---

## 9. 指标规范

### 9.1 VLN标准指标

| 指标名称 | 说明 | 计算公式 |
|----------|------|----------|
| `success` | 是否成功到达目标 | distance_to_goal < threshold ? 1 : 0 |
| `spl` | 成功加权的路径长度效率 | Success × (最短路径长度 / 实际路径长度) |
| `soft_spl` | 放宽成功条件的SPL | max(0, 1 - distance/threshold) × (最短/实际) |
| `navigation_error` | 停止位置到目标的距离 | Euclidean(agent_final_pos, goal) |
| `dtw` | 轨迹与参考路径的动态时间规整距离 | DTW(trajectory, reference_path) |

### 9.2 指标计算依赖

- **SPL**: 需要 最短路径长度（geodesic）+ 实际路径长度 + success
- **DTW**: 需要 完整轨迹 + 参考路径
- **Navigation Error**: 需要 最终位置 + 目标位置

---

## 10. 配置加载流程

```
1. 加载基准配置文件 (vln_val_seen.yaml)
   └─> 读取 extends: "vln_base"

2. 加载父配置 (vln_base.yaml)
   └─> 读取 task: "vln"

3. 加载任务配置 (tasks/vln.yaml)
   └─> 获取 actions, sensors, metrics

4. 合并配置（子配置覆盖父配置）
   └─> data_path, log_dir等字段覆盖

5. 配置验证
   ├─> 检查必填字段
   ├─> 验证数据路径存在性
   └─> 验证动作/传感器/指标的有效性
```

---

## 11. 扩展性设计

### 11.1 新增任务类型

在`configs/tasks/`下创建新配置文件：

```yaml
# configs/tasks/objectnav.yaml
task:
  type: "objectnav"
  actions: [...]
  sensors: [...]
  metrics: [...]
```

### 11.2 新增Benchmark

继承现有base或直接创建：

```yaml
# configs/benchmarks/objectnav_gibson.yaml
benchmark:
  extends: "objectnav_base"
  name: "ObjectNav Gibson"
  dataset:
    data_path: "data/datasets/objectnav/gibson/val.json.gz"
```

### 11.3 新增指标

在任务配置的metrics列表添加：

```yaml
metrics:
  - success
  - spl
  - coverage  # 新增指标
```

---

## 12. 配置示例

### 12.1 完整的val_seen配置

```yaml
benchmark:
  extends: "vln_base"
  name: "VLN Challenge 2024 - val_seen"
  description: "VLN任务验证集seen场景"
  version: "1.0"
  tags: ["vln", "r2r", "val_seen"]

  task: "vln"

  dataset:
    type: "vln"
    format: "r2r"
    data_path: "data/datasets/vln/R2R/val_seen.json.gz"
    scene_path: "data/scene_datasets/"
    split: "val_seen"
    episodes: null

  evaluation:
    max_steps: 500
    success_distance: 0.2
    stop_threshold: 0.2
    timeout: 30

  output:
    log_dir: "logs/evaluations/vln_val_seen"
    save_trajectories: true
    save_observations: false
    save_video: false
```

### 12.2 评测子集配置

```yaml
benchmark:
  extends: "vln_base"
  name: "VLN Quick Test"
  dataset:
    data_path: "data/datasets/vln/R2R/val_seen.json.gz"
    episodes: 100  # 只评测100个episode
  evaluation:
    max_steps: 100  # 减少最大步数
```

---

## 13. 与架构文档的对应关系

基于领域概念定义的配置映射：

| 领域概念 | 配置项 | 配置文件 |
|----------|--------|----------|
| **Task** | task (actions, sensors, metrics) | configs/tasks/vln.yaml |
| **Benchmark** | benchmark (name, dataset, evaluation) | configs/benchmarks/*.yaml |
| **Dataset** | dataset | configs/benchmarks/*.yaml |
| **Action** | task.actions | configs/tasks/vln.yaml |
| **Sensor** | task.sensors | configs/tasks/vln.yaml |
| **Metric** | task.metrics | configs/tasks/vln.yaml |
| **Observation** | sensors输出格式 | 文档第7.2节定义 |
| **Simulator** | simulator | configs/simulator/default.yaml |
| **Evaluation** | evaluation | configs/benchmarks/*.yaml |
| **Configuration** | 整个YAML配置系统 | configs/ |

---

## 14. 下一步工作

- [ ] 实现配置加载器（支持继承和合并）
- [ ] 实现配置验证器（schema验证）
- [ ] 创建配置模板生成工具
- [ ] 编写配置迁移脚本（从Habitat格式迁移）

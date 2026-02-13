# Challenge 评测系统数据集格式规范

## 概述

本文档定义了 Challenge 评测系统的数据集格式，支持**导航任务**和**操作任务**两大类。设计原则：

- **通用性**：单一格式支持 VLN、ObjectNav、Manipulation 等多种任务
- **简洁性**：仅保留必要字段，参考 R2R 格式
- **可扩展**：通过 `task_type` 和 `goal` 字段支持不同任务
- **分离设计**：任务数据集（评测用）与轨迹数据集（训练用）分离
- **任务无关**：动作空间由评测接口处理，数据格式不限制具体表示

---

## 数据集类型区分

| 数据集类型 | 用途 | 文件示例 | 是否包含轨迹 |
|-----------|------|----------|-------------|
| **Task Dataset** | 评测 Episode 定义 | `val_seen.json.gz` | ❌ 否 |
| **Trajectory Dataset** | 模仿学习训练数据 | `train_trajectories.jsonl.gz` | ✅ 是 |

---

## 目录结构

```
data/datasets/
├── navigation/              # 导航任务
│   ├── vln/
│   │   ├── r2r/
│   │   │   ├── train/
│   │   │   │   ├── train.json.gz              # 任务数据集
│   │   │   │   └── train_trajectories.jsonl.gz # 轨迹数据集（可选）
│   │   │   ├── val_seen/
│   │   │   │   └── val_seen.json.gz
│   │   │   └── val_unseen/
│   │   │       └── val_unseen.json.gz
│   │   └── custom/
│   │       └── test/
│   │           └── test.json.gz
│   │
│   ├── objectnav/
│   │   └── hm3d/
│   │       ├── train/
│   │       │   ├── train.json.gz
│   │       │   └── train_trajectories.jsonl.gz
│   │       └── val/
│   │           └── val.json.gz
│   │
│   └── imagenav/
│       └── habitat/
│           └── test/
│               └── test.json.gz
│
└── manipulation/            # 操作任务
    └── robotwin/
        ├── train/
        │   ├── train.json.gz
        │   └── train_trajectories.jsonl.gz
        ├── val/
        │   └── val.json.gz
        └── test/
            └── test.json.gz
```

---

## 1. Task Dataset 格式（任务数据集）

用于评测的 Episode 定义，不包含参考轨迹。

### 1.1 顶层结构

```json
{
  "episodes": [...],
  "instruction_vocab": {}
}
```

### 1.2 Episode 结构（通用）

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `episode_id` | string/int | ✅ | 唯一标识符 |
| `task_type` | string | ✅ | 任务类型：`vln`, `objectnav`, `imagenav` 等 |
| `scene_id` | string | ✅ | 场景文件路径 |
| `start_position` | [x,y,z] | ✅ | 起始位置（米） |
| `start_rotation` | [w,x,y,z] | ✅ | 起始旋转（四元数） |
| `instruction` | object | task-dependent | 指令信息（VLN 必填） |
| `goal` | object | ✅ | 目标定义（任务类型相关） |
| `info` | object | ❌ | 额外元数据 |

### 1.3 不同任务类型的 Episode 示例

#### VLN (Vision-Language Navigation)

```json
{
  "episode_id": 1,
  "task_type": "vln",
  "scene_id": "mp3d/s8pcmisQ38h/s8pcmisQ38h.glb",
  "start_position": [5.58, -1.63, 2.82],
  "start_rotation": [-0.0, 0.966, -0.0, -0.259],
  "instruction": {
    "instruction_text": "Walk down the stairs, turn right, and walk towards the place with a rug.",
    "instruction_tokens": [2384, 717, 2202, 2058, 2300, 1819, ...]
  },
  "goal": {
    "type": "position",
    "position": [11.63, -3.16, 2.0],
    "radius": 3.0
  },
  "info": {
    "geodesic_distance": 10.39,
    "difficulty": "medium"
  }
}
```

#### ObjectNav (Object Navigation)

```json
{
  "episode_id": "obj_001",
  "task_type": "objectnav",
  "scene_id": "hm3d/XXXXX.glb",
  "start_position": [2.5, 0.1, 3.2],
  "start_rotation": [0.0, 0.0, 0.0, 1.0],
  "goal": {
    "type": "object",
    "object_category": "chair",
    "object_id": "chair_123"
  },
  "info": {
    "view_points": [[2.6, 0.1, 3.5], ...],
    "difficulty": "easy"
  }
}
```

#### ImageNav (Image-Guided Navigation)

```json
{
  "episode_id": "img_001",
  "task_type": "imagenav",
  "scene_id": "gibson/XXXXX.glb",
  "start_position": [1.0, 0.0, 1.0],
  "start_rotation": [0.0, 0.0, 0.0, 1.0],
  "goal": {
    "type": "image",
    "goal_image": "goal_images/scene1_view45.jpg",
    "position": [5.2, 0.0, 7.8],
    "radius": 2.0
  },
  "info": {
    "viewpoint_id": 45
  }
}
```

#### Manipulation (操作任务 - Pick and Place)

```json
{
  "episode_id": "mani_001",
  "task_type": "manipulation",
  "manipulation_type": "pick_place",
  "scene_id": "robotwin/table1.glb",
  "robot_embodiment": {
    "type": "single_arm",
    "robot_type": "franka-panda",
    "base_position": [0.0, 0.0, 0.0],
    "gripper_type": "parallel_jaw"
  },
  "start_position": [0.5, 0.0, 0.3],
  "start_rotation": [0.0, 0.0, 0.0, 1.0],
  "instruction": {
    "instruction_text": "Pick up the red block and place it in the blue box"
  },
  "goal": {
    "type": "pick_place",
    "target_object": {
      "id": "block_red_1",
      "category": "block",
      "properties": {"color": "red"},
      "initial_position": [0.6, 0.0, 0.4]
    },
    "target_location": {
      "type": "container",
      "id": "box_blue_1",
      "category": "box",
      "properties": {"color": "blue"},
      "position": [0.3, 0.0, -0.2],
      "radius": 0.15
    },
    "success_conditions": {
      "object_in_container": true,
      "min_duration": 0.5
    }
  },
  "info": {
    "difficulty": "medium",
    "max_episode_length": 500,
    "initial_joint_positions": [0.0, -0.5, 0.0, 1.5, 0.0, 1.0, 0.0]
  }
}
```

#### Manipulation (操作任务 - Reach)

```json
{
  "episode_id": "mani_002",
  "task_type": "manipulation",
  "manipulation_type": "reach",
  "scene_id": "robotwin/table2.glb",
  "robot_embodiment": {
    "type": "single_arm",
    "robot_type": "ur5e-wsg"
  },
  "start_position": [0.0, 0.0, 0.0],
  "start_rotation": [0.0, 0.0, 0.0, 1.0],
  "goal": {
    "type": "reach",
    "target_pose": {
      "position": [0.5, 0.3, 0.4],
      "quaternion": [0.0, 0.0, 0.0, 1.0]
    },
    "tolerance": {
      "position": 0.05,
      "orientation": 0.2
    }
  },
  "info": {
    "max_episode_length": 200
  }
}
```

### 1.4 Goal 字段定义

#### 导航任务 Goal

| `goal.type` | 任务类型 | 必填字段 | 说明 |
|-------------|----------|----------|------|
| `position` | VLN | `position`, `radius` | 目标位置和容差半径 |
| `object` | ObjectNav | `object_category` | 目标物体类别 |
| `image` | ImageNav | `goal_image` | 目标图像路径 |

#### 操作任务 Goal

| `goal.type` | 操作子类型 | 必填字段 | 说明 |
|-------------|-----------|----------|------|
| `pick_place` | 抓取-放置 | `target_object`, `target_location` | 从初始位置抓取物体并放到目标位置 |
| `reach` | 到达位姿 | `target_pose` | 末端执行器到达目标位姿 |
| `tool_use` | 工具使用 | `tool`, `target_object`, `action` | 使用工具与目标物体交互 |

---

## 2. Trajectory Dataset 格式（轨迹数据集）

用于模仿学习训练，包含完整轨迹和指标。使用 JSONL 格式（每行一个轨迹）。

### 2.1 单条轨迹格式

```json
{
  "episode_id": 1,
  "task_type": "vln",
  "scene_id": "mp3d/s8pcmisQ38h/s8pcmisQ38h.glb",
  "instruction": {
    "instruction_text": "Walk down the stairs..."
  },
  "start_position": [5.58, -1.63, 2.82],
  "start_rotation": [-0.0, 0.966, -0.0, -0.259],
  "goal": {
    "type": "position",
    "position": [11.63, -3.16, 2.0],
    "radius": 3.0
  },
  "trajectory": {
    "positions": [[5.58, -1.63, 2.82], [5.32, -1.78, 2.85], ...],
    "actions": [2, 2, 2, 1, 1, 3, ...],
    "observations": ["<base64_rgb_1>", "<base64_rgb_2>", ...]
  },
  "metrics": {
    "success": 1,
    "spl": 0.95,
    "navigation_error": 0.15,
    "length": 45
  },
  "info": {
    "agent_id": "expert_dagger_001",
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

### 2.2 轨迹字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `trajectory.positions` | [[x,y,z],...] | 每步位置（导航任务） |
| `trajectory.actions` | array | 动作序列（任务相关，见下文） |
| `trajectory.observations` | [string,...] | 观测数据（base64 编码），可选 |
| `metrics.success` | int/float | 是否成功（0/1） |
| `metrics.spl` | float | Success weighted by Path Length |
| `metrics.navigation_error` | float | 最终位置到目标的距离（导航任务） |
| `metrics.length` | int | 轨迹长度（步数） |

#### 动作格式说明

**导航任务**（VLN、ObjectNav、ImageNav）：
```json
"actions": [2, 2, 2, 1, 1, 3, ...]  // 离散动作编码（0=stop, 1=forward, 2=left, 3=right）
```

**操作任务**（Manipulation）：
- 动作格式由评测接口定义，数据集记录 agent 的原始输出
- 可以是关节位置、末端位姿、或任务特定的控制信号

```json
"actions": [
  {
    "type": "continuous",
    "timestamp": 0.0,
    "joint_positions": [0.1, -0.3, 0.5, 1.2, 0.0, 0.8, 0.0],
    "gripper": 0.5
  },
  {
    "type": "continuous",
    "timestamp": 0.1,
    "ee_pose": [0.5, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0],  // [x,y,z,qw,qx,qy,qz]
    "gripper": 0.3
  }
]
```

---

## 3. 任务类型扩展

### 3.1 支持的任务类型

#### 导航任务

| `task_type` | 说明 | 必填字段 | 特殊字段 |
|-------------|------|----------|----------|
| `vln` | 视觉语言导航 | `instruction` | `goal.type=position` |
| `objectnav` | 物体导航 | - | `goal.type=object` |
| `imagenav` | 图像导航 | - | `goal.type=image` |
| `roomnav` | 房间导航 | - | `goal.type=room` |
| `multi_objectnav` | 多物体导航 | - | `goal.type=object_list` |

#### 操作任务

| `task_type` | 说明 | 必填字段 | 特殊字段 |
|-------------|------|----------|----------|
| `manipulation` | 通用操作任务 | - | `manipulation_type`, `robot_embodiment` |
| `pick_place` | 抓取-放置任务 | `target_object`, `target_location` | `goal.type=pick_place` |
| `reach` | 到达位姿任务 | `target_pose` | `goal.type=reach` |
| `tool_use` | 工具使用任务 | `tool`, `target_object` | `goal.type=tool_use` |

### 3.2 扩展新任务

添加新任务类型只需：

1. 在 `task_type` 枚举中注册新值
2. 定义该任务的 `goal` 格式
3. 实现对应的评测指标

**示例：添加 RoomNav 任务**

```json
{
  "episode_id": "room_001",
  "task_type": "roomnav",
  "scene_id": "mp3d/XXXXX.glb",
  "start_position": [1.0, 0.0, 1.0],
  "start_rotation": [0.0, 0.0, 0.0, 1.0],
  "goal": {
    "type": "room",
    "room_type": "kitchen",
    "room_id": "room_5"
  }
}
```

---

## 4. 动作编码标准

### 4.1 导航任务动作编码

| 动作值 | 动作名称 | 参数 |
|--------|----------|------|
| 0 | STOP | - |
| 1 | FORWARD | `step_size: 0.25` (默认) |
| 2 | LEFT | `turn_angle: 15°` (默认) |
| 3 | RIGHT | `turn_angle: 15°` (默认) |
| 4 | LOOK_UP | `tilt_angle: 15°` (可选) |
| 5 | LOOK_DOWN | `tilt_angle: 15°` (可选) |

### 4.2 操作任务动作格式

操作任务的动作格式**不由数据格式固定**，而是由评测系统的 Action 接口定义：

- **关节位置控制**：`[j1, j2, ..., j7, gripper]`
- **末端位姿控制**：`[x, y, z, qw, qx, qy, qz, gripper]`
- **任务特定格式**：由具体任务和机器人平台定义

**重要说明**：
- 数据格式只负责**记录** agent 输出的动作
- 动作的具体语义由评测系统和 Simulator 解析
- 不同操作任务可以使用不同的动作表示

---

## 5. 与 R2R/RXR 的对比

| 特性 | R2R | RXR | Challenge Format |
|------|-----|-----|------------------|
| 任务类型 | VLN | VLN | 多任务支持 |
| 多语言 | ❌ | ✅ | ❌ (仅英语) |
| 时间对齐 | ❌ | ✅ | ❌ |
| Guide/Follower | ❌ | ✅ | ❌ |
| Episode 级 metrics | ❌ | ✅ | ❌ (GT 数据集有) |
| 独立 GT 文件 | ✅ | ✅ | ❌ (Trajectory Dataset) |
| 任务/轨迹分离 | ❌ | ⚠️ | ✅ 清晰分离 |

---

## 6. 配置文件集成

数据集格式与 Benchmark 配置对应：

### 导航任务示例

```yaml
# configs/benchmarks/vln_val_seen.yaml
benchmark:
  task: "vln"
  dataset:
    type: "vln"
    format: "challenge"
    data_path: "data/datasets/navigation/vln/r2r/val_seen.json.gz"
    scene_path: "data/scene_datasets/"
```

```yaml
# configs/benchmarks/objectnav_hm3d.yaml
benchmark:
  task: "objectnav"
  dataset:
    type: "objectnav"
    format: "challenge"
    data_path: "data/datasets/navigation/objectnav/hm3d/val.json.gz"
```

### 操作任务示例

```yaml
# configs/benchmarks/manipulation_robotwin.yaml
benchmark:
  task: "manipulation"
  dataset:
    type: "manipulation"
    format: "challenge"
    data_path: "data/datasets/manipulation/robotwin/val.json.gz"
    scene_path: "data/scene_datasets/"

  # 机器人配置
  robot:
    type: "single_arm"
    model: "franka-panda"
    action_space: "continuous"  # 或 "ee_pose", "joint_position"
```

---

## 7. 数据验证 Schema

### 7.1 Task Dataset Schema

```python
TASK_DATASET_SCHEMA = {
    "type": "object",
    "required": ["episodes"],
    "properties": {
        "episodes": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["episode_id", "task_type", "scene_id",
                           "start_position", "start_rotation", "goal"],
                "properties": {
                    "episode_id": {"type": ["string", "integer"]},
                    # 导航任务
                    "task_type": {"enum": ["vln", "objectnav", "imagenav", "roomnav",
                                         "multi_objectnav",
                                         "manipulation", "pick_place", "reach", "tool_use"]},
                    "scene_id": {"type": "string"},
                    "start_position": {"type": "array", "minItems": 3, "maxItems": 3},
                    "start_rotation": {"type": "array", "minItems": 4, "maxItems": 4},
                    "instruction": {"type": "object"},
                    "goal": {"type": "object"},
                    # 操作任务特有字段
                    "manipulation_type": {"type": "string"},
                    "robot_embodiment": {"type": "object"},
                    "info": {"type": "object"}
                }
            }
        }
    }
}
```

### 7.2 操作任务 Episode Schema

```python
MANIPULATION_EPISODE_SCHEMA = {
    "type": "object",
    "required": ["episode_id", "task_type", "scene_id",
               "start_position", "start_rotation", "goal",
               "robot_embodiment"],
    "properties": {
        "episode_id": {"type": "string"},
        "task_type": {"enum": ["manipulation", "pick_place", "reach", "tool_use"]},
        "manipulation_type": {"enum": ["pick_place", "reach", "tool_use", "press", "pour"]},
        "scene_id": {"type": "string"},
        "start_position": {"type": "array", "minItems": 3, "maxItems": 3},
        "start_rotation": {"type": "array", "minItems": 4, "maxItems": 4},
        "robot_embodiment": {
            "type": "object",
            "required": ["type", "robot_type"],
            "properties": {
                "type": {"enum": ["single_arm"]},
                "robot_type": {"type": "string"},
                "base_position": {"type": "array"},
                "gripper_type": {"type": "string"}
            }
        },
        "goal": {"type": "object"},
        "instruction": {"type": "object"},
        "info": {"type": "object"}
    }
}
```

---

## 8. 使用示例

### 8.1 读取任务数据集

```python
import json
import gzip

with gzip.open('data/datasets/vln/r2r/val_seen.json.gz', 'rt') as f:
    data = json.load(f)

for episode in data['episodes']:
    if episode['task_type'] == 'vln':
        instruction = episode['instruction']['instruction_text']
        goal = episode['goal']['position']
        print(f"Instruction: {instruction}")
        print(f"Goal: {goal}")
```

### 8.2 读取轨迹数据集

```python
import json
import gzip

with gzip.open('data/datasets/vln/r2r/train_trajectories.jsonl.gz', 'rt') as f:
    for line in f:
        trajectory = json.loads(line)
        success = trajectory['metrics']['success']
        if success == 1:
            # 使用高质量轨迹进行训练
            pass
```

### 8.3 创建新数据集

```python
episodes = []
for scene in scenes:
    episode = {
        "episode_id": f"ep_{len(episodes)}",
        "task_type": "vln",
        "scene_id": scene['id'],
        "start_position": scene['start'],
        "start_rotation": [0, 0, 0, 1],
        "instruction": {
            "instruction_text": scene['instruction']
        },
        "goal": {
            "type": "position",
            "position": scene['goal'],
            "radius": 3.0
        }
    }
    episodes.append(episode)

with gzip.open('output.json.gz', 'wt') as f:
    json.dump({"episodes": episodes}, f)
```

---

## 9. 向后兼容

本格式支持 R2R 数据集的直接兼容：

```python
def convert_r2r_to_challenge(r2r_data):
    """将 R2R 格式转换为 Challenge 格式"""
    episodes = []
    for ep in r2r_data['episodes']:
        challenge_ep = {
            "episode_id": ep['episode_id'],
            "task_type": "vln",  # 新增字段
            "scene_id": ep['scene_id'],
            "start_position": ep['start_position'],
            "start_rotation": ep['start_rotation'],
            "instruction": ep['instruction'],  # 保持不变
            "goal": {
                "type": "position",  # 重组为 goal 对象
                "position": ep['goals'][0]['position'],
                "radius": ep['goals'][0]['radius']
            }
        }
        # 复制 info 字段
        if 'info' in ep:
            challenge_ep['info'] = ep['info']
        episodes.append(challenge_ep)
    return {"episodes": episodes}
```

---

## 10. 版本历史

| 版本 | 日期 | 变更 |
|------|------|------|
| 1.0 | 2024-01 | 初始版本，支持 VLN/ObjectNav/ImageNav |
| 1.1 | 2024-01 | 添加操作任务支持，添加 `manipulation` 任务类型和 `robot_embodiment` 字段 |
| 1.2 | 2024-01 | 扩展 Trajectory 格式支持操作任务的连续动作记录 |

---

## 11. 常见问题 (FAQ)

**Q: 为什么不使用 RXR 的多语言和时间对齐？**

A: 为了保持简洁。如果未来需要多语言支持，可以在 `instruction` 对象中添加 `language` 和 `timed_instruction` 字段。

**Q: 轨迹数据集为什么使用 JSONL 而不是 JSON？**

A: JSONL（每行一个 JSON）更适合流式处理大数据集，也方便按需加载和筛选。

**Q: 如何添加新任务类型？**

A: 在 `task_type` 中注册新值，定义对应的 `goal` 格式，实现任务和评测指标。

**Q: 评测时如何判断不同任务的 success？**

A: 根据 `goal.type` 使用不同的判断逻辑：
- 导航任务：
  - `position`: 距离 < radius
  - `object`: 物体在视野内且距离足够近
  - `image`: 视角相似度 + 距离阈值
- 操作任务：
  - `pick_place`: 物体在目标容器内并持续一定时间
  - `reach`: 末端位姿在容差范围内
  - `tool_use`: 工具与目标物体发生接触/交互

**Q: 操作任务的动作格式为什么不固定？**

A: 不同操作任务和机器人平台需要不同的动作表示（关节位置、末端位姿等）。数据格式只负责记录 agent 的输出，具体语义由评测系统和 Simulator 解析。这种设计提供了最大的灵活性。

**Q: 导航任务和操作任务可以混用同一个数据集文件吗？**

A: 可以。通过 `task_type` 字段区分不同任务类型，评测系统会根据类型选择相应的处理逻辑。但建议将不同任务类型的数据集分开存储，便于管理和使用。

**Q: 如何支持双臂操作任务？**

A: 在 `robot_embodiment.type` 中添加 `"dual_arm"` 选项，并在 `goal` 定义中扩展左右臂的目标字段。当前版本仅支持单臂任务。

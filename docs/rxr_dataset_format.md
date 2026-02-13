# RXR 数据集格式说明

## 概述

RXR (Room-to-Room) 是一个多语言视觉语言导航数据集，支持多种语言的导航指令理解和执行。

## 目录结构

```
data/rxr/
├── train/               # 训练集
│   ├── train_follower.json.gz
│   ├── train_guide.json.gz
│   └── train_guide_gt.json.gz
├── val_seen/           # 验证集（可见场景）
│   ├── val_seen_follower.json.gz
│   ├── val_seen_follower_gt.json.gz
│   ├── val_seen_guide.json.gz
│   └── val_seen_guide_gt.json.gz
├── val_unseen/         # 验证集（未见场景）
│   ├── val_unseen_follower.json.gz
│   ├── val_unseen_follower_gt.json.gz
│   ├── val_unseen_guide.json.gz
│   └── val_unseen_guide_gt.json.gz
└── test_challenge/     # 测试挑战集
    └── test_challenge_guide.json.gz
```

## 数据角色

- **Follower**（导航者）：接收指令并执行导航的 agent
- **Guide**（引导者）：提供导航指令的一方

## 文件命名规则

| 文件后缀 | 说明 |
|---------|------|
| `*_follower.json.gz` | 导航者任务数据 |
| `*_follower_gt.json.gz` | 导航者轨迹真值（Ground Truth） |
| `*_guide.json.gz` | 引导者任务数据 |
| `*_guide_gt.json.gz` | 引导者轨迹真值 |

## 数据格式详解

### 1. Follower 数据格式

```json
{
  "episodes": [
    {
      "episode_id": "60301",
      "scene_id": "mp3d/SN83YJsR3w2/SN83YJsR3w2.glb",
      "instruction": {
        "instruction_id": "0",
        "instruction_text": "You will start by standing in front of a glass door...",
        "language": "en-US",
        "annotator_id": "0",
        "edit_distance": 0.07692307692307693,
        "timed_instruction": [
          {
            "word": "You",
            "start_time": 1.0,
            "end_time": 1.5
          }
        ]
      },
      "start_position": [0.25628501176834106, 3.8914499282836914, -16.086700439453125],
      "start_rotation": [-0.0, 0.15576595430371898, 0.0, 0.9877939904048069],
      "goals": [
        {
          "position": [1.0714499950408936, 3.8914499282836914, -18.572399139404297],
          "radius": 3.0
        }
      ],
      "reference_path": [
        [0.25628501176834106, 3.8914499282836914, -16.086700439453125],
        [0.5, 3.8914499282836914, -17.0],
        [1.0714499950408936, 3.8914499282836914, -18.572399139404297]
      ],
      "info": {
        "metrics": {
          "ndtw": 0.9999999920527141,
          "ne": 0.4209725856781006,
          "sdtw": 0.9999999920527141,
          "spl": 1.0,
          "sr": 1.0
        },
        "demonstration_id": "1",
        "role": "follower"
      }
    }
  ]
}
```

#### 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `episode_id` | string | 剧本唯一标识符 |
| `scene_id` | string | 场景文件路径（Matterport3D .glb 格式） |
| `instruction` | object | 指令信息 |
| `start_position` | [x, y, z] | 起始位置坐标 |
| `start_rotation` | [w, x, y, z] | 起始旋转（四元数） |
| `goals` | array | 目标位置列表 |
| `reference_path` | [[x,y,z],...] | 参考导航路径点 |
| `info` | object | 额外信息 |

#### instruction 子字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `instruction_id` | string | 指令唯一标识符 |
| `instruction_text` | string | 指令文本内容 |
| `language` | string | 语言代码（如 en-US, te-IN, hi-IN） |
| `annotator_id` | string | 标注者ID |
| `edit_distance` | float | 编辑距离 |
| `timed_instruction` | array | 词级时间戳对齐 |

#### metrics 子字段（评估指标）

| 指标 | 全称 | 说明 |
|------|------|------|
| `ndtw` | Normalized Dynamic Time Warping | 归一化动态时间规整距离 |
| `ne` | Navigation Error | 导航误差 |
| `sdtw` | Success weighted by DTW | DTW加权的成功率 |
| `spl` | Success weighted by Path Length | 路径长度加权的成功率 |
| `sr` | Success Rate | 成功率 |

---

### 2. Guide 数据格式

```json
{
  "episodes": [
    {
      "episode_id": "1",
      "trajectory_id": "0",
      "scene_id": "mp3d/SN83YJsR3w2/SN83YJsR3w2.glb",
      "info": {
        "role": "guide"
      },
      "instruction": {
        "instruction_id": "0",
        "instruction_text": "You will start by standing in front of a glass door...",
        "language": "en-US",
        "annotator_id": "0",
        "edit_distance": 0.07692307692307693,
        "timed_instruction": [...]
      },
      "start_position": [0.25628501176834106, 3.8914499282836914, -16.086700439453125],
      "start_rotation": [-0.0, 0.15576595430371898, 0.0, 0.9877939904048069],
      "goals": [
        {
          "position": [1.0714499950408936, 3.8914499282836914, -18.572399139404297],
          "radius": 3.0
        }
      ],
      "reference_path": [
        [0.25628501176834106, 3.8914499282836914, -16.086700439453125],
        ...
      ]
    }
  ]
}
```

#### Guide 与 Follower 的区别

| 字段 | Follower | Guide |
|------|----------|-------|
| `trajectory_id` | ❌ | ✅ |
| `info.metrics` | ✅ | ❌ |
| `info.demonstration_id` | ✅ | ❌ |

---

### 3. Follower GT (Ground Truth) 格式

```json
{
  "9638": {
    "locations": [
      [-12.83810043334961, 3.761544942855835, -14.952699661254883],
      [-13.086076736450195, 3.7615439891815186, -14.920951843261719],
      ...
    ],
    "actions": [2, 2, 2, 2, 2, 2, 1, 1, 1, 3, 3, 3, 1, ...],
    "forward_steps": 21
  },
  "8884": {
    ...
  }
}
```

#### GT 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| (key) | string | episode_id 作为字典键 |
| `locations` | [[x,y,z],...] | 位置序列 |
| `actions` | [int,...] | 动作序列 |
| `forward_steps` | int | 前进步数统计 |

#### 动作编码

| 动作值 | 动作名称 | 说明 |
|--------|----------|------|
| 0 | STOP | 停止 |
| 1 | MOVE_FORWARD | 向前移动 |
| 2 | TURN_LEFT | 左转 |
| 3 | TURN_RIGHT | 右转 |

---

## 关键特性

### 1. 多语言支持

RXR 支持多种语言的导航指令：

| 语言代码 | 语言 |
|----------|------|
| en-US | 英语（美国） |
| te-IN | 泰卢固语（印度） |
| hi-IN | 印地语（印度） |
| ... | 其他语言 |

### 2. 时间对齐

`timed_instruction` 提供词级时间戳，支持：
- 语音合成的时间同步
- 视觉-语言对齐研究
- 多模态学习

### 3. 场景来源

所有场景来自 **Matterport3D (MP3D)** 数据集：
- 格式：`.glb` (GLTF Binary)
- 路径格式：`mp3d/{scene_id}/{scene_id}.glb`

### 4. 评估指标详解

- **NDTW (Normalized DTW)**: 衡量预测路径与参考路径的相似度
- **SDTW (Success DTW)**: 同时考虑成功率和路径相似度
- **SPL**: 考虑路径效率的成功率
- **SR**: 是否到达目标区域（radius 范围内）

---

## 数据集划分

| 划分 | 场景类型 | 用途 |
|------|----------|------|
| train | 训练场景 | 模型训练 |
| val_seen | 可见场景 | 验证（场景在训练集中见过） |
| val_unseen | 未见场景 | 验证（场景在训练集中未见过） |
| test_challenge | 挑战场景 | 测试比赛 |

---

## 使用示例

### 读取数据

```python
import json
import gzip

# 读取 follower 数据
with gzip.open('data/rxr/val_seen/val_seen_follower.json.gz', 'rt') as f:
    data = json.load(f)

for episode in data['episodes']:
    print(f"Episode: {episode['episode_id']}")
    print(f"Instruction: {episode['instruction']['instruction_text']}")
    print(f"Language: {episode['instruction']['language']}")
```

### 读取 GT 数据

```python
import json
import gzip

# 读取 follower GT 数据
with gzip.open('data/rxr/val_seen/val_seen_follower_gt.json.gz', 'rt') as f:
    gt_data = json.load(f)

for episode_id, trajectory in gt_data.items():
    locations = trajectory['locations']
    actions = trajectory['actions']
    print(f"Episode {episode_id}: {len(locations)} steps")
```

---

## 与 R2R 数据集的区别

| 特性 | R2R | RXR |
|------|-----|-----|
| 语言 | 仅英语 | 多语言（英语+印地语+泰卢固语等） |
| 时间对齐 | ❌ | ✅ (timed_instruction) |
| 角色 | Follower | Follower + Guide |
| 场景来源 | MP3D | MP3D |
| 评估指标 | SR, SPL, NE | SR, SPL, NE, NDTW, SDTW |

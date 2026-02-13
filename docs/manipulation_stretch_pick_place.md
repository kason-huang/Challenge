# Stretch Pick and Place 任务详细设计

> 基于用户确认需求：仅接口定义 + Stretch 机器人 + Pick and Place 任务

---

## 一、Stretch 机器人配置

### 1.1 机器人规格

```
┌─────────────────────────────────────────────────────────────┐
│              Stretch (Hello Robot)                            │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────┐    ┌───────────────────────────┐            │
│  │ Mobile  │    │  Arm (DOF: 5)            │            │
│  │ Base    │────│  ├─ Lift (0.86m)         │            │
│  │ (2-DOF) │    │  ├─ Shoulder             │            │
│  │         │    │  ├─ Elbow               │            │
│  │         │    │  ├─ Wrist                │            │
│  │         │    │  └─ Gripper (Parallel)  │            │
│  └─────────┘    └───────────────────────────┘            │
│                                                               │
└─────────────────────────────────────────────────────────────┘

自由度 (DOF):
  - Mobile Base: 2 (x, y, θ)
  - Arm: 5 (lift, shoulder, elbow, wrist, gripper)
  - Total: 7 连续控制维度
```

### 1.2 动作空间定义

```python
# simulator/robot/stretch_config.py
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class StretchRobotConfig:
    """Stretch 机器人配置"""

    robot_type: str = "stretch"

    # 自由度配置
    base_dof: int = 2          # 移动底座 (x, y, θ)
    arm_dof: int = 5            # 机械臂 (lift, shoulder, elbow, wrist, gripper)
    total_dof: int = 7

    # 关节名称（与实际机器人对应）
    joint_names: List[str] = None

    def __post_init__(self):
        self.joint_names = [
            # Mobile Base
            "translate_x",
            "translate_y",
            "rotate_z",

            # Arm
            "joint_lift",
            "joint_arm_l0",
            "joint_arm_l1",
            "joint_arm_l2",
            "joint_arm_l3",
            "joint_wrist_yaw",
            "joint_gripper_finger_left"
        ]

    # 动作范围（弧度）
    joint_limits: Dict[str, tuple] = None

    def __post_init__(self):
        if self.joint_limits is None:
            # 参考 Hello Robotics 规格
            self.joint_limits = {
                "translate_x": (-0.5, 0.5),      # meters
                "translate_y": (-0.5, 0.5),
                "rotate_z": (-3.14, 3.14),       # radians

                "joint_lift": (0.0, 1.1),          # meters
                "joint_arm_l0": (-1.8, 1.8),       # radians
                "joint_arm_l1": (-1.8, 1.8),
                "joint_arm_l2": (-1.8, 1.8),
                "joint_arm_l3": (-1.8, 1.8),
                "joint_wrist_yaw": (-1.8, 1.8),
                "joint_gripper_finger_left": (0.0, 0.04)  # meters (open)
            }

    # 末端执行器
    end_effector: str = "parallel_gripper"

    # 相机配置
    camera_config: Dict[str, Dict] = None

    def __post_init__(self):
        if self.camera_config is None:
            self.camera_config = {
                "head": {
                    "position": [0.1, 0.0, 1.2],  # 相对于底座
                    "orientation": [0, 0, 0, 1],
                    "width": 640,
                    "height": 480,
                    "fov": 60
                },
                "wrist": {
                    "position": [0, 0, 0],  # 相对于腕关节
                    "orientation": [0, 0, 0, 1],
                    "width": 320,
                    "height": 240,
                    "fov": 90
                }
            }
```

### 1.3 传感器配置

```python
# simulator/sensor/stretch_sensors.py
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class StretchSensorSuite:
    """Stretch 传感器套件"""

    # 视觉传感器
    rgb_head: Dict[str, Any]
    rgb_wrist: Dict[str, Any]
    depth_head: Dict[str, Any]

    # 本体感知
    joint_positions: Dict[str, Any]      # qpos (7-dim)
    joint_velocities: Dict[str, Any]     # qvel (7-dim)

    # 末端执行器状态
    gripper_state: Dict[str, Any]        # gripper opening (1-dim)
    ee_pose: Dict[str, Any]             # end-effector pose (7-dim: pos + quat)

    # 任务相关
    instruction: Dict[str, Any]         # 语言指令
    object_pose: Dict[str, Any]         # 目标物体位姿（可选，用于调试）

    def __post_init__(self):
        # RGB 相机（头部）
        self.rgb_head = {
            "width": 640,
            "height": 480,
            "encoding": "base64",
            "format": "RGB"
        }

        # RGB 相机（腕部）
        self.rgb_wrist = {
            "width": 320,
            "height": 240,
            "encoding": "base64",
            "format": "RGB"
        }

        # Depth 相机（头部）
        self.depth_head = {
            "width": 640,
            "height": 480,
            "encoding": "base64",
            "format": "depth"
        }

        # 关节位置
        self.joint_positions = {
            "dim": 7,
            "names": [
                "translate_x", "translate_y", "rotate_z",
                "joint_lift", "joint_arm_l0", "joint_arm_l1",
                "joint_arm_l2", "joint_arm_l3", "joint_wrist_yaw",
                "joint_gripper_finger_left"
            ]
        }

        # 关节速度
        self.joint_velocities = {
            "dim": 7
        }

        # 夹爪状态
        self.gripper_state = {
            "dim": 1,
            "range": [0.0, 0.04]  # meters
        }

        # 末端执行器位姿
        self.ee_pose = {
            "dim": 7,  # [x, y, z, qw, qx, qy, qz]
            "frame": "base_link"
        }

        # 语言指令
        self.instruction = {
            "type": "text",
            "example": "Pick up the red cup and place it on the table"
        }

        # 目标物体位姿（可选）
        self.object_pose = {
            "position": {"dim": 3},
            "orientation": {"dim": 4},
            "optional": True
        }
```

---

## 二、Pick and Place Episode 格式

### 2.1 Episode 数据结构

```python
# dataset/episode/manipulation_episode.py
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

@dataclass
class PickAndPlaceEpisode:
    """Pick and Place 任务 Episode"""

    # 通用字段
    episode_id: str
    task_type: str = "pick_and_place"
    scene_id: str = "table_setup_001"

    # 机器人配置
    robot_config: Dict[str, Any] = None
    """
    {
        "robot_type": "stretch",
        "dof": 7,
        "init_pose": {
            "base": [0.0, 0.0, 0.0],
            "joint_positions": [0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        }
    }
    """

    # 任务目标
    task_goal: Dict[str, Any] = None
    """
    Pick and Place 任务目标配置
    {
        "target_object": {
            "name": "cup_red",
            "usd_path": "objects/cup_red.usd",
            "initial_position": [0.5, 0.0, 0.8],
            "initial_rotation": [0, 0, 0, 1]
        },
        "target_location": {
            "type": "position",  # "position" | "container" | "surface"
            "position": [0.7, 0.2, 0.8],
            "radius": 0.1
        },
        "success_criteria": {
            "type": "grasp_and_lift",  # "grasp_and_lift" | "place_at_location"
            "lift_height": 0.1,        # meters
            "place_tolerance": 0.05     # meters
        }
    }
    """

    # 场景物体
    scene_objects: List[Dict[str, Any]] = None
    """
    场景中的物体列表
    [
        {
            "name": "table",
            "usd_path": "furniture/table.usd",
            "position": [0.5, 0.0, 0.4],
            "rotation": [0, 0, 0, 1],
            "scale": [1.0, 1.0, 1.0],
            "static": True
        },
        {
            "name": "cup_red",
            "usd_path": "objects/cup_red.usd",
            "position": [0.5, 0.0, 0.8],
            "rotation": [0, 0, 0, 1],
            "graspable": True
        },
        {
            "name": "bowl_blue",
            "usd_path": "objects/bowl_blue.usd",
            "position": [0.7, 0.2, 0.8],
            "rotation": [0, 0, 0, 1]
        }
    ]
    """

    # 任务指令
    instruction: Dict[str, Any] = None
    """
    {
        "text": "Pick up the red cup and place it in the blue bowl",
        "tokens": ["pick", "up", "the", "red", "cup", ...]
    }
    """

    # 参考轨迹（可选，用于轨迹相似度计算）
    reference_trajectory: Optional[Dict[str, Any]] = None
    """
    {
        "qpos_sequence": [[...], [...], ...],  # N x 7
        "ee_pose_sequence": [[...], [...], ...],  # N x 7
        "keypoints": ["grasp_start", "lift_complete", "place_start", "place_complete"]
    }
    """

    # 仿真参数
    sim_params: Dict[str, Any] = None
    """
    {
        "max_steps": 500,
        "time_step": 0.01,
        "physics_engine": "physx",
        "gravity": [0, 0, -9.81]
    }
    """

    def __post_init__(self):
        if self.robot_config is None:
            self.robot_config = {
                "robot_type": "stretch",
                "dof": 7
            }
        if self.task_goal is None:
            self.task_goal = {}
        if self.scene_objects is None:
            self.scene_objects = []
        if self.instruction is None:
            self.instruction = {}
        if self.sim_params is None:
            self.sim_params = {
                "max_steps": 500,
                "time_step": 0.01
            }
```

### 2.2 Episode JSON 示例

```json
{
  "episode_id": "stretch_pick_place_001",
  "task_type": "pick_and_place",
  "scene_id": "table_setup_001",

  "robot_config": {
    "robot_type": "stretch",
    "dof": 7,
    "init_pose": {
      "base": [0.0, 0.0, 0.0],
      "joint_positions": [0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    }
  },

  "task_goal": {
    "target_object": {
      "name": "cup_red",
      "initial_position": [0.5, 0.0, 0.8]
    },
    "target_location": {
      "type": "position",
      "position": [0.7, 0.2, 0.8],
      "radius": 0.1
    },
    "success_criteria": {
      "type": "grasp_and_lift",
      "lift_height": 0.1,
      "place_tolerance": 0.05
    }
  },

  "scene_objects": [
    {
      "name": "table",
      "usd_path": "furniture/table.usd",
      "position": [0.5, 0.0, 0.4],
      "static": true
    },
    {
      "name": "cup_red",
      "usd_path": "objects/cup_red.usd",
      "position": [0.5, 0.0, 0.8],
      "graspable": true
    }
  ],

  "instruction": {
    "text": "Pick up the red cup and place it at the target location"
  },

  "sim_params": {
    "max_steps": 500,
    "time_step": 0.01
  }
}
```

---

## 三、Pick and Place 评测指标

### 3.1 Success 指标

```python
# metrics/manipulation/pick_place_metrics.py
from .base import Metric

class PickPlaceSuccess(Metric):
    """
    Pick and Place 成功率

    成功条件：
    1. 物体被抓取（gripper_closed + object_attached）
    2. 物体被提起（object_z > initial_z + lift_height）
    3. 物体被放置到目标位置（distance(object, target) < tolerance）
    """

    def __init__(self, lift_height: float = 0.1, place_tolerance: float = 0.05):
        self.lift_height = lift_height
        self.place_tolerance = place_tolerance

        # 状态追踪
        self.object_grasped = False
        self.object_lifted = False
        self.object_placed = False
        self.success = False

    def reset(self, episode: PickAndPlaceEpisode, task):
        self.lift_height = episode.task_goal["success_criteria"].get("lift_height", 0.1)
        self.place_tolerance = episode.task_goal["success_criteria"].get("place_tolerance", 0.05)

        self.object_grasped = False
        self.object_lifted = False
        self.object_placed = False
        self.success = False

        self.initial_object_pos = episode.task_goal["target_object"]["initial_position"]
        self.target_location = episode.task_goal["target_location"]["position"]

    def update(self, state: Dict[str, Any]):
        """
        更新成功状态

        Args:
            state: {
                "gripper_state": float,           # 夹爪开合度
                "object_grasped": bool,            # 物体是否被抓取
                "object_position": [x, y, z],       # 物体当前位置
                "ee_position": [x, y, z]            # 末端执行器位置
            }
        """
        # 检查抓取
        if state.get("object_grasped", False) and state.get("gripper_state", 0.1) < 0.01:
            self.object_grasped = True

        # 检查提起
        if self.object_grasped:
            current_object_z = state.get("object_position", [0, 0, 0])[2]
            initial_object_z = self.initial_object_pos[2]
            if current_object_z > initial_object_z + self.lift_height:
                self.object_lifted = True

        # 检查放置
        if self.object_lifted:
            import numpy as np
            object_pos = np.array(state.get("object_position", [0, 0, 0]))
            target_pos = np.array(self.target_location)
            distance = np.linalg.norm(object_pos - target_pos)

            if distance < self.place_tolerance:
                self.object_placed = True

        # 综合成功条件
        self.success = self.object_grasped and self.object_lifted and self.object_placed

    def get_metric(self) -> float:
        return 1.0 if self.success else 0.0
```

### 3.2 Completion Rate 指标

```python
class PickPlaceCompletionRate(Metric):
    """
    Pick and Place 完成率（渐进指标）

    进度计算：
    - Phase 1 (Reach): 距离目标的接近度
    - Phase 2 (Grasp): 夹爪闭合度
    - Phase 3 (Lift): 提起高度与目标高度的比例
    - Phase 4 (Place): 与目标位置的接近度
    """

    def __init__(self):
        self.phase = "reach"
        self.progress = 0.0

    def reset(self, episode: PickAndPlaceEpisode, task):
        self.phase = "reach"
        self.progress = 0.0

    def update(self, state: Dict[str, Any]):
        """
        更新进度

        Args:
            state: {
                "ee_position": [x, y, z],
                "object_position": [x, y, z],
                "gripper_state": float,
                "phase": str  # "reach" | "grasp" | "lift" | "place"
            }
        """
        import numpy as np

        phase = state.get("phase", "reach")

        if phase == "reach":
            # 计算末端执行器与目标的接近度
            ee_pos = np.array(state.get("ee_position", [0, 0, 0]))
            object_pos = np.array(state.get("object_position", [0, 0, 0]))
            distance = np.linalg.norm(ee_pos - object_pos)

            # 假设最大抓取距离为 0.5m
            max_distance = 0.5
            self.progress = max(0.0, 1.0 - (distance / max_distance))

        elif phase == "grasp":
            # 夹爪闭合度
            gripper_open = state.get("gripper_state", 0.04)
            self.progress = 1.0 - (gripper_open / 0.04)

        elif phase == "lift":
            # 提起高度比例
            current_z = state.get("object_position", [0, 0, 0])[2]
            initial_z = state.get("initial_object_z", 0.8)
            target_lift = state.get("target_lift_height", 0.1)
            self.progress = min(1.0, (current_z - initial_z) / target_lift)

        elif phase == "place":
            # 与目标位置的接近度
            object_pos = np.array(state.get("object_position", [0, 0, 0]))
            target_pos = np.array(state.get("target_position", [0, 0, 0]))
            distance = np.linalg.norm(object_pos - target_pos)
            tolerance = state.get("place_tolerance", 0.05)
            self.progress = max(0.0, 1.0 - (distance / tolerance))

    def get_metric(self) -> float:
        return self.progress
```

### 3.3 Trajectory Similarity 指标

```python
class PickPlaceTrajectorySimilarity(Metric):
    """
    Pick and Place 轨迹相似度

    使用 DTW (Dynamic Time Warping) 计算与参考轨迹的相似度

    关键点对齐：
    - grasp_start: 开始抓取的帧
    - lift_complete: 提起完成的帧
    - place_start: 开始放置的帧
    - place_complete: 放置完成的帧
    """

    def __init__(self):
        self.trajectory = {
            "qpos": [],
            "ee_pose": []
        }
        self.reference = None
        self.keypoints = {
            "grasp_start": None,
            "lift_complete": None,
            "place_start": None,
            "place_complete": None
        }

    def reset(self, episode: PickAndPlaceEpisode, task):
        self.trajectory = {"qpos": [], "ee_pose": []}

        # 加载参考轨迹
        if episode.reference_trajectory:
            self.reference = episode.reference_trajectory
            self.keypoints = {k: None for k in self.reference.get("keypoints", [])}

    def update(self, state: Dict[str, Any]):
        """记录当前状态到轨迹"""
        self.trajectory["qpos"].append(state.get("qpos", []))
        self.trajectory["ee_pose"].append(state.get("ee_pose", []))

    def get_metric(self) -> float:
        if self.reference is None:
            return 0.0

        import numpy as np
        from dtaidistance import dtw

        # 计算 qpos 序列的 DTW 距离
        traj_array = np.array(self.trajectory["qpos"])
        ref_array = np.array(self.reference["qpos_sequence"])

        distance = dtw.distance(traj_array, ref_array)

        # 归一化
        max_distance = np.linalg.norm(traj_array[0] - ref_array[-1]) * len(traj_array)
        similarity = 1.0 - (distance / max_distance) if max_distance > 0 else 1.0

        return max(0.0, similarity)
```

---

## 四、观测和动作格式

### 4.1 观测格式

```python
# 仿真器返回的观测格式
observation = {
    # RGB 图像（头部相机）
    "rgb_head": "<base64_encoded_image>",

    # RGB 图像（腕部相机）
    "rgb_wrist": "<base64_encoded_image>",

    # Depth 图像（头部相机）
    "depth_head": "<base64_encoded_depth>",

    # 关节位置 (7-dim)
    "qpos": [
        0.0,    # translate_x (m)
        0.0,    # translate_y (m)
        0.0,    # rotate_z (rad)
        0.5,    # joint_lift (m)
        0.0,    # joint_arm_l0 (rad)
        0.0,    # joint_arm_l1 (rad)
        0.0,    # joint_arm_l2 (rad)
        0.0,    # joint_arm_l3 (rad)
        0.0,    # joint_wrist_yaw (rad)
        0.04     # joint_gripper_finger_left (m, open)
    ],

    # 关节速度 (7-dim)
    "qvel": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],

    # 末端执行器位姿 (7-dim)
    "ee_pose": [
        0.5,    # x (m)
        0.0,    # y (m)
        0.8,    # z (m)
        1.0,    # qw
        0.0,    # qx
        0.0,    # qy
        0.0      # qz
    ],

    # 夹爪状态 (1-dim)
    "gripper_state": 0.04,  # opening distance (m)

    # 任务信息
    "instruction": {
        "text": "Pick up the red cup and place it at the target location"
    },

    # 物体信息（可选，用于调试）
    "object_info": {
        "target_object_position": [0.5, 0.0, 0.8],
        "target_location_position": [0.7, 0.2, 0.8]
    }
}
```

### 4.2 动作格式

```python
# 参赛者服务返回的动作格式
action = {
    # 方式1：直接关节位置控制
    "type": "joint_position",

    "qpos": [
        0.05,   # translate_x
        0.0,    # translate_y
        0.1,    # rotate_z
        0.6,    # joint_lift
        -0.5,   # joint_arm_l0
        1.0,    # joint_arm_l1
        0.0,    # joint_arm_l2
        0.0,    # joint_arm_l3
        0.0,    # joint_wrist_yaw
        0.02     # gripper (partially closed)
    ]
}

# 或

action = {
    # 方式2：末端执行器目标位姿（由 IK 求解）
    "type": "ee_pose_target",

    "ee_pose_target": [
        0.5,    # x (m)
        0.0,    # y (m)
        0.9,    # z (m)
        1.0,    # qw
        0.0,    # qx
        0.0,    # qy
        0.0      # qz
    ],

    "gripper_action": 0.02  # gripper opening (m)
}
```

---

## 五、WebSocket 协议示例

### 5.1 Reset Episode 消息

```json
// Evaluator → Agent
{
  "type": "reset_episode",
  "session_id": "uuid-xxx",
  "episode": {
    "episode_id": "stretch_pick_place_001",
    "task_type": "pick_and_place",
    "robot_config": {
      "robot_type": "stretch",
      "dof": 7
    },
    "task_goal": {
      "target_object": {
        "name": "cup_red",
        "initial_position": [0.5, 0.0, 0.8]
      },
      "target_location": {
        "position": [0.7, 0.2, 0.8],
        "radius": 0.1
      },
      "success_criteria": {
        "lift_height": 0.1,
        "place_tolerance": 0.05
      }
    },
    "scene_objects": [...],
    "instruction": {
      "text": "Pick up the red cup and place it at the target location"
    }
  }
}
```

### 5.2 Observation 消息

```json
// Evaluator → Agent
{
  "type": "get_action",
  "session_id": "uuid-xxx",
  "observation": {
    "rgb_head": "<base64_encoded>",
    "rgb_wrist": "<base64_encoded>",
    "depth_head": "<base64_encoded>",
    "qpos": [0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04],
    "qvel": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "ee_pose": [0.5, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0],
    "gripper_state": 0.04,
    "instruction": {
      "text": "Pick up the red cup and place it at the target location"
    }
  }
}
```

### 5.3 Action 消息

```json
// Agent → Evaluator
{
  "type": "action",
  "session_id": "uuid-xxx",
  "action": {
    "type": "joint_position",
    "qpos": [0.05, 0.0, 0.1, 0.6, -0.5, 1.0, 0.0, 0.0, 0.0, 0.02]
  }
}
```

### 5.4 Episode End 消息

```json
// Evaluator → Agent
{
  "type": "episode_end",
  "session_id": "uuid-xxx",
  "status": "success",
  "metrics": {
    "success": 1.0,
    "completion_rate": 1.0,
    "trajectory_similarity": 0.85
  },
  "num_steps": 245
}
```

---

## 六、与 RoboVerse 的对比

| 方面 | RoboVerse | 本设计 |
|------|------------|--------|
| **任务类型** | Pick, Place, Stack, Close, Open | Pick and Place |
| **机器人** | Franka, H1, Unitree G1 | Stretch |
| **数据格式** | HDF5 存储 action/obs/trajectory | 仅 Episode 定义（不存储） |
| **随机化** | L0-L3 分层随机化 | 第一阶段不支持 |
| **指标** | Success, Completion Rate, Trajectory Similarity | Success, Completion Rate, Trajectory Similarity |
| **接口** | 统一仿真器后端 | 抽象接口（不绑定后端） |

---

## 七、后续扩展方向

1. **更多任务类型**
   - Stacking: 堆叠物体到指定高度
   - Drawer Opening: 打开抽屉
   - Door Opening: 打开门

2. **更多机器人**
   - Franka/Panda（固定底座机械臂）
   - Unitree G1（人形机器人）

3. **泛化评测**
   - 光照随机化
   - 纹理随机化
   - 物体位置/姿态随机化

4. **多模态输入**
   - 触觉传感器
   - 力觉传感器
   - 音频输入

---

*文档版本: 1.0*
*创建时间: 2026-02-12*
*基于: RoboVerse 设计 + 用户确认需求*

# 操作任务扩展设计

> 基于 RoboVerse 设计，为 VLN 评测器添加操作任务支持

---

## 一、设计原则（基于用户需求）

| 需求点 | 设计含义 |
|--------|----------|
| **抽象接口** | 不绑定具体仿真器后端，设计通用接口 |
| **评测专注** | 评测器定义任务和指标，不存储 action/observation |
| **服务化决策** | 通过 `action(obs)` 接口调用参赛者服务获取动作 |
| **暂无随机化** | 第一阶段专注于同分布评测 |

---

## 二、需要修改的核心组件

### 2.1 架构对比

```
┌─────────────────────────────────────────────────────────────────┐
│                     VLN 评测器（现有）                        │
├─────────────────────────────────────────────────────────────────┤
│  Task: VLN                                                      │
│  ├─ Action Space: [move_forward, turn_left, turn_right, stop]       │
│  ├─ Observation: [RGB, Depth, GPS, Compass, Instruction]        │
│  └─ Metrics: [Success, SPL, Navigation Error, DTW]                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ 扩展
┌─────────────────────────────────────────────────────────────────┐
│              Unified Evaluator（扩展后）                          │
├─────────────────────────────────────────────────────────────────┤
│  TaskRegistry:                                                  │
│  ├─ VLN (Navigation)                                            │
│  └─ Manipulation (新增)                                         │
│                                                                │
│  SimulatorBackend:                                                │
│  ├─ NavSimulator（现有）                                        │
│  └─ ManipSimulator（新增抽象接口）                               │
│                                                                │
│  MetricsRegistry:                                                │
│  ├─ NavigationMetrics（现有）                                     │
│  └─ ManipulationMetrics（新增）                                   │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 组件修改清单

| 组件 | 当前状态 | 需要的修改 | 文件位置 |
|------|----------|------------|----------|
| **Task 基类** | 仅支持 VLN | 抽象化，支持多任务类型 | `tasks/base.py` |
| **Task Registry** | VLN 注册表 | 添加操作任务注册 | `tasks/registry.py` |
| **Simulator 接口** | 导航仿真器 | 抽象接口，支持多后端 | `simulator/base.py` |
| **Episode 数据类** | VLN 专用字段 | 泛化，支持任务扩展 | `dataset/episode.py` |
| **Metrics 系统** | 导航指标 | 抽象接口，支持任务特定指标 | `metrics/base.py` |
| **API 协议** | VLN 专用消息 | 扩展消息类型支持操作任务 | `api/protocol.py` |

---

## 三、具体修改方案

### 3.1 Task 抽象层

#### 现有设计（VLN 专用）

```python
# tasks/vln.py（现有）
class VLNTask:
    actions = ["move_forward", "turn_left", "turn_right", "stop"]
    sensors = ["rgb", "depth", "gps", "compass", "instruction"]
    metrics = ["success", "spl", "navigation_error", "dtw"]
```

#### 抽象化改造

```python
# tasks/base.py（新建）
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class ActionSpace:
    """动作空间定义"""
    name: str                          # "discrete" | "continuous" | "hybrid"
    actions: List[str]                   # 离散动作列表
    shape: Optional[tuple] = None        # 连续动作形状
    range: Optional[tuple] = None       # 连续动作范围

@dataclass
class SensorSuite:
    """传感器套件定义"""
    sensors: Dict[str, Dict[str, Any]]   # {"sensor_name": {"width": 640, ...}}

class BaseTask(ABC):
    """任务抽象基类"""

    @property
    @abstractmethod
    def task_type(self) -> str:
        """任务类型标识"""
        pass

    @property
    @abstractmethod
    def action_space(self) -> ActionSpace:
        """动作空间定义"""
        pass

    @property
    @abstractmethod
    def sensor_suite(self) -> SensorSuite:
        """传感器套件定义"""
        pass

    @property
    @abstractmethod
    def metric_names(self) -> List[str]:
        """指标名称列表"""
        pass

    @abstractmethod
    def check_episode_complete(self, state: Dict[str, Any]) -> bool:
        """检查 Episode 是否完成"""
        pass
```

#### VLN 任务适配

```python
# tasks/vln.py（修改）
from .base import BaseTask, ActionSpace, SensorSuite

class VLNTask(BaseTask):
    @property
    def task_type(self) -> str:
        return "vln"

    @property
    def action_space(self) -> ActionSpace:
        return ActionSpace(
            name="discrete",
            actions=["move_forward", "turn_left", "turn_right", "stop"]
        )

    @property
    def sensor_suite(self) -> SensorSuite:
        return SensorSuite(sensors={
            "rgb": {"width": 640, "height": 480, "encoding": "base64"},
            "depth": {"width": 640, "height": 480, "encoding": "base64"},
            "gps": {"dim": 3},
            "compass": {"dim": 1},
            "instruction": {"type": "text"}
        })

    @property
    def metric_names(self) -> List[str]:
        return ["success", "spl", "navigation_error", "dtw"]

    def check_episode_complete(self, state: Dict[str, Any]) -> bool:
        # VLN 特定完成条件
        distance = state.get("distance_to_goal", float('inf'))
        return distance < self.config.success_distance
```

#### 操作任务实现（新增）

```python
# tasks/manipulation.py（新建）
from .base import BaseTask, ActionSpace, SensorSuite

class ManipulationTask(BaseTask):
    """操作任务基类"""

    @property
    def task_type(self) -> str:
        return "manipulation"

    @property
    def action_space(self) -> ActionSpace:
        """
        操作任务的动作空间定义
        - 根据用户需求：action 是具体的本体状态（关节位置/速度）
        - 由参赛者服务的 action(obs) 接口返回
        """
        return ActionSpace(
            name="continuous",
            shape=None,  # 由具体机器人配置决定
            range=None
        )

    @property
    def sensor_suite(self) -> SensorSuite:
        """
        操作任务的传感器配置
        参考 RoboVerse：RGB, Depth, 关节状态 (qpos), 关节速度 (qvel)
        """
        return SensorSuite(sensors={
            "rgb": {"width": 640, "height": 480, "encoding": "base64"},
            "depth": {"width": 640, "height": 480, "encoding": "base64"},
            "qpos": {"dim": None},  # 由具体机器人 DOF 决定
            "qvel": {"dim": None},
            "ee_pose": {"dim": 7},  # 末端执行器位姿 (position + quaternion)
            "instruction": {"type": "text"}  # 语言指令（如需要）
        })

    @property
    def metric_names(self) -> List[str]:
        """
        操作任务指标
        参考 RoboVerse：Success, Trajectory Similarity
        """
        return ["success", "completion_rate", "trajectory_similarity"]

    def check_episode_complete(self, state: Dict[str, Any]) -> bool:
        """
        操作任务完成条件检查
        - Success: 目标达成（物体被抓取、门被打开等）
        - Timeout: 超过最大步数
        - Error: 物理错误（碰撞、摔倒等）
        """
        if state.get("success", False):
            return True
        if state.get("num_steps", 0) >= self.config.max_steps:
            return True
        if state.get("error", False):
            return True
        return False
```

### 3.2 Episode 数据结构扩展

#### 现有设计

```python
# dataset/episode.py（现有）
@dataclass
class Episode:
    episode_id: str
    scene_id: str
    start_position: List[float]  # [x, y, z]
    start_rotation: List[float]  # quaternion
    instruction: Dict           # VLN 指令
    reference_path: List[List[float]]
    goals: List[Dict]
```

#### 泛化设计

```python
# dataset/episode.py（修改）
from typing import Union, Dict, Any
from dataclasses import dataclass, field

@dataclass
class Episode:
    """
    泛化的 Episode 数据结构
    支持导航任务和操作任务
    """
    # 通用字段
    episode_id: str
    task_type: str                # "vln" | "manipulation"
    scene_id: str

    # 任务特定配置（使用 Dict 存储灵活配置）
    task_config: Dict[str, Any] = field(default_factory=dict)

    # 初始状态（通用）
    start_position: List[float] = None      # 导航: [x,y,z], 操作: [x,y,z] 或关节位置
    start_rotation: List[float] = None      # 导航: quaternion, 操作: None

    # 任务目标
    goals: Dict[str, Any] = field(default_factory=dict)
    """
    导航任务：
        {"position": [x,y,z], "radius": 0.2}

    操作任务：
        {
            "target_object": "cup_red",
            "target_pose": [x,y,z,qw,qx,qy,qz],
            "success_criteria": "grasp_and_lift" | "place_at_location"
        }
    """

    # 指令（多模态）
    instruction: Dict[str, Any] = field(default_factory=dict)
    """
    导航任务：
        {"text": "Walk to the kitchen", "tokens": [...]}

    操作任务：
        {"text": "Pick up the red cup", or None}
    """

    # 参考数据（可选）
    reference_path: List[List[float]] = None     # 导航：参考路径
    reference_trajectory: Dict[str, Any] = None   # 操作：参考轨迹（qpos 序列）

    # 场景配置（操作任务专用）
    scene_objects: List[Dict[str, Any]] = None
    """
    操作任务中的物体配置
    [
        {"name": "cup_red", "usd_path": "...", "position": [x,y,z]},
        {"name": "table", "usd_path": "...", "position": [x,y,z]}
    ]
    """

    # 机器人配置（操作任务专用）
    robot_config: Dict[str, Any] = None
    """
    机器人配置
    {
        "robot_type": "franka" | "stretch" | "humanoid",
        "dof": 7,
        "end_effector": "gripper" | "suction"
    }
    """

    def __post_init__(self):
        """初始化后验证"""
        if self.task_type not in ["vln", "manipulation"]:
            raise ValueError(f"Unknown task_type: {self.task_type}")

        # 设置默认值
        if self.task_config is None:
            self.task_config = {}
        if self.goals is None:
            self.goals = {}
```

### 3.3 Simulator 抽象接口

#### 现有设计（导航专用）

```python
# simulator/base.py（现有）
class NavSimulator:
    def reset(self, episode: Episode) -> Observation:
        pass

    def step(self, action: str) -> Observation:
        pass

    def get_agent_state(self) -> Dict:
        pass
```

#### 抽象化改造

```python
# simulator/base.py（修改）
from abc import ABC, abstractmethod
from typing import Dict, Any, Union

class BaseSimulator(ABC):
    """仿真器抽象基类"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.backend = config.get("backend", "custom")

    @abstractmethod
    def reset(self, episode: Episode) -> Dict[str, Any]:
        """
        重置仿真器到 Episode 初始状态

        Returns:
            initial_observation: 初始观测
        """
        pass

    @abstractmethod
    def step(self, action: Union[str, Dict[str, Any], List[float]]) -> Dict[str, Any]:
        """
        执行一步仿真

        Args:
            action: 动作（可能是离散动作名、连续动作向量、本体状态）

        Returns:
            observation: 下一步观测
            reward: 奖励值（可选，用于 RL）
            done: 是否结束
            info: 额外信息
        """
        pass

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """获取当前环境状态"""
        pass

    @abstractmethod
    def is_navigable(self, position: List[float]) -> bool:
        """检查位置是否可导航（导航任务专用）"""
        pass

    def close(self):
        """关闭仿真器，释放资源"""
        pass

class NavSimulator(BaseSimulator):
    """导航仿真器（现有实现适配）"""
    pass

class ManipSimulator(BaseSimulator):
    """操作仿真器抽象接口（新增）"""

    @abstractmethod
    def set_robot_state(self, qpos: List[float], qvel: List[float] = None):
        """
        设置机器人状态

        Args:
            qpos: 关节位置
            qvel: 关节速度（可选）
        """
        pass

    @abstractmethod
    def get_robot_state(self) -> Dict[str, Any]:
        """
        获取机器人状态

        Returns:
            {
                "qpos": List[float],      # 关节位置
                "qvel": List[float],      # 关节速度
                "ee_pose": List[float],    # 末端执行器位姿
                "gripper_state": float     # 夹爪开合度
            }
        """
        pass

    @abstractmethod
    def check_success(self) -> bool:
        """
        检查任务成功条件

        Returns:
            success: 是否满足成功条件
        """
        pass
```

### 3.4 Metrics 系统扩展

#### 现有设计

```python
# metrics/base.py（现有）
class Metric:
    def reset(self, episode, task):
        pass

    def update(self, *args, **kwargs):
        pass

    def get_metric(self) -> float:
        pass
```

#### 操作任务指标（新增）

```python
# metrics/manipulation.py（新建）
from .base import Metric

class Success(Metric):
    """操作任务成功率"""

    def __init__(self):
        self.success = False

    def reset(self, episode, task):
        self.success = False

    def update(self, state: Dict[str, Any]):
        """
        更新成功状态
        - state["success"]: 由 simulator.check_success() 返回
        """
        self.success = state.get("success", False)

    def get_metric(self) -> float:
        return 1.0 if self.success else 0.0

class CompletionRate(Metric):
    """任务完成率（渐进指标）

    例如：
    - Pick 任务：物体被提起的高度 / 目标高度
    - Place 任务：物体与目标位置的接近度
    """

    def __init__(self):
        self.progress = 0.0

    def reset(self, episode, task):
        self.progress = 0.0

    def update(self, state: Dict[str, Any]):
        self.progress = state.get("progress", 0.0)

    def get_metric(self) -> float:
        return self.progress

class TrajectorySimilarity(Metric):
    """
    轨迹相似度（参考 RoboVerse）

    计算方式：
    1. DTW (Dynamic Time Warping) 距离
    2. 与参考轨迹的逐帧误差
    """

    def __init__(self):
        self.trajectory = []
        self.reference = None

    def reset(self, episode, task):
        self.trajectory = []
        self.reference = episode.reference_trajectory

    def update(self, state: Dict[str, Any]):
        """
        记录当前状态到轨迹
        - state["qpos"]: 当前关节位置
        """
        self.trajectory.append(state.get("qpos", []))

    def get_metric(self) -> float:
        if self.reference is None:
            return 0.0

        # 计算 DTW 相似度
        import numpy as np
        from dtaidistance import dtw

        traj_array = np.array(self.trajectory)
        ref_array = np.array(self.reference)

        # DTW 距离
        distance = dtw.distance(traj_array, ref_array)

        # 归一化相似度（可根据需要调整）
        max_distance = np.linalg.norm(traj_array[0] - ref_array[-1]) * len(traj_array)
        similarity = 1.0 - (distance / max_distance) if max_distance > 0 else 1.0

        return max(0.0, similarity)
```

### 3.5 API 协议扩展

#### 消息类型扩展

```python
# api/protocol.py（修改）
from typing import Dict, Any, List, Union

class MessageType:
    """消息类型枚举"""
    CONNECT = "connect"
    CONNECTED = "connected"
    RESET_EPISODE = "reset_episode"
    EPISODE_READY = "episode_ready"
    GET_ACTION = "get_action"
    ACTION = "action"
    EPISODE_END = "episode_end"
    HEARTBEAT = "heartbeat"
    ERROR = "error"
    DISCONNECT = "disconnect"

def create_reset_message(episode: Episode) -> Dict[str, Any]:
    """
    创建 Reset Episode 消息（泛化版本）

    根据 episode.task_type 选择不同的消息格式
    """
    base_msg = {
        "type": MessageType.RESET_EPISODE,
        "session_id": episode.session_id,
        "episode": {
            "episode_id": episode.episode_id,
            "task_type": episode.task_type,
        }
    }

    if episode.task_type == "vln":
        base_msg["episode"].update({
            "instruction": episode.instruction,
            "start_position": episode.start_position,
            "start_rotation": episode.start_rotation,
            "goals": episode.goals
        })

    elif episode.task_type == "manipulation":
        base_msg["episode"].update({
            "instruction": episode.instruction,
            "robot_config": episode.robot_config,
            "scene_objects": episode.scene_objects,
            "goals": episode.goals,
            "reference_trajectory": episode.reference_trajectory
        })

    return base_msg

def create_observation_message(
    task_type: str,
    sensor_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    创建观测消息（泛化版本）

    Args:
        task_type: 任务类型
        sensor_data: 传感器数据字典

    Returns:
        消息字典
    """
    return {
        "type": MessageType.GET_ACTION,
        "task_type": task_type,
        "observation": sensor_data
    }

def parse_action_message(message: Dict[str, Any]) -> Union[str, List[float], Dict]:
    """
    解析动作消息（泛化版本）

    Returns:
        action: 根据任务类型返回不同格式
        - VLN: str ("move_forward")
        - Manipulation: List[float] (qpos) 或 Dict (本体状态)
    """
    return message.get("action")
```

#### WebSocket 协议流程

```
┌──────────────┐                    ┌──────────────┐                    ┌──────────────┐
│  Evaluator   │                    │  WebSocket    │                    │  Agent Service│
│  (评测器)     │                    │  Server       │                    │  (参赛者)      │
└──────┬───────┘                    └──────┬───────┘                    └──────┬───────┘
       │                                  │                                  │
       │  1. reset_episode(episode)       │                                  │
       │  ───────────────────────────────>│                                  │
       │                                  │  2. WS: Send reset_episode        │
       │                                  │  ──────────────────────────────────>│
       │                                  │                          (task_type, config)  │
       │                                  │                                  │
       │                                  │  3. WS: Wait for ready          │
       │                                  │  <──────────────────────────────────│
       │                                  │                          (agent_ready)      │
       │                                  │                                  │
       │  4. simulator.reset()           │                                  │
       │  <───────────────────────────────│                                  │
       │                                  │                                  │
       │  5. WS: Send get_action         │                                  │
       │  <───────────────────────────────│                                  │
       │                                  │  ──────────────────────────────────>│
       │                                  │                    (observation: obs)    │
       │                                  │                                  │
       │                                  │  6. agent.act(obs)            │
       │                                  │                                  │
       │                                  │  7. WS: Receive action         │
       │                                  │  <──────────────────────────────────│
       │                                  │                    (action: a)        │
       │                                  │                                  │
       │  8. simulator.step(action)      │                                  │
       │  ───────────────────────────────>│                                  │
       │                                  │                                  │
       │  [Repeat 5-8 until done]        │                                  │
       │                                  │                                  │
       │  9. compute_metrics()           │                                  │
       │                                  │                                  │
       │  10. WS: Send episode_end       │                                  │
       │  ───────────────────────────────>│                                  │
       │                                  │  ──────────────────────────────────>│
       │                                  │                (status, metrics)     │
       │                                  │                                  │
       │                                  │  11. WS: Close                 │
       │                                  │  <──────────────────────────────────│
```

### 3.6 Task Registry 扩展

```python
# tasks/registry.py（修改）
from typing import Dict, Type
from .base import BaseTask
from .vln import VLNTask
from .manipulation import ManipulationTask

class TaskRegistry:
    """任务注册表"""

    _tasks: Dict[str, Type[BaseTask]] = {}

    @classmethod
    def register(cls, name: str):
        """注册装饰器"""
        def decorator(task_class: Type[BaseTask]):
            cls._tasks[name] = task_class
            return task_class
        return decorator

    @classmethod
    def get_task(cls, name: str, config: Dict) -> BaseTask:
        """获取任务实例"""
        if name not in cls._tasks:
            raise ValueError(f"Unknown task: {name}. Available: {list(cls._tasks.keys())}")
        return cls._tasks[name](config)

    @classmethod
    def list_tasks(cls) -> List[str]:
        """列出所有注册的任务"""
        return list(cls._tasks.keys())

# 注册内置任务
TaskRegistry.register("vln")(VLNTask)
TaskRegistry.register("manipulation")(ManipulationTask)
```

### 3.7 配置文件扩展

```yaml
# configs/benchmarks/manipulation_challenge.yaml（新建）
benchmark:
  name: "Manipulation Challenge 2024"
  description: "Robotic Manipulation Tasks"

task:
  type: "manipulation"
  config:
    robot_type: "franka"  # "franka" | "stretch" | "humanoid"
    dof: 7
    end_effector: "gripper"
    action_space: "continuous"  # "continuous" | "waypoint"
    max_steps: 500
    success_distance: 0.05  # meters

dataset:
  type: "manipulation"
  data_path: "data/datasets/manipulation/LIBERO/train"
  split: "val"
  num_episodes: 100

simulator:
  backend: "abstract"  # 不绑定具体后端
  sensors:
    rgb:
      width: 640
      height: 480
      encoding: "base64"
    depth:
      width: 640
      height: 480
      encoding: "base64"
    qpos:
      dim: 7  # 由 robot_config.dof 决定
    qvel:
      dim: 7
    ee_pose:
      dim: 7  # [x, y, z, qw, qx, qy, qz]

evaluation:
  max_steps: 500
  timeout: 300  # seconds
  metrics:
    - "success"
    - "completion_rate"
    - "trajectory_similarity"

agent_service:
  type: "remote"
  protocol: "websocket"
  endpoint: "localhost:8080"
  timeout: 30

output:
  log_dir: "logs/evaluations/manipulation"
  save_trajectories: true
  save_observations: false
```

---

## 四、实现优先级

### Phase 1: 核心抽象（必须）

- [ ] `BaseTask` 抽象类定义
- [ ] `VLNTask` 适配新抽象
- [ ] `ManipulationTask` 基本实现
- [ ] `TaskRegistry` 实现
- [ ] `Episode` 数据结构泛化

### Phase 2: 仿真器接口（必须）

- [ ] `BaseSimulator` 抽象类
- [ ] `NavSimulator` 适配
- [ ] `ManipSimulator` 接口定义
- [ ] 配置系统扩展

### Phase 3: 指标系统（重要）

- [ ] `Success` 指标
- [ ] `CompletionRate` 指标
- [ ] `TrajectorySimilarity` 指标
- [ ] 指标注册表

### Phase 4: API 协议（重要）

- [ ] 消息类型泛化
- [ ] WebSocket 协议扩展
- [ ] 参赛者 SDK 扩展

### Phase 5: 测试和文档（后续）

- [ ] 单元测试
- [ ] 集成测试
- [ ] 使用示例
- [ ] API 文档

---

## 五、关键设计决策总结

| 决策点 | 选择 | 理由 |
|--------|------|------|
| **仿真器绑定** | 抽象接口 | 用户明确只需要抽象接口 |
| **数据存储** | 仅任务定义，不存储 action/obs | 评测器职责分离 |
| **动作空间** | 通过参赛者服务获取 | 用户明确 action(obs) 接口模式 |
| **随机化** | 第一阶段不支持 | 用户明确暂不需要 |
| **指标系统** | 任务特定指标 | Success, Completion, TrajectorySimilarity |

---

## 六、与 RoboVerse 的对应关系

| RoboVerse 设计 | 本项目实现 |
|----------------|-------------|
| `RandomizationCfg` | 第一阶段不支持（用户需求） |
| `ScenarioCfg` | Episode.task_config 扩展 |
| `HDF5 Episode 存储` | 不需要（用户：评测只定义任务） |
| `Task 注册表` | `TaskRegistry` 实现 |
| `SceneRandomizer` | 暂不实现（无随机化需求） |
| `Metric 插件化` | `BaseMetric` + Registry 实现 |

---

## 七、下一步行动

请确认以下问题后开始实施：

1. **Simulator 后端选择**：虽然设计抽象接口，是否需要提供参考实现（如基于 MuJoCo/PyBullet 的简单示例）？

2. **操作任务类型**：第一阶段支持哪些具体任务？
   - Pick and Place
   - Stacking
   - Door Opening
   - 其他？

3. **机器人类型**：需要支持哪些机器人？
   - 机械臂（Franka, Panda）
   - 移动操作（Stretch）
   - 人形机器人

4. **数据集来源**：Episode 数据从哪里获取？
   - 现有数据集转换（LIBERO, RLBench）
   - 自定义格式

---

*文档版本: 1.0*
*创建时间: 2026-02-12*
*参考: RoboVerse 设计分析 + VLN 评测器架构*

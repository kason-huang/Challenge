# 操作任务支持 - Feature 设计文档

> 基于 VLN 评测架构的统一平台重构方案

---

## 一、设计目标

将现有 VLN 评测系统重构为**统一机器人任务评测平台**，同时支持：
- **导航任务** (VLN, ObjectNav, ImageNav)
- **操作任务** (Pick and Place, Stacking, Opening)

### 1.1 核心原则

| 原则 | 说明 |
|------|------|
| **统一抽象** | Task、Episode、Metric 的通用抽象 |
| **类型安全** | 使用 TypedDict 定义清晰的数据结构 |
| **插件化** | 通过 Registry 注册新任务和指标 |
| **协议兼容** | WebSocket 协议向后兼容并扩展 |
| **模块解耦** | 导航和操作任务实现分离 |

---

## 二、架构重构方案

### 2.1 架构对比

```
┌─────────────────────────────────────────────────────────────────────┐
│                    现有架构（VLN 专用）                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    VLNEvaluator                         │    │
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────┐  │    │
│  │  │  VLNTask       │  │  VLNEpisode     │  │  VLNMetrics │  │    │
│  │  │  (Concrete)    │  │  (Concrete)     │  │  (Concrete)  │  │    │
│  │  └────────────────┘  └────────────────┘  └────────────┘  │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼ 重构
┌─────────────────────────────────────────────────────────────────────┐
│                   重构后架构（统一平台）                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                UnifiedRobotEvaluator                   │    │
│  │                                                               │    │
│  │  ┌─────────────────────────────────────────────────────┐    │    │
│  │  │           Task Abstraction Layer               │    │    │
│  │  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  │    │    │
│  │  │  │ BaseTask    │  │ TaskRegistry│  │ ActionType  │  │    │    │
│  │  │  └────────────┘  └────────────┘  └────────────┘  │    │    │
│  │  └─────────────────────────────────────────────────────┘    │    │
│  │                                                               │    │
│  │  ┌─────────────────────────────────────────────────────┐    │    │
│  │  │          Concrete Tasks (Plugin-based)      │    │    │
│  │  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  │    │    │
│  │  │  │  VLNTask    │  │ManipTask   │  │ObjectNav   │  │    │    │
│  │  │  │ (Navigation)│  │(Manip.)    │  │ (Navigation)│  │    │    │
│  │  │  └────────────┘  └────────────┘  └────────────┘  │    │    │
│  │  └─────────────────────────────────────────────────────┘    │    │
│  │                                                               │    │
│  │  ┌─────────────────────────────────────────────────────┐    │    │
│  │  │           Unified Episode/Metric Layer          │    │    │
│  │  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  │    │    │
│  │  │  │ Episode     │  │ BaseMetric │  │MetricReg.  │  │    │    │
│  │  │  └────────────┘  └────────────┘  └────────────┘  │    │    │
│  │  └─────────────────────────────────────────────────────┘    │    │
│  │                                                               │    │
│  │  ┌─────────────────────────────────────────────────────┐    │    │
│  │  │          Concrete Metrics (Plugin-based)      │    │    │
│  │  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  │    │    │
│  │  │  │ Success/SPL │  │ ManipSuccess│  │TrajSimilar │  │    │    │
│  │  │  │ (Navigation)│  │(Manip.)    │  │ (Shared)    │  │    │    │
│  │  │  └────────────┘  └────────────┘  └────────────┘  │    │    │
│  │  └─────────────────────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 目录结构重构

```
robot_eval/                     # 重命名（原 nav_eval）
├── core/                      # 核心抽象层（新增）
│   ├── task/
│   │   ├── base.py           # BaseTask 抽象类
│   │   ├── registry.py       # TaskRegistry
│   │   └── types.py         # Task 相关类型定义
│   ├── episode/
│   │   ├── base.py           # 统一 Episode 基类
│   │   └── schemas.py       # Episode Schema 定义
│   ├── metric/
│   │   ├── base.py           # BaseMetric 抽象类
│   │   └── registry.py       # MetricRegistry
│   └── interfaces/
│       ├── action.py         # 统一 Action 类型
│       ├── observation.py    # 统一 Observation 类型
│       └── result.py        # 统一 Result 类型
│
├── tasks/                    # 任务实现（重构）
│   ├── base.py              # BaseTask 实现
│   ├── vln/
│   │   ├── task.py          # VLNTask (继承 BaseTask)
│   │   ├── actions.py       # VLN 动作定义
│   │   └── config.py        # VLN 配置
│   └── manipulation/          # 新增
│       ├── task.py          # ManipulationTask (继承 BaseTask)
│       ├── pick_place.py    # PickAndPlaceTask
│       ├── actions.py       # 操作动作定义
│       └── robots/          # 机器人配置
│           └── stretch.py   # Stretch 配置
│
├── metrics/                  # 指标实现（重构）
│   ├── base.py              # BaseMetric 实现
│   ├── navigation/           # 导航指标
│   │   ├── success.py       # Success (导航)
│   │   ├── spl.py           # SPL
│   │   └── dtw.py           # DTW
│   └── manipulation/         # 新增
│       ├── success.py       # Success (操作)
│       ├── completion.py    # CompletionRate
│       └── trajectory.py    # TrajectorySimilarity
│
├── simulator/                # 仿真器（扩展）
│   ├── base.py              # BaseSimulator 抽象
│   ├── navigation/           # 导航仿真器
│   │   └── nav_sim.py       # 现有实现
│   └── manipulation/         # 新增（抽象接口）
│       └── manip_sim.py     # ManipulationSimulator 接口
│
├── api/                     # API 层（扩展）
│   ├── protocol.py          # 统一协议定义
│   ├── websocket/
│   │   ├── server.py        # WebSocket 服务器
│   │   └── client.py        # WebSocket 客户端
│   └── messages/            # 消息定义
│       ├── base.py          # 基础消息类型
│       ├── navigation.py    # 导航任务消息
│       └── manipulation.py  # 操作任务消息（新增）
│
├── dataset/                  # 数据管理（扩展）
│   ├── episode.py           # Episode 数据类
│   ├── loader.py           # 数据加载器
│   └── formats/
│       ├── vln.py          # VLN 格式
│       └── manipulation.py  # 操作任务格式（新增）
│
└── registry/                 # 注册表（新增）
    ├── tasks.py             # 任务注册装饰器
    └── metrics.py           # 指标注册装饰器
```

---

## 三、核心抽象层设计

### 3.1 BaseTask 抽象

```python
# core/task/base.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Type
from dataclasses import dataclass
from core.interfaces.action import Action
from core.interfaces.observation import Observation
from core.metric.base import BaseMetric

@dataclass
class ActionSpace:
    """动作空间定义"""
    name: str                          # "discrete" | "continuous" | "hybrid"
    actions: Optional[List[str]] = None    # 离散动作列表
    shape: Optional[tuple] = None         # 连续动作形状
    range: Optional[tuple] = None         # 连续动作范围
    semantics: Optional[Dict[str, Any]] = None  # 语义标注

@dataclass
class SensorSuite:
    """传感器套件定义"""
    sensors: Dict[str, Dict[str, Any]]

    def get(self, name: str) -> Dict[str, Any]:
        """获取传感器配置"""
        return self.sensors.get(name, {})

class BaseTask(ABC):
    """任务抽象基类"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @property
    @abstractmethod
    def task_type(self) -> str:
        """任务类型标识"""
        pass

    @property
    @abstractmethod
    def task_name(self) -> str:
        """任务名称（人类可读）"""
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
    def metric_names(self) -> List[str]:
        """指标名称列表"""
        return self.config.get("metrics", [])

    @abstractmethod
    def check_episode_complete(self, state: Dict[str, Any]) -> bool:
        """检查 Episode 是否完成"""
        pass

    def get_metrics(self) -> List[Type[BaseMetric]]:
        """获取任务对应的指标类"""
        from registry.metrics import get_metrics
        return [get_metrics(name) for name in self.metric_names]
```

### 3.2 Episode 统一设计

```python
# core/episode/base.py
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

@dataclass
class BaseEpisode:
    """统一 Episode 基类"""

    # 通用字段
    episode_id: str
    task_type: str                # "vln" | "objectnav" | "manipulation" | "pick_place"
    scene_id: str

    # 初始状态
    start_state: Dict[str, Any] = field(default_factory=dict)
    """
    导航任务：{"position": [x,y,z], "rotation": [x,y,z,w]}
    操作任务：{"base_pose": [x,y,θ], "joint_positions": [...]}
    """

    # 任务目标
    goals: Dict[str, Any] = field(default_factory=dict)
    """
    导航任务：{"position": [x,y,z], "radius": 0.2}
    操作任务：{
        "target_object": {...},
        "target_location": {...},
        "success_criteria": {...}
    }
    """

    # 任务指令（多模态）
    instruction: Dict[str, Any] = field(default_factory=dict)
    """
    导航任务：{"text": "Walk to the kitchen", "tokens": [...]}
    操作任务：{"text": "Pick up the red cup", "tokens": [...]} 或 {}
    """

    # 参考数据（可选）
    reference_data: Optional[Dict[str, Any]] = None
    """
    导航任务：{"path": [[x,y,z], ...]}
    操作任务：{"trajectory": {"qpos": [...], "ee_pose": [...]}}
    """

    # 任务特定配置
    task_config: Dict[str, Any] = field(default_factory=dict)
    """
    任务特定的扩展配置，如：
    - 机器人配置（操作任务）
    - 场景物体（操作任务）
    - 随机化配置（泛化评测）
    """

    # 仿真参数
    sim_params: Dict[str, Any] = field(default_factory=dict)
    """
    {
        "max_steps": 500,
        "timeout": 300,
        "time_step": 0.01
    }
    """

    def __post_init__(self):
        """初始化后验证"""
        if self.task_type not in self._valid_task_types():
            raise ValueError(f"Unknown task_type: {self.task_type}")

    @staticmethod
    def _valid_task_types() -> List[str]:
        """有效的任务类型"""
        return ["vln", "objectnav", "image_nav", "manipulation", "pick_place", "stacking"]
```

### 3.3 BaseMetric 抽象

```python
# core/metric/base.py
from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseMetric(ABC):
    """指标抽象基类"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self._value = None

    @property
    @abstractmethod
    def metric_name(self) -> str:
        """指标名称"""
        pass

    @abstractmethod
    def reset(self, episode: BaseEpisode, task: BaseTask):
        """重置指标状态"""
        pass

    @abstractmethod
    def update(self, *args, **kwargs):
        """更新指标状态"""
        pass

    @abstractmethod
    def get_metric(self) -> Any:
        """获取当前指标值"""
        pass

    def compute_aggregated(self, values: list) -> Dict[str, float]:
        """计算聚合统计"""
        import numpy as np
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "count": len(values)
        }
```

---

## 四、VLN 任务适配

### 4.1 VLNTask 实现

```python
# tasks/vln/task.py
from core.task.base import BaseTask, ActionSpace, SensorSuite

class VLNTask(BaseTask):
    """VLN 任务（适配新架构）"""

    @property
    def task_type(self) -> str:
        return "vln"

    @property
    def task_name(self) -> str:
        return "Vision-Language Navigation"

    @property
    def action_space(self) -> ActionSpace:
        return ActionSpace(
            name="discrete",
            actions=["stop", "move_forward", "turn_left", "turn_right"],
            semantics={
                "stop": "停止并结束 Episode",
                "move_forward": "向前移动 0.25 米",
                "turn_left": "左转 15 度",
                "turn_right": "右转 15 度"
            }
        )

    @property
    def sensor_suite(self) -> SensorSuite:
        return SensorSuite(sensors={
            "rgb": {"width": 640, "height": 480, "encoding": "base64"},
            "depth": {"width": 640, "height": 480, "encoding": "base64"},
            "instruction": {"type": "text"},
            "gps": {"dim": 3},
            "compass": {"dim": 1}
        })

    def check_episode_complete(self, state: Dict[str, Any]) -> bool:
        """检查 VLN Episode 是否完成"""
        # Success 条件
        distance = state.get("distance_to_goal", float('inf'))
        if distance < self.config.get("success_distance", 0.2):
            return True

        # Timeout 条件
        if state.get("num_steps", 0) >= self.config.get("max_steps", 500):
            return True

        # Stop 动作
        if state.get("last_action") == "stop":
            return True

        return False
```

### 4.2 注册 VLNTask

```python
# registry/tasks.py (新增)
from typing import Type, Dict, List
from core.task.base import BaseTask

_task_registry: Dict[str, Type[BaseTask]] = {}

def register_task(task_type: str, task_name: str = None):
    """任务注册装饰器"""
    def decorator(task_class: Type[BaseTask]):
        _task_registry[task_type] = {
            "class": task_class,
            "name": task_name or task_class.__name__
        }
        return task_class
    return decorator

def get_task(task_type: str) -> Type[BaseTask]:
    """获取任务类"""
    if task_type not in _task_registry:
        raise ValueError(f"Unknown task_type: {task_type}")
    return _task_registry[task_type]["class"]

def list_tasks() -> List[str]:
    """列出所有注册的任务"""
    return list(_task_registry.keys())

# 注册内置任务
from tasks.vln.task import VLNTask
register_task("vln", "Vision-Language Navigation")(VLNTask)
```

---

## 五、操作任务实现

### 5.1 ManipulationTask 基类

```python
# tasks/manipulation/task.py
from core.task.base import BaseTask, ActionSpace, SensorSuite

class ManipulationTask(BaseTask):
    """操作任务基类"""

    @property
    def task_type(self) -> str:
        return "manipulation"

    @property
    def task_name(self) -> str:
        return "Robotic Manipulation"

    @property
    def action_space(self) -> ActionSpace:
        """操作任务使用连续动作空间"""
        return ActionSpace(
            name="continuous",
            shape=(self.config.get("dof", 7),),
            range=self.config.get("joint_limits", None)
        )

    @property
    def sensor_suite(self) -> SensorSuite:
        return SensorSuite(sensors={
            "rgb_head": {"width": 640, "height": 480, "encoding": "base64"},
            "rgb_wrist": {"width": 320, "height": 240, "encoding": "base64"},
            "qpos": {"dim": self.config.get("dof", 7)},
            "qvel": {"dim": self.config.get("dof", 7)},
            "ee_pose": {"dim": 7},  # [x,y,z,qw,qx,qy,qz]
            "gripper_state": {"dim": 1}
        })

    def check_episode_complete(self, state: Dict[str, Any]) -> bool:
        """检查操作任务是否完成"""
        # Success 条件（由具体任务实现）
        if state.get("success", False):
            return True

        # Timeout 条件
        if state.get("num_steps", 0) >= self.config.get("max_steps", 500):
            return True

        # Error 条件（物理错误、碰撞等）
        if state.get("error", False):
            return True

        return False
```

### 5.2 PickAndPlaceTask 实现

```python
# tasks/manipulation/pick_place.py
from .task import ManipulationTask

class PickAndPlaceTask(ManipulationTask):
    """Pick and Place 任务"""

    @property
    def task_type(self) -> str:
        return "pick_place"

    @property
    def task_name(self) -> str:
        return "Pick and Place"

    @property
    def action_space(self) -> ActionSpace:
        """Pick and Place 可以使用混合动作空间"""
        return ActionSpace(
            name="hybrid",  # 支持离散和连续动作
            discrete_actions=["stop"],
            continuous_shape=(self.config.get("dof", 7),)
        )

    def check_episode_complete(self, state: Dict[str, Any]) -> bool:
        """检查 Pick and Place 特定完成条件"""
        # 调用父类检查
        if super().check_episode_complete(state):
            return True

        # Pick and Place 特定成功条件
        goals = self.config.get("goals", {})
        success_criteria = goals.get("success_criteria", {})

        if success_criteria.get("type") == "grasp_and_lift":
            # 检查：抓取 + 提起 + 放置
            object_grasped = state.get("object_grasped", False)
            object_lifted = state.get("object_lifted", False)
            object_placed = state.get("object_placed", False)

            return object_grasped and object_lifted and object_placed

        return False

# 注册任务
from registry.tasks import register_task
register_task("pick_place", "Pick and Place")(PickAndPlaceTask)
```

---

## 六、指标系统重构

### 6.1 导航指标适配

```python
# metrics/navigation/success.py
from core.metric.base import BaseMetric
from core.episode.base import BaseEpisode
from core.task.base import BaseTask

class NavigationSuccess(BaseMetric):
    """导航任务成功率（适配新架构）"""

    @property
    def metric_name(self) -> str:
        return "success"

    def reset(self, episode: BaseEpisode, task: BaseTask):
        self.success_distance = task.config.get("success_distance", 0.2)
        self.success = False

    def update(self, distance_to_goal: float):
        self.success = distance_to_goal < self.success_distance

    def get_metric(self) -> float:
        return 1.0 if self.success else 0.0
```

### 6.2 操作任务指标

```python
# metrics/manipulation/success.py
from core.metric.base import BaseMetric

class ManipulationSuccess(BaseMetric):
    """操作任务成功率"""

    @property
    def metric_name(self) -> str:
        return "success"

    def reset(self, episode: BaseEpisode, task: BaseTask):
        self.success = False
        self.goals = episode.goals

    def update(self, **kwargs):
        """更新成功状态"""
        # 根据任务类型检查不同的成功条件
        success_criteria = self.goals.get("success_criteria", {})

        if success_criteria.get("type") == "grasp_and_lift":
            self.success = (
                kwargs.get("object_grasped", False) and
                kwargs.get("object_lifted", False) and
                kwargs.get("object_placed", False)
            )
        # 其他成功条件...

    def get_metric(self) -> float:
        return 1.0 if self.success else 0.0
```

### 6.3 指标注册系统

```python
# registry/metrics.py (新增)
from typing import Type, Dict, List
from core.metric.base import BaseMetric

_metric_registry: Dict[str, Type[BaseMetric]] = {}

def register_metric(metric_name: str):
    """指标注册装饰器"""
    def decorator(metric_class: Type[BaseMetric]):
        _metric_registry[metric_name] = metric_class
        return metric_class
    return decorator

def get_metric(metric_name: str) -> Type[BaseMetric]:
    """获取指标类"""
    if metric_name not in _metric_registry:
        raise ValueError(f"Unknown metric: {metric_name}")
    return _metric_registry[metric_name]

def list_metrics() -> List[str]:
    """列出所有注册的指标"""
    return list(_metric_registry.keys())

# 注册内置指标
from metrics.navigation.success import NavigationSuccess
from metrics.navigation.spl import SPL
from metrics.navigation.dtw import DTW
from metrics.manipulation.success import ManipulationSuccess
from metrics.manipulation.completion import CompletionRate
from metrics.manipulation.trajectory import TrajectorySimilarity

register_metric("success")(NavigationSuccess)  # 导航
register_metric("spl")(SPL)
register_metric("dtw")(DTW)
register_metric("manip_success")(ManipulationSuccess)  # 操作
register_metric("completion_rate")(CompletionRate)
register_metric("trajectory_similarity")(TrajectorySimilarity)
```

---

## 七、API 协议扩展

### 7.1 统一消息类型

```python
# api/messages/base.py
from typing import Dict, Any, TypedDict, Union
from enum import Enum

class MessageType(str, Enum):
    """统一消息类型"""
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

class BaseMessage(TypedDict):
    """基础消息结构"""
    type: MessageType
    session_id: str

class ResetEpisodeMessage(BaseMessage):
    """Reset Episode 消息（泛化）"""
    type: MessageType
    session_id: str
    episode: Dict[str, Any]  # 包含 task_type, 用于区分任务

class GetActionMessage(BaseMessage):
    """Get Action 消息（泛化）"""
    type: MessageType
    session_id: str
    task_type: str            # 新增：任务类型
    observation: Dict[str, Any]

class ActionMessage(BaseMessage):
    """Action 消息（泛化）"""
    type: MessageType
    session_id: str
    action: Union[str, List[float], Dict]  # 支持不同动作格式
```

### 7.2 消息序列化

```python
# api/protocol.py (扩展)
from typing import Dict, Any
from core.episode.base import BaseEpisode

def create_reset_message(episode: BaseEpisode) -> Dict[str, Any]:
    """
    创建 Reset Episode 消息（统一接口）

    Args:
        episode: 统一的 Episode 对象

    Returns:
        ResetEpisodeMessage
    """
    base_msg = {
        "type": "reset_episode",
        "session_id": episode.session_id,
        "episode": {
            "episode_id": episode.episode_id,
            "task_type": episode.task_type,
            "scene_id": episode.scene_id
        }
    }

    # 根据任务类型添加特定字段
    if episode.task_type in ["vln", "objectnav", "image_nav"]:
        base_msg["episode"].update({
            "instruction": episode.instruction,
            "start_position": episode.start_state.get("position"),
            "start_rotation": episode.start_state.get("rotation"),
            "goals": episode.goals,
            "reference_path": episode.reference_data.get("path") if episode.reference_data else None
        })

    elif episode.task_type in ["manipulation", "pick_place"]:
        base_msg["episode"].update({
            "instruction": episode.instruction,
            "robot_config": episode.task_config.get("robot_config"),
            "scene_objects": episode.task_config.get("scene_objects"),
            "goals": episode.goals,
            "reference_trajectory": episode.reference_data.get("trajectory") if episode.reference_data else None
        })

    return base_msg

def parse_action_message(message: Dict[str, Any], task_type: str) -> Union[str, List[float]]:
    """
    解析 Action 消息（统一接口）

    Args:
        message: 原始消息
        task_type: 任务类型

    Returns:
        解析后的动作
    """
    action = message.get("action")

    if task_type in ["vln", "objectnav", "image_nav"]:
        # 离散动作（字符串）
        if isinstance(action, str):
            return action
        raise ValueError(f"Expected discrete action for {task_type}")

    elif task_type in ["manipulation", "pick_place"]:
        # 连续动作（向量）
        if isinstance(action, list):
            return action
        raise ValueError(f"Expected continuous action for {task_type}")

    return action
```

---

## 八、配置系统扩展

### 8.1 统一配置格式

```yaml
# configs/benchmarks/unified_challenge.yaml
benchmark:
  name: "Unified Robot Challenge 2024"
  description: "Navigation and Manipulation Tasks"

# VLN 任务配置
vln_task:
  type: "vln"
  config:
    action_space: "discrete"
    actions: ["stop", "move_forward", "turn_left", "turn_right"]
    metrics: ["success", "spl", "navigation_error", "dtw"]

# 操作任务配置
manipulation_task:
  type: "pick_place"
  config:
    robot_type: "stretch"
    dof: 7
    action_space: "continuous"
    metrics: ["manip_success", "completion_rate", "trajectory_similarity"]
    goals:
      target_object:
        name: "cup_red"
      target_location:
        position: [0.7, 0.2, 0.8]
      success_criteria:
        type: "grasp_and_lift"
        lift_height: 0.1
        place_tolerance: 0.05

simulator:
  backend: "abstract"
  sensors:
    rgb: {width: 640, height: 480}
    depth: {width: 640, height: 480}

dataset:
  vln:
    type: "vln"
    data_path: "data/datasets/vln/R2R/val_seen.json.gz"
  manipulation:
    type: "manipulation"
    data_path: "data/datasets/manipulation/LIBERO/train"

evaluation:
  max_steps: 500
  timeout: 300

output:
  log_dir: "logs/evaluations"
  save_trajectories: true
```

---

## 九、实现路线图（基于用户确认）

### Phase 1: 核心抽象层（第一优先级）

```
robot_eval/
├── core/                    # 新增核心抽象层
│   ├── task/
│   │   ├── base.py          # BaseTask 抽象类
│   │   ├── types.py         # ActionSpace, SensorSuite
│   │   └── registry.py      # TaskRegistry
│   ├── episode/
│   │   ├── base.py          # BaseEpisode 统一基类
│   │   └── schemas.py       # Episode Schema 定义
│   └── metric/
│       ├── base.py          # BaseMetric 抽象类
│       └── registry.py      # MetricRegistry
```

**实现清单**：
- [x] `core/task/base.py` - BaseTask 抽象类
- [x] `core/task/types.py` - ActionSpace, SensorSuite
- [x] `core/task/registry.py` - TaskRegistry
- [x] `core/episode/base.py` - 统一 Episode 基类
- [x] `core/metric/base.py` - BaseMetric 抽象类
- [x] `core/metric/registry.py` - MetricRegistry

### Phase 2: VLN 任务适配（验证抽象层）

```
robot_eval/
├── tasks/
│   ├── vln/
│   │   ├── task.py          # VLNTask 适配新架构
│   │   ├── actions.py       # VLN 动作定义
│   │   └── config.py        # VLN 配置
```

**实现清单**：
- [x] `tasks/vln/task.py` - VLNTask 继承 BaseTask
- [x] `tasks/vln/actions.py` - VLN 动作空间定义
- [x] `tasks/vln/config.py` - VLN 配置

### Phase 3: 操作任务实现

```
robot_eval/
├── tasks/
│   └── manipulation/        # 新增
│       ├── task.py          # ManipulationTask 基类
│       ├── pick_place.py    # PickAndPlaceTask
│       └── robots/          # 机器人配置
│           └── stretch.py   # Stretch 配置
```

**实现清单**：
- [x] `tasks/manipulation/task.py` - ManipulationTask 基类
- [x] `tasks/manipulation/pick_place.py` - PickAndPlaceTask
- [x] `tasks/manipulation/robots/stretch.py` - Stretch 配置

### Phase 4: 指标系统实现

```
robot_eval/
├── metrics/
│   ├── navigation/           # 重构（适配新架构）
│   │   ├── success.py
│   │   ├── spl.py
│   │   └── dtw.py
│   └── manipulation/         # 新增
│       ├── success.py       # ManipulationSuccess
│       ├── completion.py    # CompletionRate
│       └── trajectory.py    # TrajectorySimilarity
```

**实现清单**：
- [x] `metrics/navigation/success.py` - 导航指标适配
- [x] `metrics/manipulation/success.py` - 操作成功指标
- [x] `metrics/manipulation/completion.py` - CompletionRate
- [x] `metrics/manipulation/trajectory.py` - TrajectorySimilarity

---

## 十、WebSocket 协议（保持不变）

根据用户确认：**WebSocket 接口不变，都是传递 obs，然后获取 action**

### 10.1 协议原则

```python
# 统一的接口设计
class WebSocketProtocol:
    """WebSocket 协议（任务无关）"""

    def send_observation(self, session_id: str, observation: dict):
        """
        发送观测（统一接口）

        Args:
            observation: {
                # VLN 任务
                "rgb": "...",
                "depth": "...",
                "instruction": {...},

                # 操作任务（扩展字段）
                "qpos": [...],
                "ee_pose": [...],
                "gripper_state": 0.04
            }
        """
        pass

    def receive_action(self, session_id: str) -> Union[str, list]:
        """
        接收动作（统一接口）

        Returns:
            action:
                # VLN 任务：离散动作
                "move_forward"

                # 操作任务：连续动作
                [0.05, 0.0, 0.1, ...]
        """
        pass
```

### 10.2 消息格式（自动适配）

```json
// get_action 消息（任务无关）
{
  "type": "get_action",
  "session_id": "uuid-xxx",
  "task_type": "vln",  // 或 "pick_place"
  "observation": {
    // VLN 专用
    "rgb": "<base64>",
    "instruction": {...},

    // 操作任务专用
    "qpos": [0.0, 0.0, ...],
    "ee_pose": [0.5, 0.0, 0.8, ...]
  }
}

// action 消息（任务无关）
{
  "type": "action",
  "session_id": "uuid-xxx",
  "action": {
    // VLN：离散动作
    "name": "move_forward",

    // 操作：连续动作
    "qpos": [0.05, 0.0, ...]
  }
}
```

---

## 十一、核心代码实现

### 11.1 BaseTask 抽象

```python
# core/task/base.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class ActionSpace:
    """动作空间定义"""
    name: str                          # "discrete" | "continuous" | "hybrid"
    actions: Optional[List[str]] = None    # 离散动作列表
    shape: Optional[tuple] = None         # 连续动作形状
    range: Optional[tuple] = None         # 连续动作范围

@dataclass
class SensorSuite:
    """传感器套件定义"""
    sensors: Dict[str, Dict[str, Any]]

    def get(self, name: str) -> Dict[str, Any]:
        return self.sensors.get(name, {})

class BaseTask(ABC):
    """任务抽象基类"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @property
    @abstractmethod
    def task_type(self) -> str:
        """任务类型标识"""
        pass

    @property
    @abstractmethod
    def task_name(self) -> str:
        """任务名称（人类可读）"""
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
    def metric_names(self) -> List[str]:
        """指标名称列表"""
        return self.config.get("metrics", [])

    @abstractmethod
    def check_episode_complete(self, state: Dict[str, Any]) -> bool:
        """检查 Episode 是否完成"""
        pass

    def get_metrics(self):
        """获取任务对应的指标类"""
        from core.metric.registry import get_metric
        return [get_metric(name) for name in self.metric_names]
```

### 11.2 BaseEpisode 统一

```python
# core/episode/base.py
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

@dataclass
class BaseEpisode:
    """统一 Episode 基类"""

    # 通用字段
    episode_id: str
    task_type: str                # "vln" | "manipulation" | "pick_place"
    scene_id: str

    # 初始状态（通用）
    start_state: Dict[str, Any] = field(default_factory=dict)

    # 任务目标
    goals: Dict[str, Any] = field(default_factory=dict)

    # 任务指令（多模态）
    instruction: Dict[str, Any] = field(default_factory=dict)

    # 参考数据（可选）
    reference_data: Optional[Dict[str, Any]] = None

    # 任务特定配置
    task_config: Dict[str, Any] = field(default_factory=dict)

    # 仿真参数
    sim_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """初始化后验证"""
        valid_types = ["vln", "manipulation", "pick_place",
                      "objectnav", "image_nav"]
        if self.task_type not in valid_types:
            raise ValueError(f"Unknown task_type: {self.task_type}")

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（用于 WebSocket 消息）"""
        return {
            "episode_id": self.episode_id,
            "task_type": self.task_type,
            "scene_id": self.scene_id,
            "instruction": self.instruction,
            **self.start_state,
            **self.goals
        }
```

### 11.3 VLNTask 适配

```python
# tasks/vln/task.py
from core.task.base import BaseTask, ActionSpace, SensorSuite

class VLNTask(BaseTask):
    """VLN 任务（适配新架构）"""

    @property
    def task_type(self) -> str:
        return "vln"

    @property
    def task_name(self) -> str:
        return "Vision-Language Navigation"

    @property
    def action_space(self) -> ActionSpace:
        return ActionSpace(
            name="discrete",
            actions=["stop", "move_forward", "turn_left", "turn_right"]
        )

    @property
    def sensor_suite(self) -> SensorSuite:
        return SensorSuite(sensors={
            "rgb": {"width": 640, "height": 480},
            "depth": {"width": 640, "height": 480},
            "instruction": {"type": "text"},
            "gps": {"dim": 3},
            "compass": {"dim": 1}
        })

    def check_episode_complete(self, state: Dict[str, Any]) -> bool:
        distance = state.get("distance_to_goal", float('inf'))
        return distance < self.config.get("success_distance", 0.2)

# 注册任务
from core.task.registry import register_task
register_task("vln")(VLNTask)
```

### 11.4 ManipulationTask 基类

```python
# tasks/manipulation/task.py
from core.task.base import BaseTask, ActionSpace, SensorSuite

class ManipulationTask(BaseTask):
    """操作任务基类"""

    @property
    def task_type(self) -> str:
        return "manipulation"

    @property
    def task_name(self) -> str:
        return "Robotic Manipulation"

    @property
    def action_space(self) -> ActionSpace:
        dof = self.config.get("dof", 7)
        return ActionSpace(
            name="continuous",
            shape=(dof,),
            range=self.config.get("joint_limits")
        )

    @property
    def sensor_suite(self) -> SensorSuite:
        dof = self.config.get("dof", 7)
        return SensorSuite(sensors={
            "rgb_head": {"width": 640, "height": 480},
            "rgb_wrist": {"width": 320, "height": 240},
            "qpos": {"dim": dof},
            "qvel": {"dim": dof},
            "ee_pose": {"dim": 7},
            "gripper_state": {"dim": 1}
        })

    def check_episode_complete(self, state: Dict[str, Any]) -> bool:
        # Success 条件
        if state.get("success", False):
            return True
        # Timeout 条件
        if state.get("num_steps", 0) >= self.config.get("max_steps", 500):
            return True
        return False

# 注册任务
from core.task.registry import register_task
register_task("manipulation")(ManipulationTask)
```

### 11.5 PickAndPlaceTask 实现

```python
# tasks/manipulation/pick_place.py
from .task import ManipulationTask

class PickAndPlaceTask(ManipulationTask):
    """Pick and Place 任务"""

    @property
    def task_type(self) -> str:
        return "pick_place"

    @property
    def task_name(self) -> str:
        return "Pick and Place"

    def check_episode_complete(self, state: Dict[str, Any]) -> bool:
        # 调用父类通用检查
        if super().check_episode_complete(state):
            return True

        # Pick and Place 特定成功条件
        goals = self.config.get("goals", {})
        success_criteria = goals.get("success_criteria", {})

        if success_criteria.get("type") == "grasp_and_lift":
            return (
                state.get("object_grasped", False) and
                state.get("object_lifted", False) and
                state.get("object_placed", False)
            )

        return False

# 注册任务
from core.task.registry import register_task
register_task("pick_place")(PickAndPlaceTask)
```

### 11.6 机器人配置

```python
# tasks/manipulation/robots/stretch.py
from dataclasses import dataclass

@dataclass
class StretchRobotConfig:
    """Stretch 机器人配置"""

    robot_type: str = "stretch"

    # 自由度
    dof: int = 7

    # 关节名称
    joint_names: list = None

    def __post_init__(self):
        self.joint_names = [
            "translate_x", "translate_y", "rotate_z",
            "joint_lift", "joint_arm_l0", "joint_arm_l1",
            "joint_arm_l2", "joint_arm_l3", "joint_wrist_yaw",
            "joint_gripper_finger_left"
        ]

    # 动作范围
    joint_limits: dict = None

    def __post_init__(self):
        if self.joint_limits is None:
            self.joint_limits = {
                "translate_x": (-0.5, 0.5),
                "translate_y": (-0.5, 0.5),
                "rotate_z": (-3.14, 3.14),
                "joint_lift": (0.0, 1.1),
                "joint_arm_l0": (-1.8, 1.8),
                "joint_arm_l1": (-1.8, 1.8),
                "joint_arm_l2": (-1.8, 1.8),
                "joint_arm_l3": (-1.8, 1.8),
                "joint_wrist_yaw": (-1.8, 1.8),
                "joint_gripper_finger_left": (0.0, 0.04)
            }

# 预定义配置
STRETCH_RE1_CONFIG = StretchRobotConfig(
    dof=7,
    joint_limits={
        "translate_x": (-0.3, 0.3),  # 更保守的范围
        "translate_y": (-0.3, 0.3),
        # ...
    }
)
```

---

## 十二、注册系统实现

### 12.1 TaskRegistry

```python
# core/task/registry.py
from typing import Type, Dict, List
from core.task.base import BaseTask

_task_registry: Dict[str, Type[BaseTask]] = {}

def register_task(task_type: str, task_name: str = None):
    """任务注册装饰器"""
    def decorator(task_class: Type[BaseTask]):
        _task_registry[task_type] = {
            "class": task_class,
            "name": task_name or task_class.__name__
        }
        return task_class
    return decorator

def get_task(task_type: str, config: dict) -> BaseTask:
    """获取任务实例"""
    if task_type not in _task_registry:
        raise ValueError(f"Unknown task: {task_type}")
    task_class = _task_registry[task_type]["class"]
    return task_class(config)

def list_tasks() -> List[str]:
    """列出所有注册的任务"""
    return list(_task_registry.keys())
```

### 12.2 MetricRegistry

```python
# core/metric/registry.py
from typing import Type, Dict, List
from core.metric.base import BaseMetric

_metric_registry: Dict[str, Type[BaseMetric]] = {}

def register_metric(metric_name: str):
    """指标注册装饰器"""
    def decorator(metric_class: Type[BaseMetric]):
        _metric_registry[metric_name] = metric_class
        return metric_class
    return decorator

def get_metric(metric_name: str, config: dict = None) -> BaseMetric:
    """获取指标实例"""
    if metric_name not in _metric_registry:
        raise ValueError(f"Unknown metric: {metric_name}")
    metric_class = _metric_registry[metric_name]
    return metric_class(config or {})

def list_metrics() -> List[str]:
    """列出所有注册的指标"""
    return list(_metric_registry.keys())
```

---

## 十三、使用示例

### 13.1 加载配置

```python
# example_load_task.py
from core.task.registry import get_task

# VLN 任务
vln_config = {
    "success_distance": 0.2,
    "max_steps": 500
}
vln_task = get_task("vln", vln_config)
print(f"Task: {vln_task.task_name}")
print(f"Actions: {vln_task.action_space.actions}")

# 操作任务
manip_config = {
    "dof": 7,
    "robot_type": "stretch",
    "max_steps": 500
}
manip_task = get_task("manipulation", manip_config)
print(f"Task: {manip_task.task_name}")
print(f"Action Shape: {manip_task.action_space.shape}")
```

### 13.2 创建 Episode

```python
# example_create_episode.py
from core.episode.base import BaseEpisode

# VLN Episode
vln_episode = BaseEpisode(
    episode_id="vln_001",
    task_type="vln",
    scene_id="scene_001",
    start_state={
        "position": [1.5, 0.0, 2.3],
        "rotation": [0.0, 0.0, 0.0, 1.0]
    },
    instruction={
        "text": "Walk to the kitchen"
    },
    goals={
        "position": [3.0, 1.0, 2.3],
        "radius": 0.2
    }
)

# 操作 Episode
manip_episode = BaseEpisode(
    episode_id="manip_001",
    task_type="pick_place",
    scene_id="table_setup_001",
    start_state={
        "base_pose": [0.0, 0.0, 0.0],
        "joint_positions": [0.0] * 7
    },
    instruction={
        "text": "Pick up the red cup"
    },
    goals={
        "target_object": {"name": "cup_red"},
        "target_location": {"position": [0.7, 0.2, 0.8]},
        "success_criteria": {
            "type": "grasp_and_lift",
            "lift_height": 0.1
        }
    },
    task_config={
        "robot_config": {
            "robot_type": "stretch",
            "dof": 7
        }
    }
)
```

---

## 十四、测试验证

### 14.1 单元测试

```python
# tests/core/test_task_registry.py
def test_task_registration():
    """测试任务注册"""
    from core.task.registry import register_task, get_task, list_tasks

    @register_task("test_task")
    class TestTask(BaseTask):
        @property
        def task_type(self):
            return "test_task"

    # 验证注册
    assert "test_task" in list_tasks()

    # 验证获取
    task = get_task("test_task", {})
    assert isinstance(task, TestTask)

def test_vln_task_compatibility():
    """测试 VLN 任务兼容性"""
    from tasks.vln.task import VLNTask

    task = VLNTask(config={})
    assert task.task_type == "vln"
    assert "move_forward" in task.action_space.actions
```

### 14.2 集成测试

```python
# tests/integration/test_unified_evaluator.py
def test_vln_evaluation():
    """测试 VLN 评测（新架构）"""
    from core.evaluator import UnifiedEvaluator

    config = load_config("configs/benchmarks/vln_challenge.yaml")
    evaluator = UnifiedEvaluator(config)
    results = evaluator.run()

    assert "success" in results.aggregated_metrics

def test_manipulation_evaluation():
    """测试操作任务评测（新架构）"""
    from core.evaluator import UnifiedEvaluator

    config = load_config("configs/benchmarks/manipulation_challenge.yaml")
    evaluator = UnifiedEvaluator(config)
    results = evaluator.run()

    assert "manip_success" in results.aggregated_metrics
```

---

## 十五、迁移检查清单

### 15.1 现有代码迁移

| 原路径 | 新路径 | 更改类型 |
|--------|--------|----------|
| `nav_eval/tasks/vln.py` | `robot_eval/tasks/vln/task.py` | 重构适配 |
| `nav_eval/metrics/success.py` | `robot_eval/metrics/navigation/success.py` | 重构适配 |
| `nav_eval/dataset/episode.py` | `robot_eval/core/episode/base.py` | 抽象化 |

### 15.2 向后兼容性

```python
# robot_eval/compat.py (兼容层)
from robot_eval.tasks.vln.task import VLNTask as NewVLNTask

class VLNEvaluator:
    """兼容层：旧 API → 新架构"""
    def __init__(self, config):
        self._new_task = NewVLNTask(config)

    def evaluate(self, episodes):
        # 兼容性适配
        pass
```

---

## 十六、下一步行动

### 立即行动

1. **创建核心抽象层**
   - 实现 `core/task/` 模块
   - 实现 `core/episode/` 模块
   - 实现 `core/metric/` 模块

2. **适配 VLN 任务**
   - 重构 `VLNTask` 继承 `BaseTask`
   - 验证功能不变

3. **实现操作任务**
   - 实现 `ManipulationTask`
   - 实现 `PickAndPlaceTask`
   - 添加 `Stretch` 配置

4. **更新文档**
   - 更新 API 文档
   - 添加使用示例

### 后续优化

1. **性能优化**
   - 并行评测支持
   - 内存优化

2. **功能扩展**
   - 更多机器人类型
   - 更多操作任务

3. **可视化**
   - 轨迹可视化工具
   - 结果对比工具

---

## 十、迁移路径

### 10.1 现有代码迁移

| 原路径 | 新路径 | 变更类型 |
|--------|--------|----------|
| `nav_eval/tasks/vln.py` | `robot_eval/tasks/vln/task.py` | 重构适配 |
| `nav_eval/metrics/success.py` | `robot_eval/metrics/navigation/success.py` | 重构适配 |
| `nav_eval/dataset/episode.py` | `robot_eval/core/episode/base.py` | 抽象化 |

### 10.2 兼容性策略

```python
# 为了兼容现有代码，提供过渡层
# nav_eval/compat.py (临时)
from robot_eval.tasks.vln.task import VLNTask as NewVLNTask
from robot_eval.metrics.navigation.success import NavigationSuccess as NewSuccess

# 保留旧接口
class VLNEvaluator:
    """兼容层：旧 API → 新架构"""
    def __init__(self, config):
        self._new_task = NewVLNTask(config)
        # ... 兼容性适配
```

---

## 十一、验证测试

### 11.1 单元测试

```python
# tests/core/test_task_registry.py
def test_task_registration():
    """测试任务注册"""
    from registry.tasks import register_task, get_task

    @register_task("test_task")
    class TestTask(BaseTask):
        pass

    task_cls = get_task("test_task")
    assert task_cls == TestTask

def test_vln_task_compatibility():
    """测试 VLN 任务兼容性"""
    from robot_eval.tasks.vln.task import VLNTask

    task = VLNTask(config={})
    assert task.task_type == "vln"
    assert "move_forward" in task.action_space.actions

def test_manipulation_task_creation():
    """测试操作任务创建"""
    from robot_eval.tasks.manipulation.pick_place import PickAndPlaceTask

    task = PickAndPlaceTask(config={"dof": 7})
    assert task.task_type == "pick_place"
    assert task.action_space.name == "hybrid"
```

### 11.2 集成测试

```python
# tests/integration/test_unified_evaluator.py
def test_vln_evaluation():
    """测试 VLN 评测（新架构）"""
    from robot_eval.core.evaluator import UnifiedRobotEvaluator

    config = load_config("configs/benchmarks/vln_challenge.yaml")
    evaluator = UnifiedRobotEvaluator(config)

    results = evaluator.run()
    assert "success" in results.aggregated_metrics

def test_manipulation_evaluation():
    """测试操作任务评测（新架构）"""
    from robot_eval.core.evaluator import UnifiedRobotEvaluator

    config = load_config("configs/benchmarks/manipulation_challenge.yaml")
    evaluator = UnifiedRobotEvaluator(config)

    results = evaluator.run()
    assert "manip_success" in results.aggregated_metrics
```

---

## 十二、与 RoboVerse 的对比

| 方面 | RoboVerse | 本设计 |
|------|------------|--------|
| **任务抽象** | 任务定义在配置中 | BaseTask 抽象类 + 注册表 |
| **Episode 格式** | HDF5 存储 | 统一 BaseEpisode + JSON |
| **随机化** | L0-L3 分层随机化 | 第一阶段暂不支持 |
| **指标系统** | 模块化指标 | MetricRegistry + BaseMetric |
| **机器人支持** | 多种机器人配置 | 抽象接口 + 机器人配置 |

---

## 十三、总结

### 13.1 重构收益

| 收益 | 说明 |
|------|------|
| **统一平台** | 导航和操作任务共享基础设施 |
| **可扩展性** | 插件化添加新任务和指标 |
| **类型安全** | TypedDict + 抽象基类 |
| **向后兼容** | VLN 评测功能保持可用 |

### 13.2 下一步行动

1. 创建 `core/task/` 和 `core/episode/` 抽象层
2. 重构 `VLNTask` 适配新抽象
3. 实现 `ManipulationTask` 和 `PickAndPlaceTask`
4. 实现操作任务指标系统
5. 扩展 WebSocket 协议支持操作任务

---

*文档版本: 1.0*
*创建时间: 2026-02-12*
*基于: VLN 评测架构 + RoboVerse 设计*

# 云机器人评测平台 - 整体架构设计

> 基于 VLN 评测架构 + RoboVerse 设计理念 + 操作任务扩展方案

---

## 目录

1. [设计概述](#一设计概述)
2. [设计原则](#二设计原则)
3. [核心概念定义](#三核心概念定义)
4. [系统架构](#四系统架构)
5. [核心组件设计](#五核心组件设计)
6. [数据流设计](#六数据流设计)
7. [API 协议设计](#七api-协议设计)
8. [扩展机制](#八扩展机制)
9. [配置系统](#九配置系统)
10. [实现路线图](#十实现路线图)

---

## 一、设计概述

### 1.1 项目定位

**云机器人评测平台** (Cloud Robotics Evaluation Platform) - 一个统一、可扩展的机器人任务评测系统，支持：

| 任务类型 | 说明 | 示例 |
|---------|------|------|
| **导航任务** | 视觉语言导航、物体导航、图像导航 | VLN, ObjectNav, ImageNav |
| **操作任务** | 抓取放置、堆叠、开关操作 | Pick & Place, Stacking, Opening |
| **混合任务** | 导航 + 操作的组合任务 | Go to kitchen + Pick cup |

### 1.2 设计目标

```
┌─────────────────────────────────────────────────────────────┐
│                    云机器人评测平台                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────────┐      ┌────────────────┐      ┌────────────┐  │
│  │   导航任务      │      │   操作任务      │      │  未来扩展    │  │
│  │  (VLN, ObjNav) │      │  (Pick&Place)  │      │  (社交、多智能) │  │
│  └────────────────┘      └────────────────┘      └────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐      │
│  │           统一评测架构                        │      │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐ │      │
│  │  │ Task 抽象 │  │ Metric    │  │ Episode   │ │      │
│  │  │           │  │ System    │  │ Data      │ │      │
│  │  └──────────┘  └──────────┘  └──────────┘ │      │
│  └──────────────────────────────────────────────────────┘      │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐      │
│  │           WebSocket 协议 (任务无关)              │      │
│  │  ┌─────────────────────────────────────────┐    │      │
│  │  │  send_observation(obs)               │    │      │
│  │  │  receive_action() → action            │    │      │
│  │  └─────────────────────────────────────────┘    │      │
│  └──────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

### 1.3 关键特性

| 特性 | 说明 |
|------|------|
| **任务无关** | WebSocket 协议统一，obs → action 流程不变 |
| **插件化** | 任务、指标、仿真器均可注册扩展 |
| **类型安全** | TypedDict 定义清晰的数据结构 |
| **配置驱动** | YAML 配置文件定义评测流程 |
| **服务化** | 参赛者推理服务独立部署 |

---

## 二、设计原则

### 2.1 核心设计原则

| 原则 | 说明 | RoboVerse 对应 |
|------|------|---------------|
| **统一抽象** | Task、Episode、Metric 的通用抽象 | ScenarioCfg, BaseTask |
| **类型安全** | 使用 TypedDict 定义数据结构 | HDF5 Schema |
| **插件化** | 通过 Registry 注册新任务和指标 | TaskRegistry, MetricRegistry |
| **协议兼容** | WebSocket 协议向后兼容并扩展 | 统一消息格式 |
| **模块解耦** | 导航和操作任务实现分离 | 任务类型独立 |

### 2.2 与 RoboVerse 的对应

| RoboVerse 设计 | 本设计 |
|---------------|--------|
| ScenarioCfg 统一场景配置 | BenchmarkConfig 统一评测配置 |
| TaskRegistry 任务注册表 | TaskRegistry + MetricRegistry |
| RandomizationCfg 随机化配置 | 第一阶段暂不支持 |
| HDF5 Episode 存储 | 仅定义 Episode，不存储 action/obs |
| L0-L3 泛化评测 | 同分布评测（第一阶段） |

---

## 三、核心概念定义

### 3.1 统一概念体系

```
┌─────────────────────────────────────────────────────────────────┐
│                      核心概念层次结构                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  配置层 (Configuration Layer)                                      │
│  ├── Benchmark (基准): Episode 集合 + 评测规则             │
│  ├── Task (任务): 动作空间 + 观测空间 + 成功条件          │
│  └── Simulator (仿真器): 后端 + 传感器配置                  │
│                                                                 │
│  数据层 (Data Layer)                                              │
│  ├── Scene Dataset: 3D 场景模型集合                            │
│  ├── Task Dataset: Episode 结构化集合                             │
│  └── Trajectory Dataset: Agent 执行轨迹存储（评测产出）            │
│                                                                 │
│  运行时层 (Runtime Layer)                                         │
│  ├── Episode: 单次评测实例（初始状态 + 目标）                   │
│  ├── Simulator: 场景渲染 + 物理计算 + 传感器模拟                │
│  ├── Agent: 决策者（参赛者服务）                                 │
│  └── Metrics: 评价者（Success, SPL, Completion, etc.）            │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 概念定义表

| 概念 | 定义 | 导航任务示例 | 操作任务示例 |
|------|------|-------------|-------------|
| **Task** | 任务抽象定义，描述要解决的问题 | VLNTask | ManipulationTask |
| **Benchmark** | 用于评测的一组 Episode 集合和评测规则 | "VLN Challenge 2024" | "Pick & Place Challenge" |
| **Episode** | 单次评测的完整场景定义 | 起点 + 指令 + 目标位置 | 机器人初始姿态 + 目标物体 |
| **Simulator** | 模拟物理环境和传感器观测的引擎 | NavSimulator (离散动作) | ManipSimulator (连续控制) |
| **Agent** | 执行导航/操作决策的实体 | VLN Agent | Pick & Place Agent |
| **Metric** | 量化 Agent 性能的度量标准 | Success, SPL, DTW | Success, Completion Rate |
| **Observation** | 环境对 Agent 的反馈信息 | RGB, Depth, GPS, Compass | RGB, Depth, qpos, qvel |
| **Action** | Agent 对环境的控制指令 | move_forward, turn_left | [0.05, 0.0, ...] (关节位置) |
| **Trajectory** | Agent 在 Episode 中的位置/状态序列 | [(x,y,z), ...] | [(qpos, ee_pose), ...] |

---

## 四、系统架构

### 4.1 高层架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                    评测编排层 (Orchestration Layer)            │
├─────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Benchmark      │  │   Episode      │  │    Metrics    │      │
│  │   Runner      │  │   Manager      │  │  Aggregator   │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────┬───────────────────────────────────────┘
                          │
┌───────────────────────────▼───────────────────────────────────────┐
│                    仿真引擎层 (Simulation Engine Layer)          │
├─────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │              统一仿真器 (Unified Simulator)              │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────┐ │  │
│  │  │  Scene        │  │   Agent        │  │  Sensor    │ │  │
│  │  │  Manager       │  │   Manager       │  │  Suite     │ │  │
│  │  └──────────────┘  └──────────────┘  └──────────┘ │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────┬───────────────────────────────────────┘
                          │
┌───────────────────────────▼───────────────────────────────────────┐
│                    API 接口层 (API Interface Layer)             │
├─────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │              WebSocket Client (任务无关协议)                │  │
│  │  ┌─────────────────────────────────────────────────────┐    │  │
│  │  │  send_observation(obs)  →  统一接口          │    │  │
│  │  │  receive_action()      →  统一接口          │    │  │
│  │  └─────────────────────────────────────────────────────┘    │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────┬───────────────────────────────────────┘
                          │
                    ┌───────▼────────────┐
                    │  参赛者推理服务    │
                    │  (Agent Service)  │
                    └────────────────────┘
```

### 4.2 分层架构说明

| 层级 | 职责 | 关键组件 |
|------|------|----------|
| **评测编排层** | 管理评测流程、聚合指标、生成报告 | BenchmarkRunner, EpisodeManager, MetricsAggregator |
| **仿真引擎层** | 场景渲染、物理仿真、传感器数据生成 | Scene, Agent, Sensors |
| **API 接口层** | 与参赛者服务通信，协议转换 | WebSocket Client (任务无关) |
| **推理服务层** | 参赛者部署的推理服务（外部） | VLN Agent, Manipulation Agent |

---

## 五、核心组件设计

### 5.1 统一任务抽象

```python
# core/task/base.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class ActionSpace:
    """动作空间定义（任务无关）"""
    name: str                          # "discrete" | "continuous" | "hybrid"
    actions: Optional[List[str]] = None    # 离散动作列表
    shape: Optional[tuple] = None         # 连续动作形状
    range: Optional[tuple] = None         # 连续动作范围

@dataclass
class SensorSuite:
    """传感器套件定义（任务无关）"""
    sensors: Dict[str, Dict[str, Any]]

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
```

### 5.2 导航任务实现

```python
# tasks/navigation/vln_task.py
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
register_task("vln", "Vision-Language Navigation")(VLNTask)
```

### 5.3 操作任务实现

```python
# tasks/manipulation/manipulation_task.py
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
        if state.get("success", False):
            return True
        if state.get("num_steps", 0) >= self.config.get("max_steps", 500):
            return True
        return False

# 注册任务
from core.task.registry import register_task
register_task("manipulation", "Robotic Manipulation")(ManipulationTask)
```

### 5.4 Pick & Place 任务

```python
# tasks/manipulation/pick_place_task.py
from .manipulation_task import ManipulationTask

class PickAndPlaceTask(ManipulationTask):
    """Pick and Place 任务"""

    @property
    def task_type(self) -> str:
        return "pick_place"

    @property
    def task_name(self) -> str:
        return "Pick and Place"

    def check_episode_complete(self, state: Dict[str, Any]) -> bool:
        # 复用父类通用检查
        if super().check_episode_complete(state):
            return True

        # Pick & Place 特定成功条件
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
register_task("pick_place", "Pick and Place")(PickAndPlaceTask)
```

### 5.5 统一 Episode 数据结构

```python
# core/episode/base.py
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

@dataclass
class BaseEpisode:
    """统一 Episode 基类（任务无关）"""

    # 通用字段
    episode_id: str
    task_type: str                # "vln" | "manipulation" | "pick_place"
    scene_id: str

    # 初始状态（通用）
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
    导航任务：{"text": "Walk to kitchen", "tokens": [...]}
    操作任务：{"text": "Pick up red cup", "tokens": [...]} 或 {}
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
    机器人配置（操作任务）
    场景物体（操作任务）
    随机化配置（泛化评测，暂不实现）
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
        valid_types = ["vln", "objectnav", "manipulation", "pick_place"]
        if self.task_type not in valid_types:
            raise ValueError(f"Unknown task_type: {self.task_type}")
```

---

## 六、数据流设计

### 6.1 统一评测流程

```
┌──────────────┐
│  Dataset      │ 加载 Episode
│  (.json.gz)  │────────┐
└──────────────┘         │
                        ▼
┌─────────────────────────────────────────────────────────┐
│           FOR EACH episode:                      │
│                                                 │
│  1. simulator.reset(episode)                  │
│     └─> 加载场景，设置 Agent 状态            │
│     └─> 返回初始观测                           │
│                                                 │
│  2. send_observation(obs)                  │
│     └─> WebSocket: send get_action           │
│                                                 │
│  3. receive_action()                       │
│     ◀─< WebSocket: wait for agent             │
│     └─> Agent 决策                              │
│                                                 │
│  4. simulator.step(action)                   │
│     └─> 更新环境状态                           │
│     └─> 返回下一步观测                         │
│                                                 │
│  5. record_trajectory()                  │
│     └─> 记录状态序列                           │
│                                                 │
│  6. compute_metrics()                    │
│     └─> 计算 Success, SPL, etc.             │
│                                                 │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼
              ┌──────────────────┐
              │  聚合所有指标     │
              │  生成结果报告     │
              └──────────────────┘
```

### 6.2 WebSocket 通信流程（任务无关）

```
┌─────────────┐                    ┌─────────────┐                    ┌─────────────┐
│  Simulator   │                    │  Evaluator   │                    │  Agent Service │
└──────┬──────┘                    └──────┬──────┘                    └──────┬──────┘
       │                                  │                                  │
       │  simulator.reset(episode)        │                                  │
       │  <─────────────────────────────> │                                  │
       │                                  │  WS: send get_action          │
       │                                  │  ──────────────────────────────────>│
       │                                  │                          (obs, task_type)  │
       │                                  │                                  │
       │                                  │  ───────────────────────────────────>│
       │  │  WS: wait for action           │                                  │
       │  <───────────────────────────────┘                                  │
       │                                  │                          (action)                   │
       │                                  │                                  │
       │  simulator.step(action)          │                                  │
       │  <──────────────────────────────> │                                  │
       │                                  │                                  │
       │  [Repeat until done]            │                                  │
       │                                  │                                  │
       │                                  │  WS: send episode_end          │                                  │
       │  <──────────────────────────────> │                                  │
       │                          (status, metrics)         │
       │                                  │  ──────────────────────────────────>│
       │                                  │  WS: close                  │                                  │
       │                                  │                                  │
```

---

## 七、API 协议设计

### 7.1 统一消息类型

```python
# api/protocol/message_types.py
from enum import Enum

class MessageType(str, Enum):
    """统一消息类型（任务无关）"""
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
```

### 7.2 Reset Episode 消息（任务无关）

```json
// Evaluator → Agent
{
  "type": "reset_episode",
  "session_id": "uuid-xxx",
  "episode": {
    "episode_id": "ep_001",
    "task_type": "vln",              // 或 "pick_place"
    "scene_id": "scene_001",

    // 通用字段
    "instruction": {
      "text": "Walk to kitchen"
    },
    "start_state": {...},    // 任务特定初始状态
    "goals": {...},           // 任务特定目标
    "reference_data": {...}  // 可选参考数据
  }
}
```

### 7.3 Get Action 消息（任务无关）

```json
// Evaluator → Agent
{
  "type": "get_action",
  "session_id": "uuid-xxx",
  "task_type": "vln",              // 或 "pick_place"
  "observation": {
    // VLN 任务观测
    "rgb": "<base64>",
    "depth": "<base64>",
    "instruction": {...},
    "gps": [0, 0, 0],
    "compass": 0.0,

    // 操作任务观测（扩展字段）
    "qpos": [0.0, 0.0, ...],
    "qvel": [0.0, 0.0, ...],
    "ee_pose": [0.5, 0.0, 0.8, 1.0, 0, 0, 0],
    "gripper_state": 0.04
  }
}
```

### 7.4 Action 消息（任务无关）

```json
// Agent → Evaluator
{
  "type": "action",
  "session_id": "uuid-xxx",
  "action": {
    // VLN 任务：离散动作
    "name": "move_forward",

    // 操作任务：连续动作
    "qpos": [0.05, 0.0, 0.1, ...]
  }
}
```

---

## 八、扩展机制

### 8.1 任务注册表

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

### 8.2 指标注册表

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

## 九、配置系统

### 9.1 统一配置格式

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
    max_steps: 500
    success_distance: 0.2

# 操作任务配置
manipulation_task:
  type: "pick_place"
  config:
    robot_type: "stretch"
    dof: 7
    action_space: "continuous"
    metrics: ["success", "completion_rate", "trajectory_similarity"]
    max_steps: 500
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

agent_service:
  type: "remote"
  protocol: "websocket"
  endpoint: "localhost:8080"
  timeout: 30

output:
  log_dir: "logs/evaluations"
  save_trajectories: true
```

---

## 十、实现路线图

### Phase 1: 核心抽象层（第一优先级）

- [ ] `core/task/base.py` - BaseTask 抽象类
- [ ] `core/task/types.py` - ActionSpace, SensorSuite
- [ ] `core/episode/base.py` - BaseEpisode 统一基类
- [ ] `core/metric/base.py` - BaseMetric 抽象类
- [ ] `core/task/registry.py` - TaskRegistry
- [ ] `core/metric/registry.py` - MetricRegistry

### Phase 2: VLN 任务适配（验证抽象层）

- [ ] `tasks/vln/task.py` - VLNTask 继承 BaseTask
- [ ] `tasks/vln/actions.py` - VLN 动作空间定义
- [ ] `tasks/vln/config.py` - VLN 配置

### Phase 3: 操作任务实现

- [ ] `tasks/manipulation/task.py` - ManipulationTask 基类
- [ ] `tasks/manipulation/pick_place.py` - PickAndPlaceTask
- [ ] `tasks/manipulation/robots/stretch.py` - Stretch 配置

### Phase 4: 指标系统实现

- [ ] `metrics/navigation/success.py` - 导航指标适配
- [ ] `metrics/manipulation/success.py` - 操作成功指标
- [ ] `metrics/manipulation/completion.py` - CompletionRate
- [ ] `metrics/manipulation/trajectory.py` - TrajectorySimilarity

### Phase 5: API 协议实现

- [ ] `api/protocol/message_types.py` - 统一消息类型
- [ ] `api/websocket/server.py` - WebSocket 服务器（任务无关）

### Phase 6: 测试和文档

- [ ] 单元测试
- [ ] 集成测试
- [ ] 使用文档
- [ ] API 文档

---

## 附录：目录结构

```
robot_eval/
├── core/                      # 核心抽象层（新增）
│   ├── task/
│   │   ├── base.py         # BaseTask 抽象类
│   │   ├── types.py        # ActionSpace, SensorSuite
│   │   └── registry.py     # TaskRegistry
│   ├── episode/
│   │   └── base.py         # BaseEpisode 统一基类
│   └── metric/
│       ├── base.py         # BaseMetric 抽象类
│       └── registry.py     # MetricRegistry
│
├── tasks/                     # 任务实现（重构）
│   ├── vln/
│   │   ├── task.py         # VLNTask（适配新架构）
│   │   ├── actions.py      # VLN 动作定义
│   │   └── config.py       # VLN 配置
│   └── manipulation/          # 新增
│       ├── task.py         # ManipulationTask 基类
│       ├── pick_place.py    # PickAndPlaceTask
│       └── robots/
│           └── stretch.py   # Stretch 配置
│
├── metrics/                    # 指标实现（重构）
│   ├── navigation/            # 导航指标
│   │   ├── success.py
│   │   ├── spl.py
│   │   └── dtw.py
│   └── manipulation/          # 新增
│       ├── success.py       # ManipulationSuccess
│       ├── completion.py    # CompletionRate
│       └── trajectory.py    # TrajectorySimilarity
│
├── api/                       # API 层（扩展）
│   ├── protocol/
│   │   ├── message_types.py # 统一消息类型
│   │   └── messages/
│   │       ├── navigation.py
│   │       └── manipulation.py
│   └── websocket/
│       ├── server.py
│       └── client.py
│
├── dataset/                    # 数据管理（扩展）
│   ├── episode.py            # Episode 数据类
│   ├── dataset.py
│   └── loaders/
│
├── configs/                    # 配置文件
│   ├── tasks/
│   └── benchmarks/
│
└── tests/                      # 测试套件
    ├── unit/
    └── integration/
```

---

*文档版本: 1.0*
*创建时间: 2026-02-12*
*基于: VLN 评测架构 + RoboVerse 设计 + 操作任务扩展方案*

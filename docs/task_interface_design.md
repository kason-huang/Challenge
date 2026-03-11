# 任务接口设计分析

## 概述

本文档从接口设计的角度，系统分析了任务对象应该包含的核心概念，对比了操作任务和导航任务的接口差异，提出了统一的任务接口设计框架。

---

## 目录

1. [任务接口的概念层次](#任务接口的概念层次)
2. [核心概念定义](#核心概念定义)
3. [任务类型抽象](#任务类型抽象)
4. [操作任务 vs 导航任务对比](#操作任务-vs-导航任务对比)
5. [统一接口设计](#统一接口设计)
6. [实现示例](#实现示例)

---

## 任务接口的概念层次

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  任务接口的概念层次                                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ 第1层：核心概念 (BaseTask)                                           │    │
│  │ 所有任务都必须包含的基础概念                                          │    │
│  │ - 场景配置、任务规范、交互接口、空间定义、评价标准                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              │                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ 第2层：任务类型抽象                                                  │    │
│  │ 不同任务类型的专业化接口                                              │    │
│  │ - ManipulationTask: 低层关节控制                                      │    │
│  │ - NavigationTask: 高层运动原语                                        │    │
│  │ - HybridTask: 混合任务                                               │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              │                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ 第3层：扩展概念                                                      │    │
│  │ 可选的高级功能                                                        │    │
│  │ - ActionProcessor: 动作处理                                          │    │
│  │ - StateTracker: 状态跟踪                                             │    │
│  │ - RewardShaper: 奖励塑造                                             │    │
│  │ - TrajectoryGenerator: 轨迹生成                                      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 核心概念定义

### Episode 规范

```python
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import numpy as np

@dataclass
class Episode:
    """Episode 核心规范

    定义一个任务实例的完整信息
    """
    # 唯一标识
    episode_id: str
    scene_id: str

    # 初始条件
    start_state: Dict[str, Any]

    # 任务目标
    goals: Dict[str, Any]

    # 约束条件
    max_episode_steps: int = 500
    time_limit: float = 30.0

    # 可选元数据
    difficulty: float = 1.0
    sub_tasks: Optional[List['SubTask']] = None


@dataclass
class Goal:
    """目标规范

    支持多种目标类型
    """
    goal_type: str  # "point", "object", "image", "pose", "region"

    # 位置目标
    position: Optional[np.ndarray] = None  # [x, y, z]
    rotation: Optional[np.ndarray] = None  # [qx, qy, qz, qw]

    # 容差
    position_tolerance: float = 0.1
    rotation_tolerance: float = 0.1

    # 物体目标
    object_id: Optional[str] = None
    object_category: Optional[str] = None

    # 视觉目标
    reference_image: Optional[np.ndarray] = None
    view_points: Optional[List['ViewPoint']] = None
```

### BaseTask 核心接口

```python
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Optional
import gymnasium as gym

class BaseTask(ABC):
    """任务接口的核心抽象

    这是所有任务的基类，定义了任务的完整契约。
    """

    # ========================
    # 1. 任务规范
    # ========================

    @property
    @abstractmethod
    def scenario(self) -> Any:
        """场景配置：定义环境、机器人、物体

        这是一个**描述性**的概念，说明"任务发生在什么环境中"

        包含：
        - robots: 机器人列表
        - objects: 物体列表
        - environment: 环境参数（光照、物理等）
        - simulator: 仿真器选择
        """
        pass

    @property
    @abstractmethod
    def task_spec(self) -> Dict[str, Any]:
        """任务规范：定义目标、约束、评价标准

        这是一个**规范性**的概念，说明"任务的目标是什么"

        Returns:
            {
                "task_type": "navigation" | "manipulation" | "hybrid",
                "goals": [...],              # 目标列表
                "success_threshold": ...,     # 成功阈值
                "constraints": [...],         # 约束条件
                "metrics": [...],             # 评价指标
                "reward_type": "sparse" | "dense"
            }
        """
        pass

    # ========================
    # 2. 交互接口
    # ========================

    @abstractmethod
    def reset(self, episode: Optional[Episode] = None) -> Tuple[Obs, Info]:
        """重置环境到初始状态

        这是**状态初始化**的概念

        Args:
            episode: 可选的 episode 规范，如果为 None 则使用默认初始化

        Returns:
            (observation, info) 初始观测和信息
        """
        pass

    @abstractmethod
    def step(self, action: Action) -> Tuple[Obs, Reward, Done, Truncated, Info]:
        """执行动作并返回结果

        这是**状态转移**的概念

        Args:
            action: 智能体的动作

        Returns:
            (observation, reward, done, truncated, info)
        """
        pass

    # ========================
    # 3. 空间定义
    # ========================

    @property
    @abstractmethod
    def observation_space(self) -> gym.Space:
        """观测空间：定义智能体能感知什么

        这是一个**信息获取**的概念

        返回 gymnasium.Space 对象，定义观测的结构和范围
        """
        pass

    @property
    @abstractmethod
    def action_space(self) -> gym.Space:
        """动作空间：定义智能体能做什么

        这是一个**行为能力**的概念

        返回 gymnasium.Space 对象，定义动作的结构和范围
        """
        pass

    # ========================
    # 4. 评价标准
    # ========================

    @abstractmethod
    def compute_reward(self, states: TensorState, actions: Action) -> Reward:
        """计算奖励：定义任务目标

        这是一个**目标导向**的概念

        Args:
            states: 当前环境状态
            actions: 执行的动作

        Returns:
            奖励值（标量或向量）
        """
        pass

    @abstractmethod
    def check_termination(self, states: TensorState) -> Done:
        """检查终止：定义任务完成条件

        这是一个**成功判断**的概念

        Args:
            states: 当前环境状态

        Returns:
            是否终止（布尔张量）
        """
        pass
```

---

## 任务类型抽象

### ManipulationTask（操作任务）

```python
class ManipulationTask(BaseTask):
    """操作任务：低层关节控制

    核心特征：
    - 动作是**关节级别**的（位置、速度、力矩）
    - 状态关注**物体和末端执行器**
    - 任务涉及**物理交互**（接触、抓取、碰撞）
    """

    # ========================
    # 操作任务的特有概念
    # ========================

    @property
    @abstractmethod
    def hand_state(self) -> HandState:
        """手部状态：开合、抓取状态、持有物体

        Returns:
            {
                "is_grasping": bool,
                "grasped_object": Optional[str],
                "gripper_opening": float,  # 0-1
                "grasp_force": float
            }
        """
        pass

    @property
    @abstractmethod
    def object_states(self) -> Dict[str, ObjectState]:
        """物体状态：位置、姿态、被抓取状态

        Returns:
            {
                object_name: {
                    "position": np.ndarray,
                    "rotation": np.ndarray,
                    "velocity": np.ndarray,
                    "is_grasped": bool,
                    "is_contacting": List[str]
                }
            }
        """
        pass

    @abstractmethod
    def compute_grasp_reward(self) -> Reward:
        """抓取奖励：接近物体、抓取成功

        多阶段奖励：
        1. 接近阶段：末端到物体的距离
        2. 对齐阶段：末端姿态与抓取姿态的匹配度
        3. 抓取阶段：手指闭合程度
        4. 提升阶段：物体是否被成功提起
        """
        pass

    @abstractmethod
    def check_contact(self, obj1: str, obj2: str) -> bool:
        """接触检测：两个物体是否接触

        Args:
            obj1: 第一个物体名称
            obj2: 第二个物体名称

        Returns:
            是否接触
        """
        pass

    @abstractmethod
    def get_ee_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取末端执行器位姿

        Returns:
            (position, rotation) 位置和旋转
        """
        pass
```

### NavigationTask（导航任务）

```python
class NavigationTask(BaseTask):
    """导航任务：高层运动原语

    核心特征：
    - 动作是**运动级别**的（前进、转向、停止）
    - 状态关注**位姿和目标**
    - 任务涉及**空间推理**（定位、路径规划）
    """

    # ========================
    # 导航任务的特有概念
    # ========================

    @property
    @abstractmethod
    def agent_pose(self) -> Pose:
        """智能体位姿：位置和朝向

        Returns:
            {
                "position": np.ndarray,  # [x, y, z]
                "rotation": np.ndarray,  # [qx, qy, qz, qw]
                "heading": float         # 朝向角度（弧度）
            }
        """
        pass

    @property
    @abstractmethod
    def goal_position(self) -> np.ndarray:
        """目标位置：导航目标点

        Returns:
            目标位置 [x, y, z]
        """
        pass

    @abstractmethod
    def compute_distance_to_goal(self) -> float:
        """到目标的距离：空间度量

        Returns:
            欧氏距离（或其他度量）
        """
        pass

    @abstractmethod
    def check_collision(self) -> bool:
        """碰撞检测：是否发生碰撞

        Returns:
            是否碰撞
        """
        pass

    @property
    @abstractmethod
    def navigation_map(self) -> np.ndarray:
        """导航地图：可行走区域、障碍物

        Returns:
            地图数组，可编码：
            - 可行走区域
            - 障碍物
            - 未探索区域
            - 距离场
        """
        pass

    @abstractmethod
    def get_shortest_path(self, start: np.ndarray, goal: np.ndarray) -> List[np.ndarray]:
        """获取最短路径

        Args:
            start: 起点
            goal: 终点

        Returns:
            路径点列表
        """
        pass
```

---

## 操作任务 vs 导航任务对比

### 接口概念对比表

| 维度 | 操作任务 | 导航任务 |
|------|---------|---------|
| **动作抽象层次** | 低层（关节） | 高层（运动原语） |
| | 7-20个关节自由度 | 3-10个离散动作或连续速度 |
| **状态关注点** | 物体位置、姿态 | 机器人位姿、目标距离 |
| | 末端执行器位置 | 可行走区域、障碍物 |
| | 抓取状态、接触状态 | 局部地图、全局定位 |
| **任务目标** | 物体操作 | 到达指定位置 |
| | 抓取、放置、堆叠 | 避障、路径跟随 |
| | 精密操作 | 大范围移动 |
| **奖励设计** | 稠密奖励（距离、接近） | 稠密或稀疏奖励 |
| | 多阶段奖励 | 距离奖励或稀疏成功 |
| **成功条件** | 物体到达目标位姿 | 机器人接近目标点 |
| | 抓取成功、放置成功 | 阈值内停留 |
| **物理交互** | 必须（接触、碰撞） | 避免（碰撞检测） |
| **空间尺度** | 小范围（操作空间） | 大范围（环境尺度） |
| **时间尺度** | 短（秒级） | 长（分钟级） |
| **状态表示** | TensorState 包含： | TensorState 包含： |
| | 物体、机器人关节 | 机器人 root 状态 |
| | 末端执行器 | 地图/语义信息 |

### 动作空间对比

```python
# ========================
# 操作任务：关节级控制
# ========================

ManipulationTask.action_space = Box(
    low=[-2.9, -1.8, -2.9, -3.1, -2.9, -0.0, -2.9, 0.0, 0.0],
    high=[2.9, 1.8, 2.9, -0.1, 2.9, 3.8, 2.9, 0.04, 0.04],
    shape=(9,)  # 9个关节
)

# 动作含义：直接设置关节目标位置
action = [0.5, -0.3, 1.0, ...]  # 每个值对应一个关节


# ========================
# 导航任务：运动原语
# ========================

NavigationTask.action_space = Discrete(4)
# 0: STOP
# 1: MOVE_FORWARD (0.25m)
# 2: TURN_LEFT (15°)
# 3: TURN_RIGHT (15°)

# 或连续速度控制
NavigationTask.action_space = Box(
    low=[-0.5, -1.0],  # [min_linear_vel, min_angular_vel]
    high=[0.5, 1.0],   # [max_linear_vel, max_angular_vel]
    shape=(2,)
)
```

### 观测空间对比

```python
# ========================
# 操作任务观测
# ========================

ManipulationTask.observation_space = Dict({
    "joint_positions": Box(-np.pi, np.pi, (9,)),
    "joint_velocities": Box(-inf, inf, (9,)),
    "ee_position": Box(-inf, inf, (3,)),
    "ee_orientation": Box(-1, 1, (4,)),
    "object_position": Box(-inf, inf, (3,)),
    "object_orientation": Box(-1, 1, (4,)),
    "gripper_opening": Box(0, 1, (1,)),
    "rgb_image": Box(0, 255, (H, W, 3)),
})


# ========================
# 导航任务观测
# ========================

NavigationTask.observation_space = Dict({
    "robot_position": Box(-inf, inf, (3,)),
    "robot_heading": Box(-np.pi, np.pi, (1,)),
    "goal_position": Box(-inf, inf, (3,)),
    "goal_distance": Box(0, inf, (1,)),
    "goal_angle": Box(-np.pi, np.pi, (1,)),
    "collision": Box(0, 1, (1,)),
    "local_map": Box(0, 1, (H, W)),  # 占用栅格
    "rgb_image": Box(0, 255, (H, W, 3)),
})
```

### 奖励函数对比

```python
# ========================
# 操作任务：多阶段稠密奖励
# ========================

class ManipulationReward:
    def compute(self, states, actions):
        reward = 0.0

        # 阶段1：接近物体
        ee_to_obj_dist = compute_distance(ee_pos, obj_pos)
        reward += -0.1 * ee_to_obj_dist  # 距离越小奖励越高

        # 阶段2：抓取
        if is_grasping:
            reward += 1.0  # 抓取奖励

        # 阶段3：移动到目标
        obj_to_goal_dist = compute_distance(obj_pos, goal_pos)
        reward += -0.05 * obj_to_goal_dist

        # 阶段4：放置成功
        if is_placed and check_success():
            reward += 10.0  # 成功奖励

        return reward


# ========================
# 导航任务：稀疏或距离奖励
# ========================

class NavigationReward:
    def __init__(self, reward_type="dense"):
        self.reward_type = reward_type

    def compute(self, states, actions):
        if self.reward_type == "sparse":
            # 稀疏奖励：只有成功才有奖励
            if self.check_success(states):
                return 1.0
            else:
                return 0.0

        elif self.reward_type == "dense":
            # 稠密奖励：基于距离
            dist_to_goal = self.compute_distance_to_goal(states)
            reward = -dist_to_goal * 0.01  # 距离越小奖励越高

            # 碰撞惩罚
            if self.check_collision(states):
                reward -= 0.5

            return reward
```

---

## 统一接口设计

### 核心设计原则

1. **分层抽象**：通过 ActionProcessor 解耦动作表示
2. **统一状态**：通过 TensorState 统一状态表示
3. **模块化**：通过组合模式支持复杂任务
4. **扩展性**：支持自定义任务类型和处理器

### ActionProcessor 抽象

```python
class ActionProcessor(ABC):
    """动作处理器：将抽象动作转换为具体控制

    这是**动作解释**的概念，解耦了动作表示和控制实现
    """

    @abstractmethod
    def process(self, action: Action) -> ControlCommand:
        """将抽象动作转换为控制命令

        Args:
            action: 智能体输出的抽象动作

        Returns:
            控制命令，格式为：
            {
                robot_name: {
                    "dof_pos_target": {...},
                    "dof_vel_target": {...},
                    "dof_effort_target": {...}
                }
            }
        """
        pass


class JointActionProcessor(ActionProcessor):
    """关节动作处理器：直接关节控制

    用于操作任务
    """

    def __init__(self, robot_cfg):
        self.joint_limits = robot_cfg.joint_limits
        self.joint_names = list(robot_cfg.joint_limits.keys())

    def process(self, action: np.ndarray) -> ControlCommand:
        """将连续动作转换为关节目标位置

        Args:
            action: [num_joints] 范围 [-1, 1]

        Returns:
            {robot_name: {"dof_pos_target": {joint_name: value}}}
        """
        # 反归一化到实际关节限位
        action = self.unnormalize(action)

        # 转换为字典格式
        joint_targets = {
            name: action[i].item()
            for i, name in enumerate(self.joint_names)
        }

        return {self.robot_name: {"dof_pos_target": joint_targets}}


class NavigationActionProcessor(ActionProcessor):
    """导航动作处理器：离散动作到运动

    用于导航任务
    """

    def __init__(self, robot_name, handler):
        self.robot_name = robot_name
        self.handler = handler
        self.action_params = {
            "move_forward": {"distance": 0.25},
            "turn_left": {"angle": 15.0},
            "turn_right": {"angle": -15.0},
        }

    def process(self, action: int) -> ControlCommand:
        """将离散动作转换为运动命令

        Args:
            action: 整数，0=stop, 1=forward, 2=left, 3=right

        Returns:
            {robot_name: {"dof_pos_target": {...}}}
        """
        action_map = {
            0: self._stop,
            1: self._move_forward,
            2: self._turn_left,
            3: self._turn_right,
        }

        return action_map[action]()

    def _move_forward(self):
        # 获取当前状态
        states = self.handler.get_states()
        current_pos = states.robots[self.robot_name].root_state[0, 0:3]

        # 计算目标位置
        forward_dir = np.array([0, 0, -1])
        distance = self.action_params["move_forward"]["distance"]
        target_pos = current_pos.numpy() + forward_dir * distance

        # 使用运动规划或逆运动学
        joint_targets = self._plan_motion(target_pos)

        return {self.robot_name: {"dof_pos_target": joint_targets}}
```

### StateTracker 抽象

```python
class StateTracker(ABC):
    """状态跟踪器：维护任务相关状态

    这是**状态管理**的概念
    """

    @abstractmethod
    def update(self, states: TensorState) -> None:
        """更新内部状态"""
        pass

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """获取当前状态"""
        pass


class ManipulationStateTracker(StateTracker):
    """操作状态跟踪器"""

    def __init__(self):
        self.grasped_object = None
        self.last_ee_position = None
        self.contact_pairs = set()

    def update(self, states):
        # 更新抓取状态
        self.grasped_object = self._check_grasp(states)

        # 更新接触状态
        self.contact_pairs = self._check_contacts(states)

        # 更新末端位置
        self.last_ee_position = states.robots["robot"].ee_position


class NavigationStateTracker(StateTracker):
    """导航状态跟踪器"""

    def __init__(self):
        self.path_history = []
        self.collision_count = 0
        self.distance_to_goal_history = []

    def update(self, states):
        # 记录路径
        current_pos = states.robots["robot"].root_state[0, 0:3]
        self.path_history.append(current_pos)

        # 检测碰撞
        if self._check_collision(states):
            self.collision_count += 1

        # 记录距离
        dist = self._compute_distance_to_goal(states)
        self.distance_to_goal_history.append(dist)
```

### RewardShaper 抽象

```python
class RewardShaper(ABC):
    """奖励塑造器：组合多个奖励组件

    这是**奖励工程**的概念
    """

    def __init__(self, components: List[RewardComponent]):
        """
        Args:
            components: 奖励组件列表，每个组件有权重
        """
        self.components = components

    def compute(self, states: TensorState, actions: Action) -> Reward:
        """计算总奖励"""
        total_reward = 0.0
        info = {}

        for component in self.components:
            reward, component_info = component.compute(states, actions)
            total_reward += component.weight * reward
            info[component.name] = component_info

        return total_reward, info


class RewardComponent(ABC):
    """奖励组件基类"""

    def __init__(self, weight: float, name: str):
        self.weight = weight
        self.name = name

    @abstractmethod
    def compute(self, states: TensorState, actions: Action) -> Tuple[float, Dict]:
        pass


class DistanceReward(RewardComponent):
    """距离奖励组件"""

    def __init__(self, weight: float, scale: float = 1.0):
        super().__init__(weight, "distance")
        self.scale = scale

    def compute(self, states, actions):
        pos1 = states.objects["obj1"].root_state[0, 0:3]
        pos2 = states.objects["obj2"].root_state[0, 0:3]
        distance = torch.norm(pos1 - pos2)

        reward = -distance * self.scale
        info = {"distance": distance.item()}

        return reward, info


class GraspReward(RewardComponent):
    """抓取奖励组件"""

    def compute(self, states, actions):
        is_grasping = check_grasp(states)

        if is_grasping:
            reward = 1.0
        else:
            reward = 0.0

        info = {"is_grasping": is_grasping}
        return reward, info
```

### TrajectoryGenerator 抽象

```python
class TrajectoryGenerator(ABC):
    """轨迹生成器：为高层动作生成运动轨迹

    这是**运动规划**的概念
    """

    @abstractmethod
    def generate(self, action: Action) -> Trajectory:
        """生成轨迹

        Returns:
            轨迹，包含时间戳和位姿序列
        """
        pass


class NavigationTrajectoryGenerator(TrajectoryGenerator):
    """导航轨迹生成器"""

    def __init__(self, num_steps: int = 50):
        self.num_steps = num_steps

    def generate(self, action: int) -> Trajectory:
        """为离散动作生成平滑轨迹"""
        if action == 1:  # MOVE_FORWARD
            return self._generate_forward_trajectory()
        elif action == 2:  # TURN_LEFT
            return self._generate_turn_trajectory(angle=15.0)
        elif action == 3:  # TURN_RIGHT
            return self._generate_turn_trajectory(angle=-15.0)
        else:  # STOP
            return self._generate_stop_trajectory()

    def _generate_forward_trajectory(self, distance: float = 0.25):
        """生成前进轨迹（线性插值）"""
        import numpy as np
        from scipy.interpolate import interp1d

        # 获取当前位置
        start_pos = self._get_current_position()

        # 计算终点
        forward = np.array([0, 0, -1])
        end_pos = start_pos + forward * distance

        # 线性插值
        times = np.linspace(0, 1, self.num_steps)
        positions = np.array([start_pos + t * (end_pos - start_pos)
                             for t in times])

        return Trajectory(times=times, positions=positions)


class ManipulationTrajectoryGenerator(TrajectoryGenerator):
    """操作轨迹生成器"""

    def generate(self, action: np.ndarray) -> Trajectory:
        """从关节目标位置生成轨迹

        使用运动学插值或优化
        """
        # 使用样条插值或运动规划
        return self._plan_joint_trajectory(action)
```

### 统一任务实现

```python
class UnifiedTask(BaseTask):
    """统一任务接口：支持操作和导航

    关键设计原则：
    1. 动作空间抽象化：通过 ActionProcessor 解耦
    2. 状态表示统一化：通过 TensorState 统一
    3. 任务模块化：通过组合模式支持复杂任务
    """

    def __init__(self, scenario, task_spec):
        # 核心组件
        self.scenario = scenario
        self.task_spec = task_spec
        self.handler = self._create_handler(scenario)

        # 根据任务类型选择组件
        self.action_processor = self._create_action_processor()
        self.state_tracker = self._create_state_tracker()
        self.reward_shaper = self._create_reward_shaper()

        # 可选组件
        self.traj_generator = self._create_trajectory_generator()

    def _create_action_processor(self) -> ActionProcessor:
        """工厂方法：根据任务类型创建动作处理器"""
        task_type = self.task_spec["task_type"]

        if task_type == "manipulation":
            return JointActionProcessor(self.scenario.robots[0])

        elif task_type == "navigation":
            return NavigationActionProcessor(
                self.scenario.robots[0].name,
                self.handler
            )

        elif task_type == "hybrid":
            return HybridActionProcessor(self.scenario.robots)

    def step(self, action: Action):
        """统一的步进接口

        关键：动作的解析由 ActionProcessor 处理
        """
        # 1. 将抽象动作转换为控制命令
        control_commands = self.action_processor.process(action)

        # 2. 如果有轨迹生成器，使用轨迹执行
        if self.traj_generator is not None:
            trajectory = self.traj_generator.generate(action)
            control_commands = self._execute_trajectory(trajectory)

        # 3. 应用到仿真器
        self.handler.set_dof_targets([control_commands])

        # 4. 物理步进
        self.handler.simulate()

        # 5. 获取新状态
        states = self.handler.get_states()

        # 6. 更新状态跟踪器
        self.state_tracker.update(states)

        # 7. 计算奖励和终止
        reward, reward_info = self.reward_shaper.compute(states, action)
        done = self.check_termination(states)
        info = {
            **reward_info,
            **self.state_tracker.get_state()
        }

        return states, reward, done, False, info
```

---

## 实现示例

### 示例1：简单抓取任务

```python
@register_task("manipulation.simple_pick")
class SimplePickTask(ManipulationTask):
    """简单抓取任务"""

    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="cube",
                size=(0.05, 0.05, 0.05),
                physics=PhysicStateType.RIGIDBODY,
            ),
        ],
        robots=["franka"],
    )

    max_episode_steps = 200

    def _get_initial_states(self):
        return [{
            "objects": {
                "cube": {
                    "pos": torch.tensor([0.5, 0.0, 0.1]),
                    "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                },
            },
            "robots": {
                "franka": {
                    "joint_pos": self._default_joint_pos,
                },
            },
        }]

    def _terminated(self, states):
        """成功条件：物体被抓起并提升"""
        cube_pos = states.objects["cube"].root_state[0, 2]  # z坐标
        return cube_pos > 0.3  # 提升到 30cm 以上

    def compute_reward(self, states, actions):
        """稠密奖励"""
        ee_pos = states.robots["franka"].ee_position
        cube_pos = states.objects["cube"].root_state[0, 0:3]

        # 距离奖励
        distance = torch.norm(ee_pos - cube_pos)
        reward = -0.1 * distance

        # 抓取奖励
        if self.check_contact("franka", "cube"):
            reward += 1.0

        # 提升奖励
        cube_z = states.objects["cube"].root_state[0, 2]
        reward += cube_z * 2.0

        return reward
```

### 示例2：点导航任务

```python
@register_task("navigation.point_goal")
class PointGoalTask(NavigationTask):
    """点目标导航任务"""

    scenario = ScenarioCfg(
        objects=[],
        robots=["stretch"],
    )

    max_episode_steps = 500

    def _get_initial_states(self):
        # 随机生成起点和目标
        start_pos = self._random_position()
        goal_pos = self._random_position()

        return [{
            "robots": {
                "stretch": {
                    "pos": torch.tensor(start_pos),
                    "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                },
            },
            "extras": {
                "goal_position": torch.tensor(goal_pos),
            },
        }]

    def _terminated(self, states):
        """成功条件：接近目标"""
        robot_pos = states.robots["stretch"].root_state[0, 0:3]
        goal_pos = states.extras["goal_position"]

        distance = torch.norm(robot_pos - goal_pos)
        return distance < 0.2  # 20cm 阈值

    def compute_reward(self, states, actions):
        """稀疏奖励"""
        if self._terminated(states):
            return 1.0
        else:
            return 0.0

    @property
    def action_space(self):
        """离散动作空间"""
        return gym.spaces.Discrete(4)
        # 0: stop, 1: forward, 2: left, 3: right
```

### 示例3：混合任务

```python
@register_task("hybrid.pick_and_nav")
class PickAndNavigateTask(UnifiedTask):
    """抓取并导航任务：先抓取物体，然后移动到目标位置"""

    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(name="target_object", ...),
        ],
        robots=["stretch"],  # 移动操作机器人
    )

    def __init__(self, scenario, device=None):
        super().__init__(scenario, device)

        # 任务阶段
        self.phase = "manipulation"  # 或 "navigation"

        # 为每个阶段创建独立的处理器
        self.manipulation_processor = JointActionProcessor(...)
        self.navigation_processor = NavigationActionProcessor(...)

    def step(self, action):
        """根据当前阶段选择处理器"""
        if self.phase == "manipulation":
            # 使用操作处理器
            control_commands = self.manipulation_processor.process(action)

            # 检查是否切换到导航阶段
            if self._check_grasp_success():
                self.phase = "navigation"

        else:  # navigation
            # 使用导航处理器
            control_commands = self.navigation_processor.process(action)

        # 执行控制命令
        self.handler.set_dof_targets([control_commands])
        self.handler.simulate()

        # 其余逻辑...
        states = self.handler.get_states()
        reward = self.compute_reward(states, action)
        done = self._terminated(states)

        return states, reward, done, False, {}

    def _terminated(self, states):
        """最终成功条件：物体被移动到目标位置"""
        obj_pos = states.objects["target_object"].root_state[0, 0:3]
        goal_pos = states.extras["goal_position"]

        # 检查物体是否被抓取并接近目标
        is_grasped = self.check_contact("stretch", "target_object")
        is_at_goal = torch.norm(obj_pos - goal_pos) < 0.2

        return is_grasped and is_at_goal
```

---

## 总结

### 关键概念总结表

| 概念类别 | 核心概念 | 操作任务 | 导航任务 |
|---------|---------|---------|---------|
| **环境** | Scenario | 机器人+物体+场景 | 机器人+场景+地图 |
| **目标** | Goal Spec | 物体目标位姿 | 目标位置/区域 |
| **动作** | Action Space | 关节位置/速度 | 运动原语/速度 |
| **观测** | Observation | 关节+物体+视觉 | 位姿+地图+视觉 |
| **评价** | Reward/Termination | 稠密/多阶段 | 稠密或稀疏 |
| **处理** | ActionProcessor | 关节控制映射 | 运动命令映射 |
| **状态** | StateTracker | 抓取/接触状态 | 位姿/目标距离 |
| **奖励** | RewardShaper | 距离/接触/成功 | 距离/时间/成功 |

### 设计建议

1. **明确任务类型**：在设计任务时，首先确定是操作型、导航型还是混合型

2. **选择合适的抽象层次**：
   - 操作任务：低层关节控制
   - 导航任务：高层运动原语
   - 混合任务：阶段切换或多处理器

3. **使用组合模式**：通过 ActionProcessor、StateTracker、RewardShaper 等组件组合，而非复杂继承

4. **保持接口一致性**：不同任务类型应共享 BaseTask 接口，确保算法兼容性

5. **支持扩展**：通过注册系统和配置驱动，支持新任务类型的快速添加

### 接口设计最佳实践

```python
# ✅ 推荐：清晰的接口分离
class MyTask(BaseTask):
    def __init__(self, scenario, task_spec):
        self.action_processor = create_processor(task_spec["action_type"])
        self.state_tracker = create_tracker(task_spec["task_type"])
        self.reward_shaper = create_shaper(task_spec["reward_type"])

    def step(self, action):
        commands = self.action_processor.process(action)
        self.handler.set_dof_targets(commands)
        # ...

# ❌ 避免：混合不同层次的逻辑
class MyTask(BaseTask):
    def step(self, action):
        if self.task_type == "manipulation":
            # 直接处理关节控制
            self.handler.set_dof_targets(action)
        elif self.task_type == "navigation":
            # 在这里做运动规划
            trajectory = self.plan_trajectory(action)
            # ...
```

**核心思想**：通过**分层抽象**和**组合模式**，让不同任务类型共享核心接口，同时在各自的专业领域保持灵活性。

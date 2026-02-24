# RoboVerse 核心概念与关系图谱

> **概述**: RoboVerse 是一个统一的多模拟器机器人学习平台，本文档详细描述其核心概念、对象及相互关系。

---

## 目录

- [一、整体架构视图](#一整体架构视图)
- [二、配置层概念](#二配置层概念)
- [三、任务层概念](#三任务层概念)
- [四、仿真层概念](#四仿真层概念)
- [五、学习层概念](#五学习层概念)
- [六、评测层概念](#六评测层概念)
- [七、数据流概念](#七数据流概念)
- [八、核心关系图谱](#八核心关系图谱)

---

## 一、整体架构视图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        RoboVerse 统一平台                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    │
│  │   配置层      │    │   任务层       │    │   仿真层       │    │
│  │  Scenario    │    │  BaseTaskEnv   │    │  BaseSimHandler │    │
│  │  RobotCfg    │    │  TaskRegistry  │    │  Handler(s)    │    │
│  │  Objects     │    │  Checker/Det.   │    │                │    │
│  └──────────────┘    └──────────────┘    └──────────────┘    │
│                                                                   │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    │
│  │   学习层      │    │   评测层       │    │   数据层        │    │
│  │  IL/VLA/RL   │    │  EvalRunner    │    │  Demos/Trajs   │    │
│  └──────────────┘    └──────────────┘    └──────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 二、配置层概念

### 2.1 核心配置类

#### ScenarioCfg（场景配置）

**定义**: 完整的任务场景配置

**属性**:
```python
@configclass
class ScenarioCfg:
    # === 资产组件 ===
    scene: SceneCfg | None           # 场景（桌面、房间等）
    robots: list[RobotCfg]          # 机器人列表
    objects: list[BaseObjCfg]        # 场景物体
    lights: list[BaseLightCfg]        # 灯光
    cameras: list[BaseCameraCfg]      # 相机
    ground: GroundCfg | None          # 地面
    gs_scene: GSSceneCfg | None    # Gaussian Splatting场景

    # === 运行时配置 ===
    simulator: str = None             # 模拟器选择
    num_envs: int = 1               # 并行环境数
    headless: bool = False            # 无头模式
    env_spacing: float = 1.0          # 环境间距
    decimation: int = 15              # 物理降频
    gravity: tuple = (0.0, 0.0, -9.81)  # 重力

    # === 渲染配置 ===
    render: RenderCfg                # 渲染参数
    sim_params: SimParamCfg           # 物理参数
```

**组合关系**:
```
ScenarioCfg
    ├── [1..N] RobotCfg          # 包含多个机器人
    ├── [0..M] BaseObjCfg         # 包含多个物体
    ├── [0..K] BaseLightCfg       # 包含多个灯源
    ├── [0..L] BaseCameraCfg       # 包含多个相机
    ├── GroundCfg                   # 地面配置
    └── RenderCfg                   # 渲染配置
```

**使用场景**:
- 环境创建 (`gym.make()`)
- 任务初始化 (`BaseTaskEnv(scenario, device)`)
- 向量化配置 (`num_envs` 参数)

---

#### RobotCfg（机器人配置）

**定义**: 单个机器人的完整配置

**属性**:
```python
@configclass
class RobotCfg:
    name: str                             # 机器人名称
    urdf_path: str | None                # URDF文件路径
    usd_path: str | None                 # USD文件路径
    mjcf_path: str | None                # MJCF文件路径
    scale: float = 1.0                      # 缩放因子

    # 物理属性
    gripper_open_q: list[float]           # 夹爪打开位置
    gripper_close_q: list[float]          # 夹爪关闭位置
    joint_limits: dict[str, tuple]       # 关节限制 [min, max]
    ee_body_name: str | None             # 末端执行器名称

    # 初始状态
    fix_base_link: bool = False            # 固定基座
    default_qpos: dict | None            # 默认关节位置
    default_pos: dict | None              # 默认位置
    default_quat: dict | None            # 默认四元数
```

**组合关系**:
```
RobotCfg
    ├── gripper_open_q: [finger1, finger2, ...]  # 夹爪关节
    ├── joint_limits: {                        # 所有关节限制
    │   "joint1": (-1.57, 1.57),
    │   "joint2": (-1.57, 1.57),
    │   ...
    │   }
    └── ee_body_name: "panda_hand"           # 末端链接
```

**使用场景**:
- Handler 创建机器人实例
- IK 求解器配置
- 关节控制范围检查

---

#### BaseObjCfg（物体配置）

**定义**: 场景中可操作物体的基类

**属性**:
```python
class BaseObjCfg:
    name: str                              # 物体名称
    position: tuple[float, float, float]    # 位置 [x, y, z]
    rotation: tuple[float, float, float]    # 旋转 [w, x, y, z]
    scale: tuple[float, float, float]     # 缩放 [sx, sy, sz]
    mass: float = 1.0                      # 质量
    collision_enabled: bool = True          # 启用碰撞

    # 物理类型
    physics: PhysicStateType = PhysicStateType.RIGIDBODY
    fix_base_link: bool = True              # 固定基座链接
```

**具体子类**:
- `PrimitiveCubeCfg` - 基础立方体
- `PrimitiveCylinderCfg` - 圆柱体
- `PrimitiveSphereCfg` - 球体
- `MeshObjCfg` - 网格物体

---

#### BaseLightCfg（灯光配置）

**定义**: 灯光光源配置

**属性**:
```python
class BaseLightCfg:
    name: str                              # 灯光名称
    position: tuple[float, float, float]    # 位置
    rotation: tuple[float, float, float]    # 旋转（四元数）
    intensity: float                        # 强度
    color: tuple[float, float, float]     # RGB颜色 [0-1]
    radius: float | None                  # 半径（点光源）
```

**具体子类**:
- `DistantLightCfg` - 方向光
- `SphereLightCfg` - 球面点光源
- `DiskLightCfg` - 圆盘面光源

---

#### BaseCameraCfg（相机配置）

**定义**: 相机传感器配置

**属性**:
```python
class BaseCameraCfg:
    name: str                              # 相机名称
    data_types: list[str]                # 数据类型 ["rgb", "depth", "segmentation"]
    width: int = 256                      # 图像宽度
    height: int = 256                     # 图像高度
    position: tuple[float, float, float]    # 位置
    rotation: tuple[float, float, float]    # 旋转（四元数）
    look_at: tuple[float, float, float]    # 注视目标
    fov: float = 60                        # 视场角
    near: float = 0.01                     # 近裁剪面
    far: float = 10.0                      # 远裁剪面
```

---

### 2.2 域随机化配置

#### DRConfig（域随机化配置）

**定义**: 控制域随机化的程度和方式

**属性**:
```python
@dataclass
class DRConfig:
    level: Literal[0, 1, 2, 3] = 0
    """
    随机化级别：
    - 0: 无随机化
    - 1: 场景 + 材质随机化
    - 2: Level 1 + 灯光随机化
    - 3: Level 2 + 相机随机化
    """

    scene_mode: Literal[0, 1, 2, 3] = 0
    """
    场景模式：
    - 0: 手动几何
    - 1: USD Table + 手动环境
    - 2: USD Scene + USD Table
    - 3: Full USD (Scene + Table + Desktop)
    """

    randomization_seed: int | None = None
    """随机化种子（None=随机）"""
```

**级别详解**:
| Level | 场景 | 材质 | 灯光 | 相机 | 测试能力 |
|-------|------|------|------|------|----------|
| 0 | ❌ | ❌ | ❌ | ❌ | 标准能力 |
| 1 | ✅ | ✅ | ❌ | ❌ | 泛化能力 |
| 2 | ✅ | ✅ | ✅ | ❌ | 鲁棒性 |
| 3 | ✅ | ✅ | ✅ | ✅ | 视角不变性 |

---

#### DomainRandomizationManager（域随机化管理器）

**职责**: 统一管理所有类型的随机化

**核心方法**:
```python
class DomainRandomizationManager:
    def __init__(self, config: DRConfig, scenario, handler, init_states):
        """初始化DR Manager"""
        self.config = config
        self.scenario = scenario
        self.handler = handler
        self.init_states = init_states

    def apply_randomization(self, demo_idx: int, is_initial: bool):
        """应用场景和材质随机化"""

    def update_positions_to_table(self, demo_idx: int, env_id: int):
        """更新物体位置（USD模式）"""

    def update_camera_look_at(self, env_id: int):
        """更新相机注视目标"""

    def apply_camera_randomization(self):
        """应用相机位置随机化"""
```

**使用场景**:
- Demo collection（演示数据收集）
- Policy evaluation（策略泛化测试）
- 4级渐进式泛化测试

---

## 三、任务层概念

### 3.1 任务基类

#### BaseTaskEnv（任务环境基类）

**定义**: 所有任务的父类，定义任务接口和生命周期

**核心属性**:
```python
class BaseTaskEnv:
    max_episode_steps = 100        # 最大步数
    traj_filepath = None            # 演示轨迹路径

    # 组件
    scenario: ScenarioCfg            # 场景配置
    handler: BaseSimHandler        # 模拟器处理器
    num_envs: int                # 环境数量
    device: str                   # 设备

    # 回调机制
    pre_physics_step_callback: list[Callable]   # 物理步进前
    post_physics_step_callback: list[Callable]  # 物理步进后
    reset_callback: list[Callable]                # 重置时
    close_callback: list[Callable]                # 关闭时
```

**必须实现的方法**:
```python
def _observation(self, env_states: Obs) -> Obs:
    """获取环境观测"""
    # 返回机器人状态、相机图像等
    raise NotImplementedError

def _reward(self, env_states: Obs) -> Reward:
    """计算奖励信号"""
    return torch.zeros(self.num_envs)

def _terminated(self, env_states: Obs) -> Termination:
    """判断是否终止（成功/失败）"""
    return torch.zeros(self.num_envs, dtype=torch.bool)

def _time_out(self, env_states) -> torch.Tensor:
    """判断是否超时"""
    return self._episode_steps >= self.max_episode_steps
```

**可选实现的方法**:
```python
def _privileged_observation(self, env_states: Obs) -> Obs:
    """获取特权观测（RL训练用，实际评测不可用）"""

def _observation_space(self) -> gym.Space:
    """定义观测空间"""

def _action_space(self) -> gym.Space:
    """定义动作空间"""

def _extra_spec(self) -> dict[str, BaseQueryType]:
    """额外查询规范（IK、接触力等）"""
```

**生命周期**:
```
创建
  ↓
__init__(scenario, device)
  ↓
_prepare_callbacks()
  ↓
_get_initial_states()
  ↓
handler.launch()
  ↓
[就绪状态]
```

---

### 3.2 任务注册系统

#### TaskRegistry（任务注册表）

**定义**: 管理所有已注册的任务

**注册装饰器**:
```python
from metasim.task.registry import register_task

@register_task("namespace.name", "task_id")
class MyTask(BaseTaskEnv):
    """任务实现"""
    scenario = ScenarioCfg(...)
```

**使用场景**:
```python
from metasim.task.registry import get_task_class

# 通过ID获取任务类
task_cls = get_task_class("maniskill.push_cube")

# 通过ID获取完整名称
task_full_name = get_task_full_name("push_cube")

# 列出所有任务
all_tasks = list_tasks()
```

**注册示例**:
| 命名空间 | 任务ID | 任务类 | 轨迹路径 |
|----------|---------|---------|-----------|
| `maniskill` | `push_cube` | PushCubeCfg | `roboverse_data/trajs/maniskill/push_cube/v2` |
| `maniskill` | `pick_cube` | PickCubeCfg | `roboverse_data/trajs/maniskill/pick_cube/v2` |
| `libero_90` | `libero_kitchen_scene1` | LiberoKitchenScene1Task | `roboverse_data/trajs/libero_90/...` |

---

### 3.3 成功检测系统

#### Checker/Detector（检测器/探测仪）

**定义**: 用于判断任务成功条件的组件

**类型层次**:
```
BaseQueryType (基类)
    ├── ContactForceQuery     # 接触力查询
    ├── StateQuery            # 状态查询
    ├── Relative2DSphereDetector   # 2D球形区域探测器
    ├── DetectedChecker      # 检测器
    └── ...
```

**使用示例**:
```python
# 相对位置探测器
detector = Relative2DSphereDetector(
    base_obj_name="goal_region",
    relative_pos=(0.0, 0.0, 0.0),
    radius=0.15,
    axis=(0, 1)  # XY平面
)

# 检测器
checker = DetectedChecker(
    detector=detector,
    obj_name="cube"
)

# 在任务中使用
def _terminated(self, env_states):
    return checker.check(env_states)
```

**检测器类型**:
- **几何检测**: 位置关系、距离、角度
- **接触检测**: 接触力、碰撞状态
- **状态检测**: 速度、加速度、稳定性

---

## 四、仿真层概念

### 4.1 模拟器处理器

#### BaseSimHandler（模拟器处理器基类）

**定义**: 所有模拟器实现的抽象接口

**核心方法**:
```python
class BaseSimHandler:
    def __init__(self, scenario, extra_spec):
        """初始化模拟器"""
        self.scenario = scenario
        self.robots: list[RobotState] = []
        self.objects: list[ObjectState] = []
        self.lights: list[LightState] = []
        self.cameras: list[CameraState] = []

    def launch(self) -> None:
        """启动模拟器"""
        # 创建机器人、物体、灯光、相机
        # 初始化物理世界

    def step(self) -> None:
        """物理步进"""
        # 执行物理仿真

    def set_dof_targets(self, actions) -> None:
        """设置关节目标位置"""
        # 控制机器人关节

    def get_states(self, mode="tensor") -> TensorStates:
        """获取当前状态"""
        # 返回机器人、物体、灯光、相机状态

    def render(self) -> np.ndarray:
        """渲染场景"""
        # 返回RGB图像

    def close(self) -> None:
        """关闭模拟器"""
        # 清理资源
```

**状态类型**:
```python
class TensorStates:
    robots: dict[str, RobotState]      # 机器人状态
    objects: dict[str, ObjectState]       # 物体状态
    images: dict[str, np.ndarray]        # 相机图像
```

---

### 4.2 模拟器类型

#### 支持的模拟器

| 模拟器 | Handler类 | 向量化 | 性能 | GPU加速 | 适用场景 |
|---------|-----------|--------|------|---------|----------|
| **mujoco** | MujocoHandler | ❌ | 中等 | ❌ | 快速原型 |
| **mjx** | MJXHandler | ✅ (JAX) | 极快 | ✅ (JAX) | 大规模RL |
| **isaacgym** | IsaacGymHandler | ✅ | 极快 | ✅ (CUDA) | GPU加速RL |
| **isaaclab** | IsaacLabHandler | ✅ | 高 | ✅ (CUDA) | 最新栈 |
| **genesis** | GenesisHandler | ✅ | 极快 | ✅ (CUDA) | 现代物理 |
| **pybullet** | PyBulletHandler | ✅ | 中等 | ❌ | 兼容性好 |
| **sapien2** | SAPIEN2Handler | ✅ | 高 | ✅ (CUDA) | 渲染优先 |
| **sapien3** | SAPIEN3Handler | ✅ | 高 | ✅ (CUDA) | 最新版本 |

**选择策略**:
- **评测**: MuJoCo（稳定、兼容）
- **RL训练**: IsaacGym 或 MJX（原生向量化、GPU）
- **高质量渲染**: IsaacLab 或 SAPIEN3

---

### 4.3 状态数据结构

#### Obs（观测数据）

**定义**: 环境返回的完整观测

**结构**:
```python
@dataclass
class Obs:
    robots: dict[str, RobotState]      # 机器人状态
    objects: dict[str, ObjectState]       # 物体状态
    images: dict[str, np.ndarray]        # 相机图像
```

**RobotState 子结构**:
```python
class RobotState:
    root_state: Tensor              # 根链接状态 [pos(3), quat(4)]
    body_state: Tensor               # 主体状态 [N, 7] (位置+旋转）
    joint_pos: Tensor               # 关节位置 [N_dof]
    joint_vel: Tensor               # 关节速度 [N_dof]
    joint_acc: Tensor               # 关节加速度 [N_dof]
```

**ObjectState 子结构**:
```python
class ObjectState:
    root_state: Tensor              # 根链接状态 [pos(3), quat(4)]
    joint_pos: Tensor               # 关节位置 [N_dof] (如有铰链）
```

---

#### Action（动作数据）

**定义**: 控制动作的数据结构

**结构**:
```python
# 动作格式（字典列表）
actions: list[dict] = [
    {
        "robot_name": {
            "dof_pos_target": {
                "joint1": 0.1,
                "joint2": 0.5,
                ...
            }
        }
    },
    ...
]

# 或张量格式（向量化）
actions: Tensor of shape (num_envs, num_dof)
```

---

#### Info（信息数据）

**定义**: 环境返回的额外信息

**常用字段**:
```python
info = {
    "success": Tensor,        # 是否成功（每个环境）
    "episode_info": dict,    # 回合信息
    "metric": dict,          # 任务特定指标
    "timestamp": float,       # 时间戳
}
```

---

## 五、学习层概念

### 5.1 IL策略（模仿学习）

#### DiffusionPolicy（扩散策略）

**定义**: 基于扩散模型的策略

**组件**:
```python
class DiffusionPolicy:
    # 策略网络
    model: nn.Module

    # 配置
    n_obs_steps: int = 2        # 观测历史长度
    n_action_steps: int = 8     # 动作块长度
    action_dim: int             # 动作维度
    obs_dim: int               # 观测维度

    # EMA（指数移动平均）
    ema_model: nn.Module | None
    use_ema: bool = False
```

**执行流程**:
```
观测历史 → 去噪过程 → 动作块 → 时序聚合 → 执行
```

**时序聚合**:
```
动作块 [T, B, N]
    ↓
历史动作 [1, 2, ...]
    ↓
加权平均（指数权重）
    ↓
当前动作
```

---

#### ACTPolicy（Action Chunking Transformer）

**定义**: 基于Transformer的动作分块策略

**组件**:
```python
class ACTPolicy:
    # Transformer编码器
    encoder: nn.Module

    # Transformer解码器
    decoder: nn.Module

    # 配置
    chunk_size: int = 100       # 动作块大小
    n_action_steps: int = 8
```

---

### 5.2 VLA策略（视觉-语言-动作）

#### PiPolicyRunner（Pi策略运行器）

**定义**: 物理智能π模型的评测运行器

**架构**:
```
┌──────────────┐
│  VLA Server   │ (WebSocket)
└──────┬───────┘
       │
       v (inference request)
       ↓
┌──────────────┐
│  VLA Client   │ (RoboVerse)
└──────────────┘
```

**数据流**:
```python
# 客户端发送
policy_obs = {
    "observation/image": compressed_image,      # 224x224 RGB
    "observation/wrist_image": fake_wrist,        # 占位
    "observation/state": joint_positions,        # 9维关节
    "prompt": "Pick up the red cube"           # 任务描述
}

# 服务端返回
response = {
    "actions": np.array([              # 动作块 [chunk, N, 7]
        [dx1, dy1, dz1, drx1, dry1, drz1, gripper1],  # 9D
        [dx2, dy2, dz2, drx2, dry2, drz2, gripper2],
        ...
    ])
}

# 解码为关节命令
dof_pos_target = {
    "joint1": arm_target1,
    "joint2": arm_target2,
    ...
    "gripper": gripper_width          # 2维夹爪
}
```

**特点**:
- **WebSocket通信**: 异步请求-响应
- **动作缓存**: 一次请求多个动作
- **夹爪二值化**: 指令阈值处理

---

#### SmolVLAPolicy（SmolVLA策略）

**定义**: 轻量级VLA模型（Hugging Face）

**架构**:
```
输入: [Image, State, Text]
  ↓
Vision-Language Encoder (ViT + Text Encoder)
  ↓
Action Head (MLP)
  ↓
输出: [dx, dy, dz, drx, dry, drz, gripper]
```

**LeRobot格式**:
```python
batch = {
    "observation.image": tensor[C, H, W, 3],
    "observation.state": tensor[N_dof],
    "observation.language.tokens": token_ids,
    "observation.language.attention_mask": mask
}

action = model.select_action(batch)  # [7] delta
```

**IK集成**:
```
VLA输出 (EE Delta) + 当前EE姿态 → IK求解 → 关节位置
```

---

### 5.3 RL策略（强化学习）

#### Actor-Critic架构

**定义**: 标准的Actor-Critic RL架构

**Actor网络**:
```python
class Actor(nn.Module):
    # 状态编码器
    state_encoder: nn.Sequential
        # MLP: [obs_dim → 128 → 128 → action_dim]

    # 输出层
    action_head: nn.Module
        # Tanh激活 [-1, 1] 或Sigmoid [0, 1]
```

**Critic网络**:
```python
class Critic(nn.Module):
    # 状态-动作编码器
    state_action_encoder: nn.Sequential
        # MLP: [obs_dim + action_dim → 256 → 128 → 1]

    # Q值输出
    q_head: nn.Module
        # 输出标量Q值
```

**训练组件**:
- **经验回放** (Experience Replay Buffer)
- **目标网络** (Target Network)
- **软更新** (Soft Target Update)
- **归一化器** (Observation Normalizer)

---

### 5.4 归一化器

#### EmpiricalNormalization（经验归一化）

**定义**: 运行时统计的归一化

**组件**:
```python
class EmpiricalNormalization:
    # 统计量
    mean: Tensor        # 运行均值 [obs_dim]
    var: Tensor         # 运行方差 [obs_dim]
    count: int         # 样本计数
    eps: float = 1e-8   # 数值稳定

    # 方法
    def update(self, obs: Tensor) -> Tensor:
        """更新统计并归一化观测"""
        # 归一化: (obs - mean) / sqrt(var + eps)

    def normalize(self, obs: Tensor) -> Tensor:
        """归一化观测（不更新统计）"""
        return (obs - self.mean) / torch.sqrt(self.var + self.eps)
```

**使用场景**:
- RL训练（运行时归一化）
- RL评测（使用训练时的统计）
- IL评测（通常不使用归一化）

---

## 六、评测层概念

### 6.1 评测运行器

#### BaseEvalRunner（评测运行器基类）

**定义**: 所有评测策略的基类

**核心方法**:
```python
class BaseEvalRunner:
    # 策略配置
    policy_cfg: BasePolicyCfg
        obs_config: ObsCfg            # 观测配置
        action_config: ActionCfg        # 动作配置

    # 状态
    num_envs: int
    device: str
    step: int

    # IK求解器（如需要）
    robot_ik: IKSolver | None
    curobo_n_dof: int | None

    # 动作缓存
    action_cache: list

    # 初始化
    def _init_policy(self, **kwargs) -> None:
        """初始化策略（子类实现）"""

    def __post_init__(self) -> None:
        """后处理：设置IK、时序聚合等"""
```

**子类类型**:
- `DefaultEvalRunner` - 扩散策略评测
- `PiPolicyRunner` - Pi策略评测
- `SmolVLARunner` - SmolVLA评测

---

#### DefaultEvalRunner（默认评测运行器）

**职责**: 支持扩散策略的评测

**处理流程**:
```python
def process_obs(self, obs):
    """处理观测"""
    # 1. 图像处理（归一化或原始）
    if norm_image:
        obs_dict["head_cam"] = obs["rgb"].permute(0, 3, 1, 2) / 255.0
    else:
        obs_dict["head_cam"] = obs["rgb"]

    # 2. 状态提取
    if obs_type == "joint_pos":
        obs_dict["agent_pos"] = obs["joint_qpos"]
    elif obs_type == "ee":
        # 末端执行器状态（相对坐标）
        obs_dict["agent_pos"] = ee_state_local

def predict_action(self, obs):
    """预测动作块"""
    action_chunk = self.policy.predict_action(obs)  # [T, B, N, A]
    return action_chunk

def get_action(self, obs):
    """获取单个可执行动作"""
    if len(action_cache) > 0:
        return action_cache.pop(0)
    else:
        processed_obs = self.process_obs(obs)
        action_chunk = self.predict_action(processed_obs)

        if temporal_agg:
            curr_action = get_temporal_agg_action(action_chunk)
        else:
            self.action_cache = process_action(action_chunk, obs)
            return action_cache.pop(0)
```

**时序聚合**:
```python
def get_temporal_agg_action(self, action_chunk):
    """
    历史动作 [1, 2, ..., T]
                ↓
    指数加权平均 (e^{-k * i})
                ↓
    当前动作
    """
    weights = torch.exp(-self.k * time_indices)
    weights = weights / weights.sum()

    weighted_actions = action_chunk * weights.unsqueeze(-1)
    curr_action = weighted_actions.sum(dim=0)

    return curr_action
```

---

### 6.2 观测与轨迹保存

#### ObsSaver（观测保存器）

**定义**: 自动保存视频和观测数据

**组件**:
```python
class ObsSaver:
    video_path: str          # 视频输出路径
    frames: list[np.ndarray]  # 视频帧
    fps: int = 30             # 帧率

    def add(self, obs):
        """添加观测帧"""

    def save(self):
        """保存为MP4视频"""
        # 使用imageio编码保存
```

**输出格式**:
```
output_dir/
├── episode_001.mp4
├── episode_002.mp4
├── episode_003.mp4
└── ...
```

---

#### TrajectorySaver（轨迹保存器）

**定义**: 保存完整轨迹用于重放

**v2格式**:
```python
trajs = {
    robot_name: [
        {
            "init_state": {
                "object1": {
                    "pos": [x, y, z],
                    "rot": [w, x, y, z],
                    "dof_pos": {"joint1": q1, ...}
                },
                ...
            },
            "actions": [
                {"dof_pos_target": {...}},
                {"dof_pos_target": {...}},
                ...
            ],
            "states": [  # 可选，完整状态
                {
                    "object1": {"pos": [...], "rot": [...]},
                    ...
                },
                ...
            ]
        }
    ]
}
```

**保存选项**:
- `save_states=True`: 保存完整状态（精确重放）
- `save_states=False`: 仅保存动作（节省空间）

---

## 七、数据流概念

### 7.1 配置数据流

```
命令行参数
    ↓
argparse / tyro
    ↓
ScenarioCfg.update(**kwargs)
    ↓
__post_init__()
    ↓
解析字符串配置 → 下载外部资产
```

**示例**:
```bash
# 命令行
python evaluate.py \
    --task pick_butter \
    --robot franka \
    --sim mujoco \
    --level 2

# 配置链
ScenarioCfg(
    robots=["franka"],           # 字符串 → RobotCfg对象
    simulator="mujoco"           # 字符串 → SimType枚举
    level=2                      # DRConfig
)
```

---

### 7.2 评测数据流

```
Policy Runner
    ↓
process_obs(obs)
    ↓
predict_action(processed_obs)
    ↓
get_action() 或 infer_action()
    ↓
{robot_name: {"dof_pos_target": {...}}}
    ↓
env.step(actions)
    ↓
obs, reward, terminated, truncated, info
```

**数据转换**:
```
Obs (Tensor结构)
    ↓
提取
    ├─ robots[name].joint_pos    → policy输入
    ├─ cameras[name].rgb          → policy输入
    └─ objects[name].root_state   → success判断

Action (字典列表)
    ↓
构造
    └─ [{robot_name: {"dof_pos_target": dict}}]
```

---

### 7.3 域随机化数据流

```
DomainRandomizationManager
    ↓
apply_randomization()
    ├─ scene_randomizer.randomize_scene()
    ├─ material_randomizer.randomize_material()
    └─ light_randomizer.randomize_light()
    ↓
更新 handler 中的物体/灯光属性
    ↓
update_positions_to_table()
    ↓
update_camera_look_at()
    ↓
重置环境
```

**随机化组件**:
```
SceneRandomizer
    ├─ GeometryRandomizer (几何)
    ├─ TextureRandomizer (纹理)
    └─ MaterialRandomizer (材质)

LightRandomizer
    ├─ ColorRandomizer (颜色)
    ├─ IntensityRandomizer (强度)
    └─ PositionRandomizer (位置)

CameraRandomizer
    ├─ PositionRandomizer (位置)
    └─ LookAtRandomizer (注视)
```

---

## 八、核心关系图谱

### 8.1 组合关系（Composition）

```
ScenarioCfg (组合根)
├─ [N] RobotCfg          # 1对多关系
├─ [M] BaseObjCfg         # 1对多关系
├─ [K] BaseLightCfg       # 1对多关系
├─ [L] BaseCameraCfg       # 1对多关系
├─ GroundCfg                # 1对1关系
└─ RenderCfg                # 1对1关系
```

---

### 8.2 继承关系（Inheritance）

```
配置继承树
├─ BaseObjCfg
│   ├─ PrimitiveCubeCfg
│   ├─ PrimitiveCylinderCfg
│   ├─ MeshObjCfg
│   └─ ...
├─ BaseLightCfg
│   ├─ DistantLightCfg
│   ├─ SphereLightCfg
│   ├─ DiskLightCfg
│   └─ ...
└─ BaseCameraCfg
    └─ (目前无子类)

任务继承树
├─ BaseTaskEnv
│   ├─ ManiskillBaseTask
│   │   ├─ PushCubeCfg
│   │   ├─ PickCubeCfg
│   │   └─ ...
│   ├─ Libero90BaseTask
│   │   ├─ LiberoKitchenScene1Task
│   │   └─ ...
│   └─ LeggedRobotTask
│       ├─ WalkTask
│       └─ ...

评测继承树
├─ BaseEvalRunner
│   ├─ DefaultEvalRunner
│   ├─ PiPolicyRunner
│   └─ SmolVLARunner
```

---

### 8.3 依赖关系（Dependency）

#### 环境 → 模拟器

```
ScenarioCfg
    ↓
get_sim_handler_class(SimType)
    ↓
BaseSimHandler.__init__(scenario, extra_spec)
    ↓
handler.launch()
```

**依赖规则**:
- `scenario` 必须在 `handler` 创建前完全配置
- `extra_spec` 提供模拟器特定参数
- Handler 必须实现所有 `BaseSimHandler` 方法

---

#### 任务 → 环境

```
get_task_class(task_name)
    ↓
task_cls = TaskClass  # 如 PushCubeCfg
    ↓
task_cls.scenario  # 类属性或构造参数
    ↓
BaseTaskEnv(task_scenario, device)
```

**数据流**:
```
任务配置 → 任务类 → 任务实例 → Handler → 状态/动作
```

---

#### 策略 → 环境

```
PolicyRunner
    ↓
process_obs(obs)
    ↓
predict_action(obs)
    ↓
get_action(obs)
    ↓
action_dict = {robot_name: {"dof_pos_target": dict}}
    ↓
env.step(action_dict)
```

**动作格式匹配**:
- IL策略: 输出关节位置
- VLA策略: 输出末端增量 + IK求解
- RL策略: 输出关节速度或位置

---

#### 检测器 → 任务

```
BaseQueryType (检测器基类)
    ├─ Relative2DSphereDetector
    ├─ ContactForceQuery
    └─ ...
    ↓
check(handler.get_states())
    ↓
DetectedChecker(detector, obj_name)
    ↓
在 _terminated() 中使用
```

**检测流程**:
```
env_states → detector.check() → bool (success/failed)
```

---

### 8.4 生命周期关系（Lifecycle）

#### 环境生命周期

```
创建阶段
    ↓
__init__(scenario, device)
    ↓
_prepare_callbacks()
    ↓
_get_initial_states()
    ↓
handler.launch()
    ↓
[就绪]

运行阶段
    ↓
for step in range(max_steps):
    pre_physics_step_callback()
    ↓
    set_dof_targets(actions)
    ↓
    handler.step()
    ↓
    post_physics_step_callback()
    ↓
    obs, reward, terminated, time_out = check()
    ↓
    if terminated or time_out:
        break

重置阶段
    ↓
reset()
    ↓
[新回合]

销毁阶段
    ↓
close()
    ↓
_cleanup()
    ↓
[资源释放]
```

---

#### 评测生命周期

```
初始化
    ↓
eval_runner._init_policy(**kwargs)
    ↓
randomization_manager.apply_randomization()
    ↓
env.reset()
    ↓
eval_runner.reset()

评测循环
    ↓
for ep in range(num_episodes):
    for step in range(max_steps):
        obs, reward, terminated, truncated = env.step(actions)
        ↓
        obs_saver.add(obs)
        ↓
        if terminated or truncated:
            break

统计输出
    ↓
compute_mean_std(episode_returns)
    ↓
compute_success_rate(episode_successes)
    ↓
save_json_report()
    ↓
save_video()

结束
    ↓
env.close()
    ↓
eval_runner.close()
```

---

### 8.5 数据转换关系（Data Transformation）

#### 观测转换

```
Obs (原始)
    ↓
ObsSaver.add(obs)
    ↓
np.ndarray (视频帧)
    ↓
ObsSaver.save()
    ↓
MP4文件 (H.264编码)
```

#### 动作转换

```
Policy Output
    ↓
[IL] action_chunk [T, B, N, A]
    ↓
时序聚合 (可选)
    ↓
单个动作 [B, N, A] 或 [N_dof]
    ↓
[VLA] IK求解
    ↓
关节位置 [N_dof]
    ↓
dof_pos_target: dict
```

#### 状态转换

```
Handler States
    ↓
get_states(mode="tensor")
    ↓
TensorStates
    ├─ robots[name].joint_pos
    ├─ objects[name].root_state
    └─ images[name].rgb
    ↓
提取到字典或嵌套结构
```

---

## 九、核心概念表

### 9.1 核心实体

| 概念 | 类型 | 定义 | 职责 |
|------|------|------|--------|
| **ScenarioCfg** | 配置类 | 场景完整配置 |
| **RobotCfg** | 配置类 | 机器人配置 |
| **BaseObjCfg** | 配置类 | 物体配置 |
| **BaseLightCfg** | 配置类 | 灯光配置 |
| **BaseCameraCfg** | 配置类 | 相机配置 |
| **BaseTaskEnv** | 任务类 | 任务基类 |
| **BaseSimHandler** | 处理器 | 模拟器处理器接口 |
| **BaseEvalRunner** | 评测类 | 评测运行器基类 |
| **DRConfig** | 配置类 | 域随机化配置 |
| **DomainRandomizationManager** | 管理器 | 域随机化管理器 |
| **ObsSaver** | 工具类 | 观测/视频保存 |
| **DiffusionPolicy** | 模型类 | 扩散策略模型 |
| **Actor-Critic** | 模型类 | RL策略网络 |

---

### 9.2 核心数据结构

| 数据结构 | 形状 | 用途 |
|----------|------|------|
| **Obs** | 结构化 | 环境观测 |
| **Action** | dict或Tensor | 控制动作 |
| **Reward** | Tensor[N] | 奖励信号 |
| **Termination** | Tensor[N, bool] | 终止标志 |
| **Info** | dict | 额外信息 |
| **TensorStates** | 结构化 | 完整状态 |

---

### 9.3 核心方法

| 方法 | 所在类 | 功能 | 频率 |
|------|----------|------|--------|
| **reset()** | BaseTaskEnv | 重置环境 | 回合开始 |
| **step()** | BaseTaskEnv | 步进环境 | 每帧 |
| **render()** | BaseTaskEnv | 渲染场景 | 可选 |
| **predict_action()** | BaseEvalRunner | 预测动作 | 每步 |
| **get_action()** | BaseEvalRunner | 获取动作 | 每步 |
| **check()** | BaseQueryType | 成功检测 | 每步 |
| **normalize()** | EmpiricalNormalization | 归一化 | 训练/评测 |
| **launch()** | BaseSimHandler | 启动模拟器 | 一次 |

---

## 十、概念关系总结

### 10.1 分层架构

```
┌─────────────────────────────────────────────────────┐
│              用户层               │
│  - 命令行参数                         │
│  - JSON/YAML配置                      │
│  - 任务ID                              │
└──────────────────┬──────────────────────────────┘
                 │
                 ↓
┌─────────────────────────────────────────────┐
│            配置层                    │
│  - ScenarioCfg                         │
│  - DRConfig                            │
│  - 任务注册                            │
└──────────────────┬──────────────────────────────┘
                 │
                 ↓
┌─────────────────────────────────────────────┐
│            任务层                     │
│  - BaseTaskEnv (子类实现)          │
│  - Checker/Detector                  │
│  - 奖励/终止逻辑                      │
└──────────────────┬──────────────────────────────┘
                 │
                 ↓
┌─────────────────────────────────────────────┐
│           仿真层                    │
│  - BaseSimHandler (实现类)           │
│  - Mujoco/IsaacGym/Genesis等        │
│  - 状态查询系统                       │
└──────────────────┬──────────────────────────────┘
                 │
                 ↓
┌─────────────────────────────────────────────┐
│          学习层                    │
│  - IL (Diffusion/ACT)              │
│  - VLA (Pi0/SmolVLA)               │
│  - RL (Actor-Critic)                 │
│  - 归一化器                            │
└──────────────────┬──────────────────────────────┘
                 │
                 ↓
┌─────────────────────────────────────────────┐
│          评测层                    │
│  - BaseEvalRunner (子类)           │
│  - ObsSaver                            │
│  - TrajectorySaver                     │
│  - 域随机化管理器                    │
└──────────────────┬──────────────────────────────┘
                 │
                 ↓
┌─────────────────────────────────────────────┐
│          数据层                    │
│  - Demos (演示数据)                   │
│  - Trajectories (轨迹)                │
│  - Checkpoints (模型权重)             │
│  - Videos (评测视频)                   │
│  - JSON Reports (评测报告)           │
└─────────────────────────────────────────────┘
```

---

### 10.2 关键流程

#### 环境创建流程

```
任务ID
  ↓
get_task_class()
  ↓
task_cls.scenario (获取场景配置)
  ↓
ScenarioCfg.update(**kwargs) (更新配置)
  ↓
get_sim_handler_class()
  ↓
handler = HandlerClass(scenario)
  ↓
handler.launch() (启动模拟器)
  ↓
env = GymEnvWrapper(task_cls, scenario)
```

---

#### IL评测流程

```
checkpoint路径
  ↓
加载checkpoint (torch.load)
  ↓
初始化DiffusionPolicy
  ↓
policy.eval() (设置为评测模式)
  ↓
for episode in num_episodes:
    env.reset()
    runner.reset()
    for step in max_steps:
        obs = env.step(action)  # 上一步
        processed_obs = process_obs(obs)  # 处理观测
        action_chunk = policy.predict_action(processed_obs)  # 预测动作块
        if temporal_agg:
            action = temporal_agg(action_chunk)  # 时序聚合
        else:
            action = action_chunk[0]  # 使用块中第一个
        obs, reward, terminated, truncated = env.step(action)
        obs_saver.add(obs)
        if terminated or truncated:
            break
    compute_stats()
```

---

#### VLA评测流程

```
模型路径
  ↓
加载SmolVLAPolicy/PiClient
  ↓
for episode in num_episodes:
    randomization_manager.apply_randomization()
    env.reset()
    runner.reset()
    for step in max_steps:
        obs = env.step(action)
        policy_obs = build_policy_observation(obs)  # 构建观测
        if cache_empty:
            action_chunk = client.infer(policy_obs)  # 请求VLA
            update_cache(action_chunk)
        action = get_cached_action()  # 获取缓存的动作
        if use_ik:
            action = solve_ik(action)  # IK求解
        obs, reward, terminated, truncated = env.step(action)
        obs_saver.add(obs)
        if terminated or truncated:
            break
    compute_stats()
```

---

#### RL评测流程

```
checkpoint路径
  ↓
加载Actor和ObsNormalizer
  ↓
actor.eval(), obs_normalizer.eval()
  ↓
for episode in num_episodes:
    env.reset()
    for step in max_steps:
        norm_obs = obs_normalizer.normalize(obs)  # 归一化
        action = actor(norm_obs)  # 预测动作
        obs, reward, terminated, truncated = env.step(action)
        if terminated or truncated:
            break
    compute_stats()
```

---

## 十一、设计模式总结

### 11.1 模式识别

| 模式 | 应用位置 | 说明 |
|--------|----------|------|
| **策略模式** | BaseEvalRunner | 支持多种策略类型，统一接口 |
| **工厂模式** | get_task_class + TaskRegistry | 动态任务注册和创建 |
| **处理器模式** | BaseSimHandler + 具体实现 | 统一模拟器接口 |
| **观察者模式** | 回调机制 | pre/post physics step回调 |
| **包装器模式** | GymEnvWrapper | 统一Gym API |
| **随机化模式** | DomainRandomizationManager | 统一随机化管理 |

---

### 11.2 核心设计原则

1. **配置优先**: 通过配置类控制所有行为
2. **类型安全**: 使用Pydantic/dataclass配置验证
3. **接口抽象**: 通过基类定义最小接口
4. **模块化**: 每个组件独立可替换
5. **向量化**: 支持多环境并行加速
6. **可扩展**: 通过装饰器注册新任务
7. **跨模拟器**: 抽象Handler支持8+模拟器

---

## 总结

RoboVerse 的核心设计哲学是：

**🎯 统一性**: 单一平台支持多种模拟器和算法
**🔌 灵活性**: 配置驱动的可扩展架构
**⚡ 高效性**: 原生向量化和GPU加速
**🎨 可视化**: 内置视频录制和轨迹保存
**🎲 泛化能力**: 4级域随机化测试
**📊 标准化**: Gymnasium API和统一评测指标
**🧩 模块化**: 清晰的层次和职责分离

这些概念和关系构成了RoboVerse作为统一机器人学习平台的基础架构。

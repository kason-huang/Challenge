# RoboVerse 评测流程详解

> **概述**: RoboVerse 评测系统是一个灵活、可扩展的框架，支持多种模拟器、算法和域随机化级别的机器人操作任务评测。

---

## 目录

- [一、整体架构](#一整体架构)
- [二、配置层](#二配置层)
- [三、环境创建层](#三环境创建层)
- [四、评测执行层](#四评测执行层)
- [五、评测循环流程](#五评测循环流程)
- [六、结果输出层](#六结果输出层)
- [七、命令行接口](#七命令行接口)
- [八、关键特性总结](#八关键特性总结)

---

## 一、整体架构

```
配置层 (Configuration)
    ↓
环境创建层 (Environment Creation)
    ↓
任务定义层 (Task Definition)
    ↓
评测执行层 (Evaluation Execution)
    ↓
结果输出层 (Results & Metrics)
```

### 核心文件结构

```
RoboVerse-main/
├── metasim/
│   ├── scenario/
│   │   └── scenario.py              # ScenarioCfg - 场景配置
│   ├── task/
│   │   ├── base.py                    # BaseTaskEnv - 任务基类
│   │   ├── gym_registration.py         # Gym 接口注册
│   │   └── registry.py                # 任务注册表
│   ├── randomization/
│   │   └── dr_manager.py              # DRConfig - 域随机化
│   └── sim/
│       └── base.py                    # BaseSimHandler - 模拟器基类
├── roboverse_pack/
│   └── tasks/                          # 具体任务定义
│       ├── maniskill/
│       │   ├── push_cube.py             # 示例任务
│       │   └── maniskill_base.py       # Maniskill基类
│       └── ...
└── roboverse_learn/
    ├── il/
    │   └── runners/
    │       └── default_eval_runner.py  # IL评测运行器
    ├── vla/
    │   ├── pi0/
    │   │   └── pi_eval.py            # Pi0评测
    │   └── SmolVLA/
    │       └── smolvla_eval.py        # SmolVLA评测
    └── rl/
        └── fast_td3/
            └── evaluate.py             # FastTD3评测
```

---

## 二、配置层

### 1. 场景配置 (ScenarioCfg)

**核心类**: `ScenarioCfg`
**位置**: `metasim/scenario/scenario.py`

#### 配置项详解

```python
@configclass
class ScenarioCfg:
    """场景配置类 - 定义任务环境的所有组件"""

    # === 资产配置 ===
    scene: SceneCfg | None = None           # 场景配置（桌面、地面等）
    robots: list[RobotCfg] = []           # 机器人列表（可多机器人）
    objects: list[BaseObjCfg] = []        # 场景物体（立方体、工具等）
    lights: list[BaseLightCfg] = []       # 灯光配置
    cameras: list[BaseCameraCfg] = []     # 相机配置
    gs_scene: GSSceneCfg | None = None  # Gaussian Splatting场景
    ground: GroundCfg | None = None       # 地面配置

    # === 运行时配置 ===
    simulator: str = None                     # 模拟器后端
    num_envs: int = 1                        # 并行环境数量
    headless: bool = False                     # 无头模式（不显示GUI）
    env_spacing: float = 1.0                   # 环境间距（向量化时）
    decimation: int = 15                        # 物理解算降频
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)  # 重力加速度

    # === 渲染配置 ===
    render: RenderCfg = RenderCfg()        # 渲染模式、质量等
    sim_params: SimParamCfg = SimParamCfg() # 物理参数（时间步等）

    def __post_init__(self) -> None:
        """后处理：解析字符串配置，下载外部资产"""
        # 自动将字符串解析为配置对象
        # 下载Hugging Face等外部资产
```

#### 配置示例

```python
# MuJoCo + Franka + 相机 + 灯光
scenario = ScenarioCfg(
    # 资产
    robots=[RobotCfg("franka")],
    objects=[PrimitiveCubeCfg(name="cube", size=[0.04, 0.04, 0.04])],
    lights=[
        DiskLightCfg(name="ceiling_main", intensity=12000.0),
        SphereLightCfg(name="corner_light", intensity=5000.0)
    ],
    cameras=[
        PinholeCameraCfg(
            name="camera",
            data_types=["rgb"],
            width=256,
            height=256,
            pos=(1.0, 0.0, 0.75),
            look_at=(0.0, 0.0, 0.0)
        )
    ],

    # 运行时
    simulator="mujoco",
    num_envs=1,
    headless=True,

    # 渲染
    render=RenderCfg(mode="raytracing")
)
```

---

### 2. 任务配置 (Task Configuration)

#### 基类: BaseTaskEnv

**位置**: `metasim/task/base.py`

```python
class BaseTaskEnv:
    """任务环境基类 - 所有任务的父类"""

    max_episode_steps = 100         # 最大步数
    traj_filepath = None            # 演示轨迹路径（用于重放）

    def __init__(self, scenario, device=None):
        """初始化任务环境"""
        self.scenario = scenario
        self.handler = get_sim_handler(scenario)  # 模拟器处理器
        self._episode_steps = torch.zeros(num_envs)
        self._prepare_callbacks()

    # === 必须实现的方法 ===
    def _observation(self, env_states: Obs) -> Obs:
        """获取环境观测（机器人状态、相机图像等）"""
        raise NotImplementedError

    def _reward(self, env_states: Obs) -> Reward:
        """计算奖励信号"""
        return torch.zeros(num_envs, dtype=torch.float32)

    def _terminated(self, env_states: Obs) -> Termination:
        """判断是否满足终止条件（成功/失败）"""
        return torch.zeros(num_envs, dtype=torch.bool)

    def _time_out(self, env_states) -> torch.Tensor:
        """判断是否超时"""
        return self._episode_steps >= self.max_episode_steps

    # === 可选实现的方法 ===
    def _privileged_observation(self, env_states: Obs) -> Obs:
        """特权观测（用于RL训练，不可用于实际评测）"""
        return env_states

    def _observation_space(self) -> gym.Space:
        """定义观测空间"""
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(0,))

    def _action_space(self) -> gym.Space:
        """定义动作空间"""
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(0,))

    # === 回调机制 ===
    pre_physics_step_callback: list[Callable] = []   # 物理步进前
    post_physics_step_callback: list[Callable] = []  # 物理步进后
    reset_callback: list[Callable] = []              # 重置时
    close_callback: list[Callable] = []              # 关闭时
```

#### 任务注册

```python
from metasim.task.registry import register_task

@register_task("maniskill.push_cube", "push_cube")
class PushCubeCfg(ManiskillBaseTask):
    """推方块任务"""
    episode_length = 500

    scenario = ScenarioCfg(
        objects=[
            PrimitiveCubeCfg(name="cube", size=[0.04, 0.04, 0.04]),
            PrimitiveCylinderCfg(
                name="goal_region",
                radius=0.15,
                collision_enabled=False  # 仅用于可视化
            )
        ]
    )

    # 成功检测器
    checker = DetectedChecker(
        detector=Relative2DSphereDetector(
            base_obj_name="goal_region",
            relative_pos=(0.0, 0.0, 0.0),
            radius=0.15,
            axis=(0, 1)
        ),
        obj_name="cube"
    )

    traj_filepath = "roboverse_data/trajs/maniskill/push_cube/v2"
```

---

### 3. 域随机化配置 (Domain Randomization)

#### 核心类: DomainRandomizationManager

**位置**: `metasim/randomization/dr_manager.py`

```python
@dataclass
class DRConfig:
    """域随机化配置"""

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
    - 0: 手动几何（默认）
    - 1: USD Table + 手动环境
    - 2: USD Scene + USD Table
    - 3: Full USD (Scene + Table + Desktop objects)
    """

    randomization_seed: int | None = None
    """随机化种子（None则使用随机种子）"""


class DomainRandomizationManager:
    """统一的域随机化管理器"""

    def __init__(
        self,
        config: DRConfig,
        scenario: ScenarioCfg,
        handler: BaseSimHandler,
        init_states: list | None = None,
        render_cfg = None
    ):
        """初始化DR Manager"""
        self.config = config
        self.scenario = scenario
        self.handler = handler
        self.init_states = init_states

    # === 随机化方法 ===
    def apply_randomization(self, demo_idx: int, is_initial: bool):
        """应用场景、材质随机化"""

    def update_positions_to_table(self, demo_idx: int, env_id: int):
        """更新物体位置到桌面（USD模式）"""

    def update_camera_look_at(self, env_id: int):
        """更新相机看向位置"""

    def apply_camera_randomization(self):
        """应用相机位置随机化"""
```

#### 随机化级别详解

| 级别 | 场景 | 材质 | 灯光 | 相机 | 说明 |
|------|------|------|------|------|------|
| 0 | ❌ | ❌ | ❌ | ❌ | 无随机化（标准评测） |
| 1 | ✅ | ✅ | ❌ | ❌ | 场景+材质变化 |
| 2 | ✅ | ✅ | ✅ | ❌ | Level 1 + 灯光变化 |
| 3 | ✅ | ✅ | ✅ | ✅ | Level 2 + 相机视角变化 |

---

## 三、环境创建层

### 1. Gym 接口注册

**核心文件**: `metasim/task/gym_registration.py`

#### 单环境创建

```python
import gymnasium as gym

# 创建单个环境
env = gym.make(
    f"RoboVerse/{task_name}",    # 注册的任务ID
    robots=["franka"],                # 机器人配置
    simulator="mujoco",               # 模拟器后端
    headless=True,                    # 无头模式
    cameras=[camera_cfg],              # 相机配置
    lights=[lights],                   # 灯光配置
    device="cuda"                     # 设备
)

# 使用
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(action)
env.render()
env.close()
```

#### 向量环境创建

```python
from gymnasium import make_vec

# 创建向量环境（多环境并行）
env = make_vec(
    f"RoboVerse/{task_name}",
    num_envs=8,                    # 并行环境数
    robots=["franka"],
    simulator="isaacgym",            # isaacgym原生向量化
    headless=True,
    device="cuda"
)

# 使用（向量化操作）
obs, info = env.reset()
obs, rewards, terminateds, truncateds, infos = env.step([action1, action2, ..., action8])
# obs, rewards 等都是形状为 (8, ...) 的张量
env.close()
```

### 2. 环境包装器

#### GymEnvWrapper (单环境)

```python
class GymEnvWrapper(gym.Env):
    """Gymnasium兼容的单环境包装器"""

    def __init__(
        self,
        task_name: str,
        device: str | None = None,
        **scenario_kwargs: Any
    ):
        # 强制单环境
        scenario_kwargs["num_envs"] = 1

        self.task_cls = get_task_class(task_name)
        self.scenario = self.task_cls.scenario.update(**scenario_kwargs)
        self.task_env = self.task_cls(self.scenario, device)

        # Gym API
        self.action_space = self.task_env.action_space
        self.observation_space = self.task_env.observation_space
        self.metadata = {"autoreset_mode": "same-step"}

    def reset(self, seed=None, options=None):
        """重置环境"""
        super().reset(seed=seed)
        obs, info = self.task_env.reset()
        return obs, info

    def step(self, action):
        """步进环境"""
        obs, reward, terminated, truncated, info = self.task_env.step(action)
        return obs, reward, terminated, truncated, info

    def render(self):
        """渲染环境"""
        img = self.task_env.render()
        return np.array(img, copy=True) if img is not None else None
```

#### GymVectorEnvAdapter (向量环境)

```python
class GymVectorEnvAdapter(VectorEnv):
    """向量环境适配器 - 利用后端原生向量化"""

    def __init__(
        self,
        task_name: str,
        device: str | None = None,
        **scenario_kwargs: Any
    ):
        self.task_cls = get_task_class(task_name)
        self.scenario = self.task_cls.scenario.update(**scenario_kwargs)
        self.task_env = self.task_cls(self.scenario, device)

        # VectorEnv API
        self.num_envs = self.task_env.num_envs
        self.device = self.task_env.device

        # 后端原生向量化（单进程，多环境）
        # vs Gymnasium的异步向量化（多进程）
```

### 3. 模拟器处理器

#### 支持的模拟器

| 模拟器 | 向量化 | 性能 | 适用场景 |
|---------|--------|------|----------|
| **mujoco** | ❌ (单进程) | 中等 | 快速原型 |
| **mjx** | ✅ (JAX) | 极快 | 大规模RL训练 |
| **isaacgym** | ✅ (原生) | 极快 | GPU加速的RL |
| **isaaclab** | ✅ (原生) | 高 | 最新NVIDIA栈 |
| **genesis** | ✅ | 极快 | 现代物理引擎 |
| **pybullet** | ✅ | 中等 | 兼容性好 |
| **sapien2** | ✅ | 高 | 渲染质量优先 |
| **sapien3** | ✅ | 高 | 最新版本 |

#### 模拟器初始化流程

```python
# 1. 获取模拟器处理器类
handler_class = get_sim_handler_class(SimType("mujoco"))

# 2. 创建处理器
handler: BaseSimHandler = handler_class(scenario, extra_spec)

# 3. 启动模拟器
handler.launch()

# 4. 访问底层接口
handler.set_dof_targets(actions)          # 设置关节目标
states = handler.get_states(mode="tensor")  # 获取状态
handler.step()                             # 物理步进
```

---

## 四、评测执行层

### 1. IL 评测 (Imitation Learning)

#### DefaultEvalRunner

**位置**: `roboverse_learn/il/runners/default_eval_runner.py`

```python
class DefaultEvalRunner(BaseEvalRunner):
    """默认评测运行器 - 支持扩散策略"""

    def _init_policy(self, default_runner, **kwargs):
        """初始化策略"""
        # 加载checkpoint
        payload = torch.load(checkpoint_path, pickle_module=dill)
        cfg = payload["cfg"]

        # 获取策略（支持EMA）
        policy = default_runner.model
        if cfg.train_config.training_params.use_ema:
            policy = default_runner.ema_model

        policy.to(device)
        policy.eval()
        self.policy = policy

        # 配置动作块
        self.obs = deque(maxlen=cfg.n_obs_steps + 1)

    def process_obs(self, obs):
        """处理观测以匹配策略输入"""

        obs_dict = {}

        # 图像处理
        if self.policy_cfg.obs_config.norm_image:
            obs_dict["head_cam"] = obs["rgb"].permute(0, 3, 1, 2) / 255.0
        else:
            obs_dict["head_cam"] = obs["rgb"]

        # 状态处理
        if self.policy_cfg.obs_config.obs_type == "joint_pos":
            obs_dict["agent_pos"] = obs["joint_qpos"]
        elif self.policy_cfg.obs_config.obs_type == "ee":
            # 末端执行器状态（相对于机器人基座）
            ee_pos_local = quaternion_apply(
                quaternion_invert(robot_quat), ee_pos - robot_pos
            )
            obs_dict["agent_pos"] = torch.cat([
                ee_pos_local,
                ee_quat_local,
                gripper_state
            ], dim=1)

        return obs_dict

    def predict_action(self, obs):
        """预测动作块"""
        with torch.no_grad():
            action_chunk = self.policy.predict_action(obs)["action"]
            action_chunk = action_chunk.transpose(0, 1)  # (chunk, env, action_dim)
        return action_chunk

    def get_action(self, obs):
        """获取单个可执行动作（支持动作缓存）"""

        # 检查缓存
        if len(self.action_cache) > 0:
            curr_action = self.action_cache.pop(0)
        else:
            # 预测新块
            processed_obs = self.process_obs(obs)
            action_chunk = self.predict_action(processed_obs)

            if self.policy_cfg.action_config.temporal_agg:
                # 时序聚合
                curr_action = self.get_temporal_agg_action(action_chunk)
            else:
                # 直接使用块第一个动作
                self.action_cache = process_action(action_chunk, obs)
                curr_action = self.action_cache.pop(0)

        self.step += 1

        # 转换为动作字典
        actions = self.action_to_dict(curr_action)
        return actions
```

---

### 2. VLA 评测 (Vision-Language-Action)

#### Pi0 评测 (Physical Intelligence)

**位置**: `roboverse_learn/vla/pi0/pi_eval.py`

```python
class PiPolicyRunner:
    """Pi策略运行器 - WebSocket客户端"""

    def __init__(
        self,
        env,
        scenario,
        policy_host: str,
        policy_port: int,
        image_size: int = 224,
        gripper_threshold: float = 0.02,
        device: str = "cuda",
        actions_per_call: int | None = None
    ):
        # WebSocket客户端（连接到策略服务器）
        self.client = websocket_client_policy.WebsocketClientPolicy(
            host=policy_host,
            port=policy_port
        )

        # IK求解器
        self.ik_solver = setup_ik_solver(scenario.robots[0], "pyroki")

    def _compress_image(self, obs) -> np.ndarray:
        """压缩图像到指定尺寸"""
        rgb = obs.cameras["camera"].rgb
        resized = image_tools.resize_with_pad(rgb, self.image_size, self.image_size)
        return image_tools.convert_to_uint8(resized)

    def _build_policy_observation(self, obs) -> Dict[str, Any]:
        """构建策略观测"""
        return {
            "observation/image": self._compress_image(obs),
            "observation/wrist_image": fake_wrist,  # 无腕部相机时用假数据
            "observation/state": self._extract_robot_state(obs),
            "prompt": self._get_prompt()  # 任务描述
        }

    def _decode_single_action(self, action: np.ndarray) -> list[dict]:
        """解码单个动作为关节目标"""
        # 夹爪二值化
        finger_vals = action[:2]
        gripper_binary = 1.0 if finger_vals.mean() > self.gripper_threshold else 0.0
        gripper_widths = process_gripper_command(gripper_binary, robot_cfg, device)

        # 手臂关节目标
        arm_target = action[2:]

        # 组合
        joint_target = torch.cat([arm_target, gripper_widths])

        # 转换为字典
        dof_pos_target = {
            joint_name: float(joint_target[i])
            for i, joint_name in enumerate(joint_names)
        }

        return [{robot_name: {"dof_pos_target": dof_pos_target}}]

    def infer_action(self, obs) -> list[dict]:
        """推理动作（支持缓存）"""

        # 检查缓存
        if self.cached_actions is None or self.cache_remaining <= 0:
            # 请求新块
            policy_obs = self._build_policy_observation(obs)
            response = self.client.infer(policy_obs)
            chunk = np.asarray(response["actions"])

            self.cached_actions = chunk
            self.cache_index = 0
            self.cache_remaining = len(chunk) if actions_per_call is None else min(actions_per_call, len(chunk))

        # 返回缓存动作
        action_vec = self.cached_actions[self.cache_index]
        self.cache_index += 1
        self.cache_remaining -= 1

        return self._decode_single_action(action_vec)
```

#### SmolVLA 评测 (Hugging Face)

**位置**: `roboverse_learn/vla/SmolVLA/smolvla_eval.py`

```python
class SmolVLARunner:
    """SmolVLA运行器 - LeRobot格式"""

    def __init__(self, env, scenario, checkpoint_path: str, device: str):
        # 加载模型
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        self.model = SmolVLAPolicy.from_pretrained(checkpoint_path).eval().to(device)

        # IK求解器
        self.ik_solver = setup_ik_solver(scenario.robots[0], "pyroki")

    def predict_action(self, observation=None):
        """预测动作"""
        # 提取图像
        rgb_data = obs.cameras["camera"].rgb
        image = Image.fromarray(rgb_data.detach().cpu().numpy())

        # 获取任务描述
        instruction = getattr(env.task_env, "task_desc", self.task_name)

        # 准备批次
        batch = {
            "observation.image": image_tensor,
            "observation.state": state_tensor,
            "observation.language.tokens": tokenized["input_ids"],
            "observation.language.attention_mask": tokenized["attention_mask"]
        }

        # 预测
        action = self.model.select_action(batch).squeeze(0).cpu().numpy()
        return action

    def ee_control_actions(self, obs):
        """转换为末端执行器控制并求解IK"""

        # 1. 获取VLA动作（增量）
        action = self.predict_action(obs)
        ee_pos_delta = action[:3]      # [dx, dy, dz]
        ee_rot_delta = action[3:6]     # [drx, dry, drz]
        gripper_open = action[6]          # [0=closed, 1=open]

        # 2. 当前末端状态
        curr_ee_pos_local = quaternion_apply(
            quaternion_invert(robot_quat), ee_pos_world - robot_pos
        )
        curr_ee_quat_local = quaternion_multiply(
            quaternion_invert(robot_quat), ee_quat_world
        )

        # 3. 计算目标姿态
        ee_pos_target = curr_ee_pos_local + ee_pos_delta

        # 旋转增量转四元数
        ee_quat_delta = matrix_to_quaternion(
            euler_angles_to_matrix(ee_rot_delta, "XYZ")
        )
        ee_quat_target = quaternion_multiply(curr_ee_quat_local, ee_quat_delta)

        # 4. 求解IK
        q_solution, ik_succ = self.ik_solver.solve_ik_batch(
            ee_pos_target,
            ee_quat_target,
            curr_robot_q
        )

        # 5. 处理夹爪
        gripper_widths = process_gripper_command(gripper_open, robot_cfg, device)

        # 6. 组合动作
        actions = self.ik_solver.compose_joint_action(
            q_solution,
            gripper_widths,
            current_q=curr_robot_q,
            return_dict=True
        )

        return actions
```

---

### 3. RL 评测 (Reinforcement Learning)

#### FastTD3 评测

**位置**: `roboverse_learn/rl/fast_td3/evaluate.py`

```python
def evaluate(
    env,
    actor,
    obs_normalizer,
    num_episodes: int,
    device: torch.device,
    scenario = None,
    task_name: str = "eval",
    amp_enabled: bool = False,
    render: bool = True,
    video_path: str = None,
    render_each_episode: bool = True,
    save_traj: bool = True,
    save_states: bool = True,
    save_every_n_steps: int = 1,
    traj_dir: str = "eval_trajs"
) -> dict:
    """评测策略"""

    # 1. 设置为评测模式
    actor.eval()
    obs_normalizer.eval()

    # 2. 初始化统计
    episode_returns = []
    episode_lengths = []
    episode_successes = []
    episodes_completed = 0
    current_returns = torch.zeros(num_envs, device=device)
    current_lengths = torch.zeros(num_envs, device=device)

    # 3. 视频保存器
    frames = [] if (render and not render_each_episode) else None
    episode_frames = {} if render_each_episode else None

    # 4. 轨迹保存器
    current_episode_actions = {}
    current_episode_states = {}

    # 5. 重置环境
    obs, info = env.reset()

    # 6. 评测循环
    max_steps = env.max_episode_steps * num_episodes

    for step in range(max_steps):
        # 预测动作
        with torch.no_grad(), autocast(enabled=amp_enabled):
            norm_obs = obs_normalizer(obs)
            actions = actor(norm_obs)

        # 执行动作
        next_obs, rewards, terminated, time_out, infos = env.step(actions.float())

        # 更新统计
        current_returns += rewards
        current_lengths += 1

        # 记录轨迹
        if save_traj and (step % save_every_n_steps == 0):
            handler_states = env.handler.get_states(mode="tensor")
            action_record = {
                "dof_pos_target": {
                    name: float(pos)
                    for name, pos in zip(joint_names, joint_positions)
                }
            }
            current_episode_actions[env_id].append(action_record)

            if save_states:
                current_state = extract_state_dict(env, scenario, env_idx=env_id)
                current_episode_states[env_id].append(current_state)

        # 检查完成
        dones = terminated | time_out
        if dones.any():
            for env_id in range(num_envs):
                if dones[env_id]:
                    episode_returns.append(current_returns[env_id].item())
                    episode_lengths.append(current_lengths[env_id].item())

                    if "success" in infos:
                        episode_successes.append(infos["success"][env_id].item())

                    # 保存视频
                    if render_each_episode:
                        video_path = f"{base_dir}/ep{env_id:02d}_{episodes_completed:02d}.mp4"
                        imageio.mimsave(video_path, episode_frames[env_id], fps=30)

                    # 保存轨迹
                    if save_traj and len(current_episode_actions[env_id]) > 0:
                        episode_data = {
                            "init_state": current_episode_init_state[env_id],
                            "actions": current_episode_actions[env_id],
                            "states": current_episode_states[env_id]
                        }
                        all_episodes[env_id].append(episode_data)

                    episodes_completed += 1
                    current_returns[env_id] = 0
                    current_lengths[env_id] = 0

            # 重置已完成的环境
            obs, info = env.reset()

        obs = next_obs

    # 7. 计算统计
    stats = {
        "mean_return": np.mean(episode_returns),
        "std_return": np.std(episode_returns),
        "mean_length": np.mean(episode_lengths),
        "std_length": np.std(episode_lengths),
        "num_episodes": len(episode_returns),
    }

    if episode_successes:
        stats["success_rate"] = np.mean(episode_successes)

    return stats
```

---

### 4. 域随机化应用

```python
# 初始化DR Manager
randomization_manager = DomainRandomizationManager(
    config=DRConfig(
        level=2,               # 场景+材质+灯光
        scene_mode=1,            # USD Table
        randomization_seed=42       # 可复现
    ),
    scenario=env.scenario,
    handler=env.task_env.handler,
    init_states=init_states,
    render_cfg=RenderCfg(mode="raytracing")
)

# 评测循环中应用
for ep in range(num_episodes):
    # 1. 应用随机化（仅首回合或特定索引）
    if ep == 0:
        randomization_manager.apply_randomization(
            demo_idx=ep % len(init_states),
            is_initial=True
        )

    # 2. 更新物体位置（USD模式）
    randomization_manager.update_positions_to_table(
        demo_idx=ep % len(init_states),
        env_id=0
    )

    # 3. 更新相机朝向
    randomization_manager.update_camera_look_at(env_id=0)

    # 4. 应用相机随机化（Level 3）
    if level >= 3:
        randomization_manager.apply_camera_randomization()

    # 5. 重置环境到初始状态
    obs, info = env.task_env.reset(states=[init_states[ep % len(init_states)]])
```

---

## 五、评测循环流程

### 通用评测循环模式

```python
def evaluate_episode(
    env,
    runner,
    max_steps: int,
    episode_num: int,
    output_dir: str,
    randomization_manager = None,
    demo_idx: int = 0,
    init_states = None
) -> Dict[str, Any]:
    """评测单个回合"""

    # === 阶段1: 应用域随机化 ===
    if randomization_manager is not None:
        randomization_manager.apply_randomization(
            demo_idx=demo_idx,
            is_initial=(episode_num == 1)
        )
        randomization_manager.update_positions_to_table(demo_idx=demo_idx, env_id=0)
        randomization_manager.update_camera_look_at(env_id=0)
        randomization_manager.apply_camera_randomization()

    # === 阶段2: 重置环境 ===
    if randomization_manager is not None and init_states is not None:
        # 使用指定初始状态重置
        obs, info = env.task_env.reset(states=[init_states[demo_idx]])
    else:
        # 默认重置
        obs, info = env.reset()

    # 重置运行器状态
    runner.reset()

    # === 阶段3: 初始化视频保存 ===
    os.makedirs(output_dir, exist_ok=True)
    obs_saver = ObsSaver(video_path=f"{output_dir}/episode_{episode_num:03d}.mp4")
    obs_saver.add(obs)

    # === 阶段4: 统计初始化 ===
    stats = {
        "steps": 0,
        "success": False,
        "total_reward": 0.0,
        "start_time": time.time()
    }

    # === 阶段5: 步进循环 ===
    for step in range(max_steps):
        # 5.1 获取动作
        if hasattr(runner, 'get_action'):
            actions = runner.get_action(obs)          # IL/RL
        elif hasattr(runner, 'infer_action'):
            actions = runner.infer_action(obs)        # VLA
        elif hasattr(runner, 'ee_control_actions'):
            actions = runner.ee_control_actions(obs)  # VLA with IK

        # 5.2 执行动作
        obs, reward, terminated, truncated, info = env.step(actions)

        # 5.3 更新统计
        stats["steps"] += 1
        stats["total_reward"] += float(reward)

        # 5.4 保存视频帧
        obs_saver.add(obs)

        # 5.5 检查完成
        term = terminated.any().item() if hasattr(terminated, 'any') else bool(terminated)
        trunc = truncated.any().item() if hasattr(truncated, 'any') else bool(truncated)

        if term or trunc:
            stats["success"] = True
            break

    # === 阶段6: 保存结果 ===
    obs_saver.save()

    stats["end_time"] = time.time()
    stats["duration"] = stats["end_time"] - stats["start_time"]

    return stats
```

### 多回合评测

```python
# 全局统计
aggregate = {
    "total_episodes": 0,
    "total_successes": 0,
    "total_rewards": [],
    "episode_results": []
}

# 多回合循环
for ep in range(num_episodes):
    print(f"\n{'=' * 50}")
    print(f"Episode {ep + 1}/{num_episodes}")
    print(f"{'=' * 50}")

    demo_idx = ep % len(init_states) if randomization_manager else 0

    # 评测单个回合
    result = evaluate_episode(
        env=env,
        runner=runner,
        max_steps=max_steps,
        episode_num=ep + 1,
        output_dir=output_dir,
        randomization_manager=randomization_manager,
        demo_idx=demo_idx,
        init_states=init_states if randomization_manager else None
    )

    # 更新全局统计
    aggregate["total_episodes"] += 1
    aggregate["episode_results"].append(result)
    aggregate["total_rewards"].append(result["total_reward"])

    if result["success"]:
        aggregate["total_successes"] += 1

    # 实时成功率
    sr = aggregate["total_successes"] / aggregate["total_episodes"]
    print(f"  Steps: {result['steps']}")
    print(f"  Success: {result['success']}")
    print(f"  Reward: {result['total_reward']:.2f}")
    print(f"  Success rate: {sr:.1%}")
```

---

## 六、结果输出层

### 1. 评测指标

#### 指标类型

```python
stats = {
    # === 成功率相关 ===
    "success": bool,                  # 单回合是否成功
    "success_rate": float,              # 总成功率 (0-1)
    "total_successes": int,            # 成功回合数

    # === 奖励相关 ===
    "total_reward": float,            # 单回合总奖励
    "mean_return": float,              # 平均返回值
    "std_return": float,               # 返回值标准差
    "total_rewards": list[float],       # 所有回合奖励列表

    # === 长度相关 ===
    "steps": int,                     # 单回合步数
    "mean_length": float,              # 平均步数
    "std_length": float,               # 步数标准差
    "episode_lengths": list[int],        # 所有回合步数列表

    # === 时间相关 ===
    "start_time": float,               # 开始时间戳
    "end_time": float,                 # 结束时间戳
    "duration": float,                 # 回合时长（秒）
    "total_time": float,                # 总评测时长（秒）
    "total_episodes": int,              # 总回合数

    # === 配置相关 ===
    "episode_results": list[dict],     # 详细回合结果
}
```

### 2. 视频输出

#### 每回合视频

```python
from metasim.utils.obs_utils import ObsSaver

# 创建视频保存器
obs_saver = ObsSaver(video_path=f"{output_dir}/episode_{episode_num:03d}.mp4")

# 添加帧
obs_saver.add(obs)  # 每步调用

# 保存视频
obs_saver.save()  # 自动保存为MP4
```

**输出示例**:
```
smolvla_eval_output/
├── episode_001.mp4
├── episode_002.mp4
├── episode_003.mp4
└── ...
```

#### 合并视频（单环境）

```python
# 收集所有帧
frames = []
for step in range(max_steps):
    obs, reward, terminated, truncated, info = env.step(action)
    frames.append(env.render())

# 保存为单个视频
import imageio.v2 as iio
os.makedirs(os.path.dirname(video_path), exist_ok=True)
iio.mimsave(video_path, frames, fps=30)
```

### 3. JSON 结果报告

```python
import json
import time

# 完整结果
result = {
    "config": vars(args),                         # 命令行参数
    "eval_stats": aggregate,                       # 统计结果
    "timestamp": time.strftime("%Y%m%d_%H%M%S"),  # 时间戳
    "dr_config": {                                   # 域随机化配置
        "level": args.level,
        "scene_mode": args.scene_mode,
        "seed": args.randomization_seed
    }
}

# 保存
os.makedirs(args.output_dir, exist_ok=True)
result_path = os.path.join(
    args.output_dir,
    f"pi_eval_{args.task}_{result['timestamp']}.json"
)
with open(result_path, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2)

print(f"Saved results to {result_path}")
```

**输出示例**:
```json
{
  "config": {
    "task": "pick_butter",
    "robot": "franka",
    "sim": "mujoco",
    "num_episodes": 10,
    "max_steps": 250,
    "level": 2,
    "scene_mode": 1,
    "randomization_seed": 42
  },
  "eval_stats": {
    "total_episodes": 10,
    "total_successes": 8,
    "total_rewards": [23.5, 28.1, 25.3, ...],
    "episode_results": [...]
  },
  "timestamp": "20250125_143022",
  "dr_config": {
    "level": 2,
    "scene_mode": 1,
    "seed": 42
  }
}
```

### 4. 轨迹文件保存 (可选)

#### v2 格式

```python
from metasim.utils.demo_util import save_traj_file

# 轨迹数据结构
trajs = {
    robot_name: [
        {
            "init_state": {
                "object1": {
                    "pos": [x, y, z],
                    "rot": [w, x, y, z],
                    "dof_pos": {"joint1": q1, "joint2": q2, ...}
                },
                "object2": {...},
                ...
            },
            "actions": [
                {"dof_pos_target": {"joint1": q1, "joint2": q2, ...}},
                {"dof_pos_target": {...}},
                ...
            ],
            "states": [  # 可选，完整状态
                {
                    "object1": {"pos": [...], "rot": [...], "dof_pos": {...}},
                    ...
                },
                ...
            ]
        }
    ]
}

# 保存
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"{task_name}_{robot_name}_eval_{timestamp}_v2.pkl"
filepath = os.path.join(traj_dir, filename)
save_traj_file(trajs, filepath)
```

**输出示例**:
```
eval_trajs/
├── pick_butter_franka_eval_20250125_143022_v2.pkl
└── ...
```

---

## 七、命令行接口

### 1. Pi0 评测

```bash
python roboverse_learn/vla/pi0/pi_eval.py \
    --task pick_butter \
    --robot franka \
    --sim mujoco \
    --policy-host localhost \
    --policy-port 8000 \
    --num-episodes 10 \
    --max-steps 250 \
    --solver pyroki \
    --device cuda \
    --image-size 224 \
    --gripper-threshold 0.02 \
    --actions-per-call 5 \
    --level 2 \
    --scene-mode 1 \
    --randomization-seed 42 \
    --output-dir ./pi_eval_output
```

### 2. SmolVLA 评测

```bash
python roboverse_learn/vla/SmolVLA/smolvla_eval.py \
    --model_path ./checkpoints/005000 \
    --task pick_butter \
    --robot franka \
    --sim mujoco \
    --solver pyroki \
    --num-episodes 10 \
    --max-steps 250 \
    --device cuda \
    --level 3 \
    --scene-mode 2 \
    --randomization-seed 42 \
    --output-dir ./smolvla_eval_output
```

### 3. FastTD3 (RL) 评测

```bash
python roboverse_learn/rl/fast_td3/evaluate.py \
    --checkpoint models/walk_10000.pt \
    --num-episodes 10 \
    --device-rank 0 \
    --num-envs 8 \
    --headless \
    --render 1 \
    --render-each-episode \
    --video-path output/eval_rollout.mp4 \
    --save-traj 1 \
    --save-states 1 \
    --save-every-n-steps 1 \
    --traj-dir eval_trajs
```

### 4. IL 评测

```bash
# 通常嵌入在训练脚本中，通过参数控制
bash roboverse_learn/il/il_run.sh \
    --task_name_set pick_butter \
    --policy_name ddpm_dit \
    --eval
```

---

## 八、关键特性总结

### 1. 多模拟器支持

| 模拟器 | 向量化 | 性能 | GPU加速 | 适用场景 |
|---------|--------|------|---------|----------|
| **mujoco** | ❌ | 中等 | ❌ | 快速原型、兼容性好 |
| **mjx** | ✅ | 极快 | ✅ (JAX) | 大规模RL训练 |
| **isaacgym** | ✅ | 极快 | ✅ (CUDA) | GPU加速的RL |
| **isaaclab** | ✅ | 高 | ✅ (CUDA) | 最新NVIDIA栈 |
| **genesis** | ✅ | 极快 | ✅ (CUDA) | 现代物理引擎 |
| **pybullet** | ✅ | 中等 | ❌ | 兼容性好 |
| **sapien2** | ✅ | 高 | ✅ (CUDA) | 渲染质量优先 |
| **sapien3** | ✅ | 高 | ✅ (CUDA) | 最新版本 |

### 2. 域随机化

| 级别 | 场景 | 材质 | 灯光 | 相机 | 测试目标 |
|------|------|------|------|------|----------|
| **0** | ❌ | ❌ | ❌ | ❌ | 标准评测能力 |
| **1** | ✅ | ✅ | ❌ | ❌ | 场景泛化能力 |
| **2** | ✅ | ✅ | ✅ | ❌ | 光照鲁棒性 |
| **3** | ✅ | ✅ | ✅ | ✅ | 视角不变性 |

### 3. 多种策略类型

| 策略类型 | 动作空间 | 动作块 | 时序聚合 | IK求解 | 代表模型 |
|---------|---------|--------|---------|--------|----------|
| **IL (ACT)** | Joint Pos | ✅ (chunk) | ✅ | ❌ | Diffusion Policy |
| **IL (DP)** | Joint Pos | ✅ (chunk) | ✅ | ❌ | Flow Matching |
| **VLA (Pi0)** | Joint Pos | ✅ (cached) | ❌ | ❌ | Transformer |
| **VLA (SmolVLA)** | EE Delta | ❌ | ❌ | ✅ | LeRobot Policy |
| **RL (TD3)** | Joint Vel | ❌ | ❌ | ❌ | Actor-Critic |

### 4. 视频录制

```python
# ObsSaver自动处理
obs_saver = ObsSaver(video_path="output.mp4")
obs_saver.add(obs)  # 每步添加
obs_saver.save()  # 自动编码保存
```

### 5. 轨迹保存

```python
# 完整状态保存（用于精确重放）
save_states=True  # 包含所有物体和机器人的位置/旋转/关节
                # 文件较大但可完全恢复场景

# 仅动作保存（节省空间）
save_states=False  # 只保存机器人关节命令
                # 需要演示数据才能重放
```

### 6. 成功检测

```python
# 基于Detector-Checker模式
checker = DetectedChecker(
    detector=Relative2DSphereDetector(
        base_obj_name="goal_region",
        relative_pos=(0.0, 0.0, 0.0),
        radius=0.15,
        axis=(0, 1)  # XY平面
    ),
    obj_name="cube"
)

# 自动计算
success = checker.check(handler.get_states())

# 替代：手工实现
def _terminated(self, env_states: Obs) -> Termination:
    cube_pos = env_states.objects["cube"].root_state[:, :3]
    target_pos = env_states.objects["goal"].root_state[:, :3]
    distance = torch.norm(cube_pos - target_pos, dim=1)
    return distance < 0.05
```

### 7. 性能优化

| 优化项 | 说明 | 命令行参数 |
|--------|------|-------------|
| **无头模式** | 不渲染GUI，仅后台运行 | `--headless` |
| **向量环境** | 多环境并行评测 | `--num-envs N` |
| **动作缓存** | 减少策略推理次数 | `--actions-per-call N` |
| **AMP** | 自动混合精度训练 | `amp_enabled=True` |
| **降频录制** | 减少保存的帧数 | `--save-every-n-steps 5` |
| **异步向量化** | Gymnasium AsyncVectorEnv | `num_envs=8` |

---

## 九、使用示例

### 完整评测流程（Pi0为例）

```bash
# === 步骤1: 启动策略服务器 ===
# Terminal 1
cd third_party/openpi
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_roboverse_lora \
    --policy.dir=/path/to/checkpoint

# === 步骤2: 运行评测 ===
# Terminal 2
python roboverse_learn/vla/pi0/pi_eval.py \
    --task PickCube \
    --robot franka \
    --sim mujoco \
    --policy-host localhost \
    --policy-port 8000 \
    --num-episodes 10 \
    --max-steps 250 \
    --level 2 \
    --scene-mode 1 \
    --randomization-seed 42 \
    --output-dir ./pi_eval_output

# === 步骤3: 查看结果 ===
# 检查输出目录
ls -lh ./pi_eval_output/

# 查看JSON报告
cat ./pi_eval_output/pi_eval_PickCube_20250125_143022.json

# 播放视频
ffplay ./pi_eval_output/episode_001.mp4
```

### 域随机化测试流程

```bash
# Level 0: 无随机化（基准）
python smolvla_eval.py --model_path checkpoints/000100 \
    --task pick_butter --level 0 \
    --output-dir ./eval_dr0

# Level 1: 场景+材质随机化
python smolvla_eval.py --model_path checkpoints/000100 \
    --task pick_butter --level 1 \
    --output-dir ./eval_dr1

# Level 2: +灯光随机化
python smolvla_eval.py --model_path checkpoints/000100 \
    --task pick_butter --level 2 \
    --output-dir ./eval_dr2

# Level 3: +相机随机化（最强泛化测试）
python smolvla_eval.py --model_path checkpoints/000100 \
    --task pick_butter --level 3 \
    --output-dir ./eval_dr3

# 对比结果
for level in 0 1 2 3; do
    success_rate=$(jq '.eval_stats.success_rate' ./eval_dr$level/smolvla_eval_*.json)
    echo "Level $level: Success Rate = $(echo "$success_rate * 100" | bc)%" 
done
```

---

## 十、常见问题

### Q1: 如何选择模拟器？

**A**:
- **评测**: 推荐 `mujoco`（稳定、兼容）
- **训练RL**: 推荐 `isaacgym` 或 `mjx`（原生向量化、GPU加速）
- **高质量渲染**: 推荐 `isaaclab` 或 `sapien3`

### Q2: 如何调试评测失败？

**A**:
```python
# 1. 检查动作空间
print(f"Action shape: {action.shape}")
print(f"Expected: ({num_envs}, {action_dim})")

# 2. 检查观测空间
print(f"Obs keys: {obs.keys()}")
print(f"Joint pos shape: {obs['joint_qpos'].shape}")

# 3. 检查IK求解
print(f"IK success rate: {ik_succ.mean().item():.2%}")

# 4. 可视化失败回合
# 保存失败回合的视频，手动观察
```

### Q3: 如何加速评测？

**A**:
1. 使用向量环境: `--num-envs 8`
2. 无头模式: `--headless`
3. 降低录制频率: `--save-every-n-steps 5`
4. 减少视频分辨率: `width=128 height=128`

### Q4: 如何复现评测结果？

**A**:
```bash
# 设置相同种子
--randomization-seed 42
--seed 42

# 设置相同随机化级别
--level 2
--scene-mode 1

# 使用相同的checkpoint
--checkpoint path/to/checkpoint.pt
```

---

## 总结

RoboVerse 的评测系统提供了：

✅ **灵活性**: 支持8+模拟器、多种算法类型
✅ **可扩展性**: 任务注册机制、配置继承
✅ **泛化测试**: 4级域随机化
✅ **可视化**: 自动视频录制、轨迹保存
✅ **标准化**: Gymnasium API、统一指标

适用于研究、开发和部署阶段的机器人学习任务评测。

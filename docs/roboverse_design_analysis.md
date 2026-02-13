# RoboVerse 平台设计分析

> 基于 RoboVerse 论文和代码库的架构分析，特别关注 Benchmark 相关设计

---

## 目录

1. [总体架构](#一总体架构)
2. [Benchmark 设计](#二benchmark-设计)
3. [泛化评测机制](#三泛化评测机制)
4. [数据格式设计](#四数据格式设计)
5. [场景随机化系统](#五场景随机化系统)
6. [评测指标体系](#六评测指标体系)
7. [对本项目的参考价值](#七对本项目的参考价值)

---

## 一、总体架构

### 1.1 平台定位

RoboVerse 是一个**云机器人评测平台**，旨在提供：
- 统一的数据格式
- 标准化的评测协议
- 可扩展的基础设施
- 支持机器人学习算法的规模化评测和泛化能力评估

### 1.2 核心组件

```
┌─────────────────────────────────────────────────────────────┐
│                      RoboVerse Platform                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │  MetaSim    │  │  RoboVerse   │  │  RoboVerse   │   │
│  │              │  │     Pack     │  │    Learn     │   │
│  │ 统一模拟器    │  │   任务包      │  │  学习算法库    │   │
│  │   抽象层     │  │              │  │   (IL/RL)    │   │
│  └──────────────┘  └──────────────┘  └──────────────┘   │
│                                                             │
│  ┌────────────────────────────────────────────────────────┐  │
│  │              Generation (资产转换)                     │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

| 组件 | 职责 | 主要内容 |
|------|------|----------|
| **MetaSim** | 统一模拟器抽象层 | 支持多后端（Isaac, SAPIEN, MuJoCo等） |
| **RoboVerse Pack** | 任务定义 | 具体的机器人任务配置 |
| **RoboVerse Learn** | 算法实现 | 模仿学习（ACT, Diffusion）和强化学习（PPO, TD3） |
| **Generation** | 资产处理 | 格式转换（URDF→USD）和数据生成 |

---

## 二、Benchmark 设计

### 2.1 评测协议架构

```
┌─────────────────────────────────────────────────────────────┐
│                 RoboVerse Benchmark System                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Episode Data (HDF5)                     │  │
│  │  • Action sequences                                │  │
│  │  • Observations (RGB, Depth, State)               │  │
│  │  • Metadata (robot, scene, language)               │  │
│  └──────────────────────────────────────────────────────┘  │
│                          │                                │
│                          ▼                                │
│  ┌──────────────────────────────────────────────────────┐  │
│  │           Evaluation Engine (统一评测协议)             │  │
│  │  • Episode replay                                  │  │
│  │  • Metric computation                              │  │
│  │  • Randomization application                       │  │
│  └──────────────────────────────────────────────────────┘  │
│                          │                                │
│         ┌────────────────┴────────────────┐               │
│         ▼                                 ▼               │
│  ┌──────────────┐                 ┌──────────────┐       │
│  │  Scenarios   │                 │   Metrics    │       │
│  │ (多场景支持)  │                 │ Success, SPL │       │
│  └──────────────┘                 └──────────────┘       │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 评测流程

```
┌──────────────────────────────────────────────────────────┐
│  1. 数据收集 (Data Collection)                          │
│     • 多种场景 / 机器人 / 任务                            │
│     • 生成 HDF5 episode 数据                            │
│     • 标注语言指令和成功条件                             │
└──────────────────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│  2. 训练 (Training - RoboVerse Learn)                  │
│     • IL: ACT, Diffusion Policy, DiT, VITA             │
│     • RL: PPO, TD3, SAC, Fast-TD3                      │
└──────────────────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│  3. 评测 (Evaluation)                                  │
│     • Level 0: 同分布评测 (In-Distribution)             │
│     • Level 1-3: 泛化评测 (Out-of-Distribution)         │
│     • 记录指标: Success Rate, SPL, Trajectory Similarity │
└──────────────────────────────────────────────────────────┘
```

---

## 三、泛化评测机制

### 3.1 随机化配置系统

RoboVerse 的核心创新是通过**分层随机化**实现不同难度的泛化评测。

```python
@configclass
class RandomizationCfg:
    """随机化配置"""

    camera: bool = False
    """随机化相机位姿"""

    light: bool = False
    """随机化光照方向、温度、强度"""

    ground: bool = False
    """随机化地面"""

    reflection: bool = False
    """随机化材质（粗糙度、金属度、反射率）"""

    table: bool = False
    """随机化桌面颜色"""

    wall: bool = False
    """添加墙壁和天花板，随机化墙壁"""

    scene: bool = False
    """随机化场景"""

    level: Literal[0, 1, 2, 3] = 0
    """随机化难度级别"""

    def __post_init__(self):
        """自动应用级别配置"""
        if self.level >= 1:
            self.table = True
            self.ground = True
            self.wall = True
        if self.level >= 2:
            self.camera = True
        if self.level >= 3:
            self.light = True
            self.reflection = True
```

### 3.2 难度级别详解

| Level | 随机化内容 | 测试能力 | 典型表现 |
|-------|------------|----------|----------|
| **L0** | 无随机化 | 基线性能 | 模型过拟合风险高 |
| **L1** | table + ground + wall | 视觉域内泛化 | 颜色/纹理变化 |
| **L2** | L1 + camera | 视角鲁棒性 | 相机位置/角度变化 |
| **L3** | L2 + light + reflection | 完全域泛化 | 光照/材质/场景变化 |

**设计理念**：渐进式难度增加，便于定位模型泛化瓶颈。

### 3.3 使用示例

```python
# 定义随机化配置
randomization = RandomizationCfg(
    camera=False,
    light=False,
    ground=False,
    reflection=False
)

# 使用级别快速配置
randomization = RandomizationCfg(level=2)  # 自动启用 L2 的所有随机化

# 集成到场景配置
scenario = ScenarioCfg(
    task=task,
    robot=robot,
    cameras=[camera],
    randomization=randomization,
    try_add_table=True
)
```

---

## 四、数据格式设计

### 4.1 HDF5 Episode 结构

```
episode/
├── action                    # 动作序列
│   └── (N, action_dim)
├── observations              # 观测数据
│   ├── images              # RGB 图像
│   │   └── (N, H, W, 3)
│   ├── depths              # 深度图
│   │   └── (N, H, W)
│   ├── qpos                # 关节位置
│   │   └── (N, dof)
│   └── qvel                # 关节速度
│       └── (N, dof)
└── metadata                # 元数据
    ├── robot_type          # 机器人类型
    ├── scene_id           # 场景ID
    ├── task_name          # 任务名称
    └── language           # 语言指令（如有）
```

### 4.2 关键设计原则

1. **自包含**: 每个 episode 包含完整的数据，便于离线评估
2. **可扩展**: metadata 支持自定义字段
3. **多模态**: 支持 RGB、Depth、State 等多种观测
4. **标准化**: 统一的数据格式支持跨任务训练

---

## 五、场景随机化系统

### 5.1 三层场景架构

SceneRandomizer 提供分层管理：

```python
@configclass
class SceneRandomCfg:
    """场景随机化配置"""

    environment_layer: SceneLayerCfg | None = None
    """环境层：地板、墙壁、天花板"""

    workspace_layer: SceneLayerCfg | None = None
    """工作区层：桌子、桌面"""

    objects_layer: SceneLayerCfg | None = None
    """物体层：干扰物"""

    auto_flush_visuals: bool = True
    """自动刷新视觉更新"""

    only_if_no_scene: bool = False
    """仅在无场景时创建"""
```

### 5.2 场景元素类型

| 类型 | 配置类 | 说明 | 示例 |
|------|--------|------|------|
| **手动几何** | ManualGeometryCfg | 程序化生成的几何体 | 立方体桌子、平面地板 |
| **单个资产** | USDAssetCfg | 单个 USD 模型 | 特定桌子模型 |
| **资产池** | USDAssetPoolCfg | 从池中随机选择 | 从多个桌子模型中选一个 |

### 5.3 层级配置示例

```python
# 工作区层配置
workspace_layer = SceneLayerCfg(
    shared=True,  # 所有环境共享
    elements=[
        ManualGeometryCfg(
            name="table",
            geometry_type="cube",
            size=(1.8, 1.8, 0.1),
            position=(0.5, 0, 0.4),
        )
    ]
)

# 物体层配置
objects_layer = SceneLayerCfg(
    shared=False,  # 每个环境独立
    elements=[
        USDAssetPoolCfg(
            name="distractor",
            usd_paths=[
                "assets/cup1.usd",
                "assets/cup2.usd",
                "assets/bowl1.usd",
            ],
            selection_strategy="random"
        )
    ]
)
```

### 5.4 资产生命周期

```
┌─────────────────────────────────────────────────────────┐
│              SceneRandomizer Lifecycle                  │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. Create     创建几何体/加载 USD                      │
│      │                                                   │
│      ▼                                                   │
│  2. Register   注册到 ObjectRegistry                    │
│      │                                                   │
│      ▼                                                   │
│  3. Transform 设置变换（位置/旋转/缩放）                   │
│      │                                                   │
│      ▼                                                   │
│  4. Switch    替换 USD 资产（资产池场景）                │
│      │                                                   │
│      ▼                                                   │
│  5. Delete     删除动态对象                               │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 六、评测指标体系

### 6.1 操作任务

| 指标 | 说明 | 计算方式 |
|------|------|----------|
| **Success Rate** | 任务成功率 | 满足成功条件的 episode 比例 |
| **Trajectory Similarity** | 轨迹相似度 | 与演示轨迹的 DTW 距离 |

### 6.2 导航任务

| 指标 | 说明 | 计算方式 |
|------|------|----------|
| **Success Rate** | 是否到达目标 | 距离阈值内判定成功 |
| **SPL** | 成功加权的路径长度 | $SPL = \frac{S \cdot L_{opt}}{L_{actual}}$ |
| **Navigation Error** | 导航误差 | 终点与目标的欧氏距离 |

### 6.3 Benchmark 结果示例

**In-Distribution Evaluation** (训练集分布):

| Task | RGB | RGBD | PointCloud | Fusion |
|------|-----|------|------------|--------|
| CloseBoxL0 | 0.81 | 0.91 | 0.92 | **1.00** |
| CloseBoxL1 | 0.40 | 0.58 | 0.88 | **0.94** |
| CloseBoxL2 | 0.42 | 0.30 | 0.62 | **0.95** |

**Out-of-Distribution** (零样本泛化):

| Task | RGB | RGBD | PointCloud | Fusion |
|------|-----|------|------------|--------|
| CloseBoxL0 | 0.52 | 0.72 | 0.94 | **0.97** |
| CloseBoxL1 | 0.20 | 0.50 | 0.88 | **0.95** |
| CloseBoxL2 | 0.32 | 0.38 | 0.42 | **0.95** |

**关键发现**：
- Fusion 模型在所有场景下表现最佳
- Level 2 随机化对纯 RGB 方法挑战较大
- Point Cloud 方法在泛化场景中表现稳定

---

## 七、对本项目的参考价值

### 7.1 可直接借鉴的设计

#### A. 数据格式
- 采用 HDF5 作为 episode 存储格式
- 分离 action、observations、metadata
- 支持 RGB、Depth、State 等多模态观测

#### B. 泛化评测
- **L0-L3 分层随机化**机制
- 渐进式难度设计便于定位瓶颈
- 配置驱动的随机化系统

#### C. 场景管理
- 场景/机器人/任务的**解耦设计**
- ScenarioCfg 统一配置接口
- 资产池支持动态切换

#### D. 评测协议
- 统一的 Episode Replay 机制
- 标准化的指标计算
- 自动化的评测流程

### 7.2 与 VLN 评测的结合

| RoboVerse 设计 | VLN 项目应用 |
|----------------|-------------|
| L0-L3 随机化 | 视觉外观随机化（光照、纹理） |
| SceneRandomizer | 场景动态切换（建筑物、天气） |
| HDF5 Episode | VLN 轨迹数据存储 |
| SPL 指标 | VLN 导航成功率指标 |
| Fusion 模型 | 多模态（视觉+语言）融合 |

### 7.3 建议采用的模式

```python
# 建议的 VLN 评测配置结构
@configclass
class VLNEvaluationCfg:
    """VLN 评测配置"""

    # 数据集
    dataset: str = "R2R"
    split: str = "val_seen"

    # 随机化
    randomization: VLNRandomizationCfg = VLNRandomizationCfg(
        lighting=True,      # 光照变化
        texture=True,       # 纹理变化
        camera_pose=True,   # 相机位姿变化
        weather=True,       # 天气效果
        level=1            # 难度级别
    )

    # 评测指标
    metrics: list[str] = [
        "success_rate",
        "spl",
        "navigation_error",
        "trajectory_length"
    ]

    # 代理配置
    agent: AgentCfg = AgentCfg(...)
```

---

## 八、关键代码位置参考

| 功能模块 | 文件路径 |
|----------|----------|
| 场景随机化 | `metasim/randomization/scene_randomizer.py` |
| 随机化配置 | `metasim/scenario/scenario.py` |
| 场景定义 | `metasim/scenario/scene.py` |
| 相机随机化 | `metasim/randomization/camera_randomizer.py` |
| 光照随机化 | `metasim/randomization/light_randomizer.py` |
| 材质随机化 | `metasim/randomization/material_randomizer.py` |
| 评测脚本 | `roboverse_learn/rl/eval*.py` |
| 训练脚本 | `roboverse_learn/il/train.py` |

---

## 九、总结

RoboVerse 的设计亮点：

1. **统一的抽象层**: MetaSim 屏蔽多模拟器差异
2. **分层随机化**: L0-L3 渐进式泛化评测
3. **配置驱动**: 通过配置文件定义场景和随机化
4. **资产管理**: 自动下载、转换、实例化
5. **标准化**: 统一的数据格式和评测协议

这些设计为本项目的 VLN 评测架构提供了重要参考，特别是在**泛化能力评测**和**场景管理**方面。

---

*文档生成时间: 2026-02-12*
*参考来源: RoboVerse 论文 + GitHub 代码库*

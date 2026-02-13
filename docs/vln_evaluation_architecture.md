# VLN导航任务评测系统架构设计文档

## 1. 项目概述

### 1.1 目标
设计一个独立的VLN（Vision-Language Navigation）评测系统，用于学术竞赛。参考Habitat生态系统设计模式，但完全独立实现。

### 1.2 核心需求
- **场景**: 学术竞赛评测系统（类似Habitat Challenge）
- **任务**: VLN（视觉语言导航），支持可扩展架构以容纳未来任务
- **交互方式**: 参赛者部署独立推理服务，评测器通过API调用
- **指标**: SPL, Success Rate, Navigation Error/DTW + 可扩展指标系统
- **仿真**: 独立仿真引擎（不依赖Habitat-sim）

### 1.3 设计原则
- **模块化**: 各组件职责清晰，松耦合
- **可扩展**: 插件化架构，易于添加新任务、指标、仿真器
- **公平性**: 统一评测标准，确保竞赛公正
- **易用性**: 为参赛者提供清晰的SDK和文档

### 1.4 领域概念定义

#### 1.4.1 核心概念

| 概念 | 定义 | 职责 | 示例 |
|------|------|------|------|
| **Task（任务）** | 导航任务的抽象定义，描述要解决什么问题 | 定义动作空间、观测空间、目标条件 | VLN, ObjectNav, ImageNav |
| **Benchmark（基准）** | 用于评测的一组Episode集合和评测规则 | 组织Episode、定义评测配置、输出结果 | "VLN Challenge 2024 - val_seen" |
| **Episode（回合/场景）** | 单次评测任务实例，包含初始状态和目标 | 描述场景、起点、终点、指令 | episode_001: "从卧室走到厨房" |
| **Simulator（仿真器）** | 模拟物理环境和传感器观测的引擎 | 渲染场景、计算物理、生成观测 | Custom Simulator |
| **Agent（代理/智能体）** | 执行导航决策的实体，由参赛者实现 | 接收观测、输出动作 | Team XYZ's VLNAgent |
| **Metric（指标）** | 衡量Agent性能的量化标准 | 计算单次或聚合性能值 | Success, SPL, DTW |
| **Sensor（传感器）** | 获取环境观测的接口 | 生成特定类型的观测数据 | RGB Camera, Depth Sensor |
| **Action（动作）** | Agent对环境的控制指令 | 改变Agent状态或位置 | move_forward, turn_left |
| **Observation（观测）** | 环境对Agent的反馈信息 | 提供当前状态的多模态数据 | RGB图像、深度图、指令 |
| **Trajectory（轨迹）** | Agent在Episode中的位置序列 | 记录导航路径、用于指标计算 | [(x1,y1,z1), (x2,y2,z2), ...] |
| **Scene Dataset（场景数据集）** | 3D场景模型的集合 | 提供仿真环境、渲染基础 | HM3D, Gibson, Replica, Matterport3D |
| **Task Dataset（任务数据集）** | Episode的结构化集合 | 提供评测用例、保证可复现性 | R2R val_seen (1000 episodes) |
| **Trajectory Dataset（轨迹数据集）** | Agent执行轨迹的结构化存储 | 保存执行结果、支持重放分析 | 成功轨迹集合、失败案例分析 |
| **Configuration（配置）** | 系统行为的参数化描述 | 控制评测流程、任务参数 | YAML配置文件 |

#### 1.4.2 概念关系图

```
┌─────────────────────────────────────────────────────────────────┐
│                         Benchmark（基准）                        │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │                    Task（任务）                            │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐│ │
│  │  │  Action     │  │  Sensor     │  │  Metric              ││ │
│  │  │  Space      │  │  Suite      │  │  Definitions         ││ │
│  │  └─────────────┘  └─────────────┘  └─────────────────────┘│ │
│  └───────────────────────────────────────────────────────────┘ │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │              Task Dataset（任务数据集）                  │ │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐     │ │
│  │  │Episode 1│  │Episode 2│  │Episode 3│  │Episode N│     │ │
│  │  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘     │ │
│  └───────┼───────────┼───────────┼───────────┼────────────┘ │
└──────────┼───────────┼───────────┼───────────┼─────────────┘
           │           │           │           │
           ▼           ▼           ▼           ▼
    ┌──────────────────────────────────────────────────────┐
    │              Episode Execution（回合执行）              │
    │                                                       │
    │  ┌────────────────┐         ┌────────────────┐      │
    │  │   Simulator    │────────▶│  Observation   │      │
    │  │   (仿真器)      │         │  (观测)         │      │
    │  └────────┬───────┘         └────────┬───────┘      │
    │           │                          │              │
    │           │ loads                    │              │
    │           ▼                          │              │
    │  ┌────────────────────┐              │              │
    │  │  Scene Dataset     │              │              │
    │  │  (场景数据集)       │              │              │
    │  └────────────────────┘              │              │
    │                                      │              │
    │  ┌────────────────┐         ┌────────▼───────┐      │
    │  │     Agent      │◀────────│   VLNAgent     │      │
    │  │  (代理/智能体)  │         │  (参赛者实现)    │      │
    │  └───────┬────────┘         └────────────────┘      │
    │          │                                          │
    │          ▼                                          │
    │  ┌────────────────┐         ┌────────────────┐      │
    │  │    Action      │────────▶│   Trajectory  │      │
    │  │    (动作)       │         │   (轨迹)        │      │
    │  └────────────────┘         └────────┬───────┘      │
    │                                      │              │
    │                                      ▼              │
    │                              ┌────────────────┐     │
    │                              │    Metric      │     │
    │                              │   (指标计算)     │     │
    │                              └────────┬───────┘     │
    └───────────────────────────────────────┼──────────────┘
                                            │
                                            ▼ saves to
                                    ┌────────────────────┐
                                    │ Trajectory Dataset │
                                    │ (轨迹数据集)        │
                                    └────────────────────┘
```

#### 1.4.3 概念层次关系

**1. 配置层（Configuration Layer）**
```
Configuration
├── Task Config (定义任务规则)
├── Benchmark Config (定义评测集合)
└── Simulator Config (定义仿真参数)
```

**2. 数据层（Data Layer）**
```
┌── Scene Dataset（场景数据集）
│   ├── Scene 1 (3D模型、纹理、网格)
│   ├── Scene 2
│   └── Scene N
│
├── Task Dataset（任务数据集/Episode集合）
│   ├── Episode 1
│   │   ├── scene_id → Scene Dataset
│   │   ├── start_position
│   │   ├── instruction
│   │   └── reference_path
│   ├── Episode 2
│   └── Episode N
│
└── Trajectory Dataset（轨迹数据集 - 评测产出）
    ├── Trajectory 1 (执行结果)
    │   ├── episode_id
    │   ├── positions: [(x1,y1,z1), ...]
    │   ├── actions: [move_forward, ...]
    │   └── metrics: {success: 1.0, spl: 0.85}
    ├── Trajectory 2
    └── Trajectory N
```

**3. 运行时层（Runtime Layer）**
```
Episode Execution
├── Simulator (环境)
│   ├── Scene ← from Scene Dataset
│   └── Sensors (传感器)
├── Agent (决策者)
└── Metrics (评价者)
```

#### 1.4.4 概念间的依赖关系

| 依赖关系 | 描述 |
|----------|------|
| **Benchmark → Task** | Benchmark必须指定一个Task类型 |
| **Benchmark → Task Dataset** | Benchmark引用任务数据集的特定split |
| **Task Dataset → Scene Dataset** | Task Dataset中的Episode引用Scene Dataset中的场景 |
| **Simulator → Scene Dataset** | Simulator加载Scene Dataset中的3D场景模型 |
| **Episode → Simulator** | Episode执行需要Simulator实例 |
| **Episode → Agent** | Episode需要Agent做出决策 |
| **Agent → Task** | Agent必须符合Task的动作/观测空间 |
| **Metric → Task** | Metric定义依赖于Task类型 |
| **Simulator → Sensors** | Simulator通过Sensors生成观测 |
| **Observation → Sensors** | Observation是Sensors的输出结果 |
| **Trajectory → Episode** | Trajectory记录Episode中的位置序列 |
| **Trajectory Dataset → Trajectory** | Trajectory Dataset保存Episode执行产生的轨迹 |
| **Metric → Trajectory** | 部分指标需要轨迹数据计算 |

#### 1.4.5 关键概念详解

**Task（任务）**
- **是什么**: 定义"要解决什么问题"
- **包含内容**:
  - 动作空间: Agent可以执行的操作
  - 观测空间: Agent能接收的信息
  - 目标条件: 何时视为完成
  - 指标集合: 如何评估性能
- **为什么需要**: 不同导航任务有不同规则，Task提供抽象
- **示例**: VLN任务需要理解自然语言指令，ObjectNav需要找到指定物体

**Benchmark（基准）**
- **是什么**: 一个可执行的评测配置
- **包含内容**:
  - 引用的Task定义
  - 数据集路径和split
  - 评测参数（max_steps, timeout等）
  - 输出配置（日志、轨迹保存）
- **为什么需要**: 提供标准化的评测环境，确保公平对比
- **示例**: "VLN Challenge 2024 - val_seen" 定义了在val_seen split上如何评测

**Episode（回合）**
- **是什么**: 单次评测的完整场景定义
- **生命周期**: INIT → RUNNING → COMPLETED/FAILED
- **包含内容**:
  - 场景信息: scene_id
  - 初始状态: start_position, start_rotation
  - 任务目标: instruction, goals
  - 参考路径: reference_path（用于DTW等指标）
- **为什么需要**: 提供可重复的评测用例
- **独立性**: 每个Episode可独立执行，便于并行

**Simulator（仿真器）**
- **是什么**: 模拟物理世界的软件引擎
- **职责**:
  - 场景渲染: 生成视觉观测
  - 物理计算: 碰撞检测、位置更新
  - 传感器模拟: 生成RGB、Depth等数据
- **隔离性**: 每个Worker进程有独立Simulator实例
- **为什么需要**: 提供可控、可重复的测试环境

**Agent（代理/智能体）**
- **是什么**: 参赛者实现的导航算法
- **职责**:
  - 接收观测（Observation）
  - 决策并输出动作（Action）
  - 可维护内部状态（RNN隐藏层、地图等）
- **接口方式**: 通过WebSocket/gRPC与评测系统通信
- **隔离性**: 作为独立服务运行，评测系统调用其API

**Metric（指标）**
- **是什么**: 量化Agent性能的度量标准
- **计算时机**:
  - Episode结束时: Success, SPL, Navigation Error
  - Episode过程中: Collisions, Steps
- **接口模式**: reset() → update() → get_metric()
- **为什么需要**: 提供客观的性能评估标准

**Sensor（传感器）**
- **是什么**: 模拟真实传感器的数据采集接口
- **类型**: 视觉（RGB/Depth）、位置（GPS）、朝向（Compass）、语义
- **配置**: 分辨率、视野、范围等参数
- **为什么需要**: 提供多模态观测，模拟真实机器人感知

**Observation（观测）**
- **是什么**: 某时刻环境的完整状态表示
- **组成**: 各Sensors的输出集合 + 任务特定信息（如instruction）
- **数据流**: Simulator.step(action) → next_observation
- **格式**: 字典结构，支持base64编码的图像数据

**Trajectory（轨迹）**
- **是什么**: Agent在Episode中经过的位置序列
- **用途**:
  - 计算路径效率指标（SPL）
  - 计算路径相似度（DTW）
  - 可视化分析
- **记录频率**: 每步记录Agent位置

**Scene Dataset（场景数据集）**
- **是什么**: 3D场景模型的集合，提供仿真环境的基础
- **包含内容**:
  - 3D模型: 房屋、建筑物的几何网格
  - 纹理材质: 视觉渲染所需的纹理贴图
  - 导航网格: 可行走区域的定义
  - 语义标注: 物体类别、房间类型（可选）
- **格式**: GLB/USD/FBX等3D格式
- **存储位置**: `data/scene_datasets/`
- **为什么需要**: 提供逼真的视觉环境，是Simulator渲染观测的基础
- **示例**:
  - HM3D: 216个高保真室内场景
  - Gibson: 572个真实扫描场景
  - Replica: 18个高质量室内场景
  - Matterport3D: 90个建筑场景

**Task Dataset（任务数据集）**
- **是什么**: Episode的结构化集合，定义评测用例
- **包含内容**:
  - Episode列表: 每个Episode包含起点、终点、指令
  - 指令数据: 自然语言描述（VLN）或目标物体（ObjectNav）
  - 参考路径: 专家演示的最短路径（用于DTW等指标）
  - 场景索引: 指向Scene Dataset中的场景ID
- **格式**: JSON.gz压缩文件
- **Split划分**: train/val_seen/val_unseen/test
- **存储位置**: `data/datasets/{task_type}/{dataset_name}/`
- **为什么需要**: 提供标准化的评测用例，确保可复现性和公平对比
- **示例**:
  - R2R: 14,000条指令，7个场景
  - RxR: 126,000条多语言指令
  - SOHN: ObjectNav目标物体数据集
- **数据流向**: Task Dataset → Episode → Simulator → Observation

**Trajectory Dataset（轨迹数据集）**
- **是什么**: Agent执行轨迹的结构化存储，评测产出的数据
- **核心用途**: **作为模仿学习（Imitation Learning）的训练数据集**
- **包含内容**:
  - 位置序列: Agent在每个step的坐标 [(x,y,z), ...]
  - 动作序列: Agent采取的所有动作 [move_forward, turn_left, ...]
  - 观测快照: RGB/Depth图像（用于训练视觉模型）
  - 指标结果: success, spl, navigation_error等（用于筛选高质量轨迹）
  - 元数据: episode_id, timestamp, agent_version等
- **格式**: JSON/JSONL/HDF5
- **存储位置**: `logs/evaluations/{benchmark}/trajectories/` 或 `data/trajectory_datasets/`
- **为什么需要**:
  - **模仿学习训练**: 为新模型提供专家演示数据（主要用途）
  - **高质量数据筛选**: 基于success/spl等指标筛选成功轨迹用于训练
  - 错误分析: 研究失败案例的原因
  - 可视化: 生成导航轨迹视频
  - 可复现性: 记录完整执行过程
- **用途**:
  - **模仿学习训练集**: 从成功轨迹中学习导航策略（Behavior Cloning）
  - **数据增强**: 对轨迹进行变换和扩展
  - **课程学习**: 按难度（轨迹长度、SPL等）组织训练数据
  - Agent行为分析和调试
  - 评测结果归档和可视化

#### 1.4.6 概念与代码映射

| 概念 | 类/模块 | 文件位置 |
|------|---------|----------|
| Benchmark | `BenchmarkConfig` | `core/config/` |
| Task | `Task`, `TaskRegistry` | `tasks/` |
| Episode | `Episode` | `dataset/episode.py` |
| Scene Dataset | `SceneDataset`, `SceneManager` | `simulator/scene/` |
| Task Dataset | `TaskDataset`, `DatasetLoader` | `dataset/dataset.py` |
| Trajectory Dataset | `TrajectoryDataset`, `TrajectoryWriter` | `dataset/trajectory.py` |
| Simulator | `Simulator` | `simulator/base.py` |
| Agent | 参赛者实现 | 外部服务 |
| Sensor | `Sensor` | `simulator/sensors/` |
| Metric | `Metric` | `metrics/base.py` |
| Observation | `Observations` (TypedDict) | `core/interfaces/` |
| Action | `Action` (TypedDict) | `core/interfaces/` |
| Trajectory | `List[Tuple[float,float,float]]` | `utils/geometry.py` |

---

## 2. 系统架构

### 2.1 高层架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                    EVALUATION ORCHESTRATOR                      │
│                    （评测编排层）                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │  Benchmark   │  │   Episode    │  │    Metrics   │         │
│  │   Runner     │  │   Manager    │  │  Aggregator  │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                    SIMULATION ENGINE LAYER                      │
│                    （仿真引擎层）                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │           INDEPENDENT SIMULATOR                          │  │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐       │  │
│  │  │   Scene    │  │   Agent    │  │  Sensor    │       │  │
│  │  │  Manager   │  │  Manager   │  │   Suite    │       │  │
│  │  └────────────┘  └────────────┘  └────────────┘       │  │
│  └──────────────────────────────────────────────────────────┘  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                       API INTERFACE LAYER                       │
│                    （通信接口层）                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              REMOTE AGENT CLIENT                         │  │
│  │        REST API / gRPC Protocol Buffers                  │  │
│  └──────────────────────────────────────────────────────────┘  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                    ┌───────▼────────────┐
                    │  PARTICIPANT'S     │
                    │  INFERENCE SERVICE │
                    │  （参赛者推理服务）  │
                    └────────────────────┘
```

### 2.2 分层架构说明

| 层级 | 职责 | 关键组件 |
|------|------|----------|
| **评测编排层** | 管理评测流程、聚合指标、生成报告 | BenchmarkRunner, EpisodeManager, MetricsAggregator |
| **仿真引擎层** | 场景渲染、物理仿真、传感器数据生成 | Scene, Agent, Sensors |
| **通信接口层** | 与参赛者服务通信，协议转换 | REST/gRPC Client, Protocol Buffers |
| **推理服务层** | 参赛者部署的推理服务（外部） | VLNAgent实现 |

---

## 3. 核心组件设计

### 3.1 评测编排器 (Evaluation Orchestrator)

#### 3.1.1 分层模块结构

```
EvaluationOrchestrator (总控制器)
    │
    ├─→ BenchmarkRunner (基准运行器)
    │      │
    │      └─→ EpisodeManager (回合管理器)
    │            │
    │            ├─→ ParallelExecutor (并行执行器)
    │            │      │
    │            │      ├─→ WorkerPool (进程池)
    │            │      └─→ TaskQueue (任务队列)
    │            │
    │            └─→ EpisodeExecutor (单回合执行器)
    │
    └─→ MetricsAggregator (指标聚合器)
          │
          ├─→ RealtimeAggregator (实时聚合)
          └─→ FinalAggregator (最终聚合)
```

#### 3.1.2 模块职责

| 模块 | 职责 |
|------|------|
| **EvaluationOrchestrator** | 加载Benchmark配置、初始化所有子模块、协调评测流程、生成最终报告 |
| **BenchmarkRunner** | 管理单个Benchmark的执行、创建EpisodeManager和MetricsAggregator、控制评测生命周期 |
| **EpisodeManager** | 管理Episode序列、分发Episode到Worker、跟踪执行状态 |
| **ParallelExecutor** | 管理进程池（multiprocessing.Pool）、维护任务队列、负载均衡 |
| **EpisodeExecutor** | 执行单个Episode、管理Simulator和AgentClient交互、记录轨迹和指标 |
| **MetricsAggregator** | 实时收集Episode结果、计算统计量（均值、标准差、分位数）、生成JSON报告 |

#### 3.1.3 核心接口

```
EvaluationOrchestrator:
    def run_benchmark(benchmark_config: BenchmarkConfig) → EvaluationResults

BenchmarkRunner:
    def run(agent_client: AgentClient) → BenchmarkResults

EpisodeManager:
    def submit_episodes(episodes: List[Episode])
    def get_results() → List[EpisodeResults]

MetricsAggregator:
    def add_episode_result(result: EpisodeResults)
    def get_aggregated_metrics() → Dict[str, Statistics]
    def export_json(path: str)
```

#### 3.1.4 状态管理

**评测级别状态**：
```
INIT → RUNNING → COMPLETED
        ↓
      FAILED
```

**Episode级别状态**：
```
PENDING → RUNNING → COMPLETED
            ↓
          FAILED / TIMEOUT
```

#### 3.1.5 错误处理策略

| 错误类型 | 处理方式 |
|----------|----------|
| **Simulator错误** | 标记Episode失败，跳过，记录错误信息 |
| **Agent通信错误** | 重试3次，仍失败则跳过 |
| **超时处理** | 单Episode超时后终止，跳过继续 |
| **数据收集错误** | 跳过该Episode，不影响整体评测 |

#### 3.1.6 进程级并行机制

使用 `multiprocessing.Pool` 实现进程级并行：

- 每个Worker进程拥有：
  - 独立的Simulator实例
  - 独立的AgentClient连接
  - 共享只读的Dataset

#### 3.1.7 结果聚合和导出

**JSON输出格式**：
```json
{
  "benchmark": "VLN Challenge 2024",
  "timestamp": "2024-01-15T10:30:00Z",
  "config": {...},
  "episodes": [
    {
      "episode_id": "ep_001",
      "status": "completed",
      "metrics": {"success": 1.0, "spl": 0.85},
      "trajectory": [...],
      "num_steps": 45
    }
  ],
  "aggregated": {
    "success": {"mean": 0.65, "std": 0.12, "count": 100},
    "spl": {"mean": 0.52, "std": 0.18, "count": 100}
  },
  "failed_episodes": [
    {"episode_id": "ep_050", "reason": "timeout"}
  ]
}
```

### 3.2 仿真引擎 (Simulation Engine)

**职责**：
- 独立仿真器（不使用Habitat-sim）
- 场景加载和管理
- 基于动作更新代理状态
- 传感器观测生成（RGB, Depth, GPS, Instruction）
- 碰撞检测和物理计算
- 测地距离计算

**核心接口**：
```
- reset(episode) → Observations
- step(action) → Observations
- get_agent_state() → AgentState
- geodesic_distance(start, end) → float
- is_navigable(position) → bool
```

**仿真器选项**：
- 自定义Python仿真器
- Unity/C++后端
- 第三方仿真引擎（通过插件）

### 3.3 指标系统 (Metrics System)

**职责**：
- 评测指标计算
- 可扩展的插件架构
- Metric依赖管理
- 单回合和聚合统计

**标准Metrics**：

| Metric | 描述 | 计算公式 |
|------|------|----------|
| **Success** | 是否成功到达目标 | distance_to_goal < threshold ? 1 : 0 |
| **SPL** | 成功加权的路径长度 | Success × (最短路径 / 实际路径) |
| **Soft-SPL** | 放宽成功条件的SPL | max(0, 1 - distance/shortest) × (最短/实际) |
| **Navigation Error** | 停止位置到目标的距离 | Euclidean(agent_final_pos, goal) |
| **DTW** | 动态时间规整距离 | DTW(agent_trajectory, reference_path) |
| **Coverage** | 参考路径覆盖百分比 | covered_points / total_reference_points |

**Metric接口**：
```
- reset_metric(episode, task)
- update_metric(*args, **kwargs)
- get_metric() → value
```

### 3.4 推理服务客户端 (Agent Client)

**职责**：
- 与参赛者推理服务通信
- 协议转换（评测系统 ↔ 参赛者API）
- 重试逻辑和超时处理
- 请求/响应序列化

**支持的协议**：

| 协议 | 优点 | 适用场景 |
|------|------|----------|
| **REST** | 简单易用，易于调试 | 开发测试、小型竞赛 |
| **gRPC** | 高性能、类型安全 | 生产环境、大规模竞赛 |

**核心接口**：
```
- reset_episode(episode) → Observations
- get_action(observations) → Action
- health_check() → Status
```

### 3.5 数据集管理 (Dataset & Episode)

**职责**：
- 回合数据集加载和解析
- 数据集格式验证
- 回合过滤和采样
- 场景缓存管理

**Episode数据结构**（根据领域概念定义）：
```python
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class Episode:
    """
    Episode概念的实现

    根据领域概念定义，Episode是单次评测的完整场景定义，
    包含场景信息、初始状态、任务目标和参考路径。
    """
    episode_id: str
    scene_id: str
    start_position: List[float]  # [x, y, z]
    start_rotation: List[float]  # quaternion [x, y, z, w]
    instruction: Dict           # {"text": str, "tokens": List[str]}
    reference_path: List[List[float]]  # [[x,y,z], ...]
    goals: List[Dict]          # [{"position": [x,y,z], "radius": float}]
```

### 3.6 配置系统 (Configuration System)

**职责**：
- 分层配置（任务、基准、仿真器）
- 配置验证和模式检查
- 覆盖机制（CLI、文件、环境变量）

**配置层次**（基于领域概念定义）：
```
Configuration (配置概念的实现)
├── Benchmark Config (Benchmark概念的参数化)
│   ├─ Task Config (Task概念的参数化)
│   │   ├─ actions: Action Space定义
│   │   ├─ sensors: Sensor Suite配置
│   │   └─ metrics: Metric集合
│   ├─ Dataset Config (Dataset概念的参数化)
│   │   ├─ data_path: Dataset文件路径
│   │   ├─ split: 数据集split
│   │   └─ episodes: Episode数量
│   ├─ Evaluation Config (评测参数，非独立概念)
│   │   ├─ max_steps
│   │   ├─ success_distance
│   │   └─ timeout
│   └─ Output Config (输出配置，非独立概念)
│       ├─ log_dir
│       ├─ save_trajectories
│       └─ save_observations
└── Simulator Config (Simulator概念的参数化)
    ├─ backend: 仿真器后端
    └─ sensors: 传感器配置
```

---

## 4. API规范设计 - WebSocket

### 4.1 连接模式

**长连接 + 心跳机制**：
- 参赛者服务作为WebSocket客户端连接到评测系统
- 心跳间隔30秒，超时60秒
- 连接断开时自动重连

### 4.2 消息类型

| 消息类型 | 方向 | 描述 |
|----------|------|----------|
| **Connect** | Client→Server | 建立连接，发送agent信息 |
| **Connected** | Server→Client | 连接确认，返回session_id |
| **ResetEpisode** | Client→Server | 请求重置episode |
| **EpisodeReady** | Server→Client | Episode重置完成，返回初始观测 |
| **GetAction** | Server→Client | 发送当前观测，请求动作 |
| **Action** | Client→Server | 返回选择的动作 |
| **EpisodeEnd** | Server→Client | Episode结束，发送指标结果 |
| **Heartbeat** | 双向 | 保活消息 |
| **Error** | Server→Client | 错误通知 |
| **Disconnect** | 双向 | 断开连接通知 |

### 4.3 消息格式（JSON）

**连接消息**：
```json
{
  "type": "connect",
  "agent_id": "team_xyz",
  "protocol_version": "1.0"
}
```

**Reset Episode**：
```json
{
  "type": "reset_episode",
  "session_id": "uuid-xxx",
  "episode": {
    "episode_id": "ep_001",
    "instruction": {
      "text": "Walk down the hallway and enter the kitchen",
      "tokens": ["walk", "down", "the", "hallway", ...]
    },
    "start_position": [1.5, 0.0, 2.3],
    "start_rotation": [0.0, 0.0, 0.0, 1.0]
  }
}
```

**Episode Ready**：
```json
{
  "type": "episode_ready",
  "session_id": "uuid-xxx",
  "observation": {
    "rgb": "<base64_encoded>",
    "depth": "<base64_encoded>",
    "instruction": {...},
    "gps": [0.0, 0.0],
    "compass": 0.0
  }
}
```

**Action循环**：
```json
// Server → Client
{
  "type": "get_action",
  "session_id": "uuid-xxx",
  "observation": {
    "rgb": "<base64_encoded>",
    "depth": "<base64_encoded>",
    "instruction": {...}
  }
}

// Client → Server
{
  "type": "action",
  "session_id": "uuid-xxx",
  "action": "move_forward",
  "action_args": {}
}
```

**Episode End**：
```json
{
  "type": "episode_end",
  "session_id": "uuid-xxx",
  "status": "success" | "timeout" | "error",
  "metrics": {
    "success": 1.0,
    "spl": 0.85,
    "navigation_error": 0.15
  },
  "num_steps": 45
}
```

### 4.4 多Episode并发

- 每个Episode使用独立的WebSocket连接
- 评测系统为每个Episode分配唯一session_id
- 参赛者可以同时处理多个episode连接
- 连接池大小可配置

### 4.5 参赛者SDK设计

**SDK组件**：

| 组件 | 功能 |
|------|------|
| **Agent接口** | VLNAgent抽象类，参赛者继承实现 |
| **WebSocket客户端** | 自动处理连接、心跳、重连 |
| **服务器模板** | 评测系统WebSocket服务器实现 |
| **测试工具** | 本地测试工具 |

**类型定义**（基于领域概念）：
```python
from typing import TypedDict, List, Dict, Optional

class Episode(TypedDict):
    """Episode概念的类型定义"""
    episode_id: str
    scene_id: str
    start_position: List[float]  # [x, y, z]
    start_rotation: List[float]  # [x, y, z, w]
    instruction: Dict  # {"text": str, "tokens": List[str]}
    reference_path: List[List[float]]
    goals: List[Dict]

class Observation(TypedDict):
    """Observation概念的类型定义"""
    rgb: str  # base64 encoded
    depth: str  # base64 encoded
    instruction: Dict
    gps: List[float]  # [x, y, z]
    compass: float  # angle in radians

class Action(TypedDict):
    """Action概念的类型定义"""
    action: str
    action_args: Dict

class VLNAgent:
    """VLN Agent接口

    参赛者继承此类并实现reset和act方法。
    """

    def reset(self, episode: Episode) -> None:
        """新回合重置

        Args:
            episode: Episode概念的完整数据结构
        """
        pass

    def act(self, observation: Observation) -> Action:
        """根据观测选择动作

        Args:
            observation: Observation概念的完整数据结构

        Returns:
            Action概念的完整数据结构
        """
        return {'action': 'move_forward', 'action_args': {}}
```

---

## 5. 数据流设计

### 5.1 评测流程

```
┌──────────────┐
│  Dataset     │ 加载回合
│  (.json.gz)  │────────┐
└──────────────┘         │
                        ▼
┌─────────────────────────────────────────────────┐
│           FOR EACH episode:                     │
│                                                 │
│  1. simulator.reset(episode)                    │
│     └─> 加载场景，设置代理位置                   │
│     └─> 返回初始观测                             │
│                                                 │
│  2. agent_client.reset(episode)                 │
│     ├─> REMOTE: POST /episode/reset             │
│     └─> LOCAL: agent.reset()                    │
│                                                 │
│  3. WHILE NOT over AND steps < max:             │
│     a. agent_client.get_action(obs)             │
│     b. simulator.step(action)                   │
│     c. 记录轨迹                                  │
│                                                 │
│  4. 计算回合指标                                │
│                                                 │
└─────────────────────────────────────────────────┘
                        │
                        ▼
              ┌──────────────────┐
              │  聚合所有指标     │
              │  生成结果报告     │
              └──────────────────┘
```

### 5.2 通信流程（WebSocket）

```
┌─────────────┐                    ┌─────────────┐                    ┌─────────────┐
│  仿真器      │                    │  评测系统     │                    │ 参赛者服务   │
│ (Simulator)  │                    │ (Evaluator)  │                    │ (Agent)       │
└──────┬──────┘                    └──────┬──────┘                    └──────┬──────┘
       │                                  │                                  │
       │  simulator.reset(episode)        │                                  │
       │  <───────────────────────────────│                                  │
       │                                  │  (initial_observation)               │
       │                                  │                                  │
       │  WS: send episode_ready          │                                  │
       │  ───────────────────────────────────────────────────────────>│
       │                                  │                          (session_id, obs)      │
       │                                  │                                  │
       │  WS: wait for action           │                                  │
       │  <────────────────────────────────────────────────────────────│
       │                                  │                          (action)                   │
       │                                  │                                  │
       │  simulator.step(action)          │                                  │
       │  <───────────────────────────────│                                  │
       │                                  │                                  │
       │  (next_observation)              │                                  │
       │  ────────────────────────────────>│                                  │
       │                                  │                                  │
       │  [Repeat step loop until episode over]                   │
       │                                  │                                  │
       │  WS: send episode_end          │                                  │
       │  ───────────────────────────────────────────────────────────>│
       │                          (session_id, status, metrics)         │
       │                                  │                          WS: close                │
       │                                  │  ─────────────────────────────────────────────>│
       │                                  │                          WS: connect (next)    │
       │                                  │  <───────────────────────────────────────────
```

**WebSocket消息流向**：
1. 评测系统建立WebSocket服务器
2. 参赛者服务连接（WS: connect）
3. 评测系统发送（WS: episode_ready）初始观测
4. 参赛者返回动作（WS: action）
5. 评测系统执行simulator.step(action)
6. 重复3-5直到episode结束
7. 评测系统发送（WS: episode_end）并关闭连接
8. 参赛者服务可发起新连接（WS: connect）开始下一episode

---

## 6. 可扩展性设计

### 6.1 任务扩展机制

通过**任务注册表**添加新任务：

```
内置任务:
- VLN (Vision-Language Navigation)
- ObjectNav (Object Navigation)
- ImageNav (Image-Guided Navigation)

未来可扩展:
- RoomNav
- Multi-ObjectNav
- Social Navigation
```

**任务定义要素**：
- 动作空间（离散/连续）
- 观测空间（传感器配置）
- 评测指标集合

### 6.2 指标扩展机制

通过**指标注册表**添加新指标：

```
注册装饰器: @register_metric("metric_name")

指标类别:
- 导航指标（Success, SPL, DTW）
- 效率指标（Time, Steps）
- 安全指标（Collisions）
- 可视化指标（Trajectory）
```

### 6.3 仿真器扩展机制

通过**仿真器注册表**支持多种后端：

```
支持后端:
- Custom (自定义Python实现)
- Unity (Unity引擎)
- External (第三方仿真器)

扩展点:
- 场景格式
- 传感器类型
- 物理引擎
```

---

## 7. 目录结构设计

```
nav_eval/
├── core/                   # 核心组件
│   ├── interfaces         # 抽象接口定义
│   ├── environment        # 评测环境
│   ├── orchestrator      # 评测编排器
│   └── registry          # 组件注册表
│
├── simulator/             # 仿真引擎
│   ├── base              # 仿真器抽象类
│   ├── scene             # 场景管理
│   ├── sensors           # 传感器套件
│   └── backends          # 不同后端实现
│
├── metrics/               # 指标系统
│   ├── base              # 指标抽象类
│   ├── navigation        # 导航指标
│   ├── trajectory        # 轨迹指标
│   └── registry          # 指标注册表
│
├── api/                   # 通信接口
│   ├── base              # 客户端抽象类
│   ├── grpc              # gRPC实现
│   ├── rest              # REST实现
│   └── local             # 本地实现
│
├── dataset/               # 数据管理
│   ├── episode           # Episode数据类
│   ├── dataset           # Dataset类
│   └── loaders           # 数据加载器
│
├── tasks/                 # 任务定义
│   ├── base              # 任务抽象类
│   ├── vln               # VLN任务
│   └── registry          # 任务注册表
│
├── config/                # 配置系统
│   ├── config            # 配置类
│   └── loader            # 配置加载器
│
├── benchmarks/            # 基准定义
│   └── registry          # 基准注册表
│
└── utils/                 # 工具库
    ├── geometry          # 几何计算
    ├── visualization     # 可视化
    └── logging           # 日志

nav_eval_sdk/             # 参赛者SDK
├── agent_interface       # Agent接口
├── server               # 服务端脚手架
├── client               # 测试客户端
└── examples             # 示例代码

configs/                 # 配置文件
├── tasks/               # 任务配置
├── benchmarks/          # 基准配置
└── simulator/           # 仿真器配置

data/                    # 数据目录
└── datasets/            # 数据集

tests/                   # 测试套件
└── ...

examples/                # 使用示例
└── ...
```

### 目录结构与领域概念的映射

| 代码目录 | 对应概念 | 说明 |
|----------|----------|------|
| `tasks/` | Task | Task概念实现 |
| `benchmarks/` | Benchmark | Benchmark概念实现 |
| `dataset/` | Dataset, Episode | Dataset和Episode概念实现 |
| `simulator/` | Simulator, Sensor | Simulator和Sensor概念实现 |
| `metrics/` | Metric | Metric概念实现 |
| `api/` | Agent, Action, Observation | Agent通信和数据结构 |
| `config/` | Configuration | Configuration概念实现 |

---

## 8. 关键设计决策

### 8.1 API协议选择

| 方案 | 选择 | 理由 |
|------|------|------|
| REST vs gRPC | **两者都支持** | REST降低门槛，gRPC提高性能 |

### 8.2 仿真器实现

| 方案 | 选择 | 理由 |
|------|------|------|
| 使用Habitat-sim | ❌ | 失去独立性 |
| 独立实现 | ✅ | 完全控制，公平竞争 |

### 8.3 动作空间

| 方案 | 选择 | 理由 |
|------|------|------|
| 离散动作 | ✅ 默认 | VLN标准，易理解 |
| 连续控制 | ✅ 可选 | 高级用户 |

### 8.4 指标计算时机

| 方案 | 选择 | 理由 |
|------|------|------|
| 在线计算 | ✅ 采用 | 及时报错 |
| 后处理 | ✅ 采用 | 复杂指标 |

### 8.5 并行评测

| 方案 | 选择 | 理由 |
|------|------|------|
| VectorEnv | ✅ | 高效，符合RL习惯 |

---

## 9. 配置示例

```yaml
# VLN Challenge 基准配置
benchmark:
  name: "VLN Challenge 2024"
  description: "Vision-Language Navigation on R2R"

task:
  type: "vln"
  actions: ["stop", "move_forward", "turn_left", "turn_right"]
  metrics: ["success", "spl", "navigation_error", "dtw"]

simulator:
  backend: "custom"
  sensors:
    rgb: {width: 640, height: 480}
    depth: {width: 640, height: 480}
    instruction: {}
    gps: {}
    compass: {}

dataset:
  type: "vln"
  data_path: "data/datasets/vln/R2R/val_seen.json.gz"

evaluation:
  max_steps: 500
  success_distance: 0.2

agent_service:
  type: "remote"
  protocol: "grpc"
  endpoint: "localhost:8085"
  timeout: 30

output:
  log_dir: "logs/evaluations"
  save_trajectories: true
```

---

## 10. 与Habitat的对比

| 方面 | Habitat | 本系统 |
|------|---------|--------|
| 仿真引擎 | Habitat-sim (C++) | 独立实现 |
| Agent运行 | 同进程/Docker | 独立服务（API） |
| 评测模式 | Local + Remote (gRPC) | Remote (REST/gRPC) |
| 扩展性 | Registry模式 | Registry模式 |
| 配置系统 | Hydra | Hydra风格 |
| 数据格式 | JSON.gz | JSON.gz（兼容） |

**设计借鉴**：
- ✅ Measure接口模式（reset/update/get）
- ✅ Episode数据格式
- ✅ Registry注册机制
- ✅ 分层配置系统

**设计差异**：
- ✅ 独立仿真引擎
- ✅ 参赛者服务化部署
- ✅ REST + gRPC双协议
- ✅ 更简洁的SDK接口

---

## 11. 实现路线图

### Phase 1: 核心基础设施 (优先级最高)
- 定义核心抽象接口
- 实现配置系统
- 数据集加载器

### Phase 2: 仿真和指标
- 仿真器框架
- 标准指标实现
- 指标注册表

### Phase 3: API和SDK
- REST/gRPC客户端
- 参赛者SDK
- 服务端脚手架

### Phase 4: 评测引擎
- 评测编排器
- 并行评测支持
- 结果导出

### Phase 5: 高级功能
- 可视化工具
- 性能优化
- Docker部署

### Phase 6: 测试和文档
- 集成测试
- 用户文档
- 示例代码

---

## 12. 验证计划

### 12.1 单元测试
- [ ] 指标计算正确性
- [ ] 数据加载和解析
- [ ] API序列化/反序列化

### 12.2 集成测试
- [ ] 端到端评测流程
- [ ] API通信测试
- [ ] 配置加载测试

### 12.3 性能测试
- [ ] 并行评测吞吐量
- [ ] API调用延迟
- [ ] 大规模数据集评测

### 12.4 真实场景测试
- [ ] 示例Agent实现
- [ ] VLN数据集评测
- [ ] 结果可视化验证

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

**职责**：
- 管理评测生命周期（初始化、执行、结算）
- 协调仿真器和代理客户端
- 控制回合序列和状态管理
- 聚合跨回合指标
- 处理超时和错误

**核心功能**：
```
- run_evaluation(agent_client) → EvaluationResults
- run_episode(episode) → EpisodeResults
- aggregate_metrics(episode_results) → Dict
- export_results(results) → JSON/CSV
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
- 指标依赖管理
- 单回合和聚合统计

**标准指标**：

| 指标 | 描述 | 计算公式 |
|------|------|----------|
| **Success** | 是否成功到达目标 | distance_to_goal < threshold ? 1 : 0 |
| **SPL** | 成功加权的路径长度 | Success × (最短路径 / 实际路径) |
| **Soft-SPL** | 放宽成功条件的SPL | max(0, 1 - distance/shortest) × (最短/实际) |
| **Navigation Error** | 停止位置到目标的距离 | Euclidean(agent_final_pos, goal) |
| **DTW** | 动态时间规整距离 | DTW(agent_trajectory, reference_path) |
| **Coverage** | 参考路径覆盖百分比 | covered_points / total_reference_points |

**指标接口**：
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

**Episode数据结构**：
```
episode_id: str
scene_id: str
start_position: [x, y, z]
start_rotation: [x, y, z, w]  # quaternion
instruction:
  - text: str
  - tokens: List[str]
reference_path: List[[x, y, z]]
goals: List[{position, radius}]
```

### 3.6 配置系统 (Configuration System)

**职责**：
- 分层配置（任务、基准、仿真器）
- 配置验证和模式检查
- 覆盖机制（CLI、文件、环境变量）

**配置层次**：
```
Benchmark Config
  ├─ Task Config (actions, measurements, sensors)
  ├─ Simulator Config (backend, agent, sensors)
  ├─ Dataset Config (path, split, scenes)
  └─ Evaluation Config (max_steps, timeout)
```

---

## 4. API规范设计

### 4.1 REST API规范

**端点设计**：

| 端点 | 方法 | 功能 |
|------|------|------|
| `/api/v1/episode/reset` | POST | 重置回合，获取初始观测 |
| `/api/v1/agent/act` | POST | 提交观测，获取动作 |
| `/api/v1/episode/validate` | POST | 健康检查 |

**请求/响应格式**：
```
POST /api/v1/episode/reset
Request:
{
  "episode_id": "ep_001",
  "scene_id": "scene_123",
  "instruction": {
    "text": "Walk down the hallway and enter the kitchen",
    "tokens": ["walk", "down", "the", "hallway", ...]
  },
  "start_position": [1.5, 0.0, 2.3],
  "start_rotation": [0.0, 0.0, 0.0, 1.0]
}

Response:
{
  "status": "ready",
  "initial_observation": {
    "rgb": "<base64_encoded>",
    "depth": "<base64_encoded>",
    "instruction": {...},
    "gps": [0.0, 0.0],
    "compass": 0.0
  }
}
```

### 4.2 gRPC协议规范

**服务定义**：
```
service InferenceService {
  rpc ResetEpisode(EpisodeData) returns (InitialObservation);
  rpc GetAction(Observation) returns (Action);
  rpc HealthCheck(Empty) returns (ServiceStatus);
}
```

**消息类型**：
```
- EpisodeData: 回合数据
- Instruction: 语言指令
- Observation: 传感器观测
- Action: 代理动作
- ServiceStatus: 服务状态
```

### 4.3 参赛者SDK设计

**SDK组件**：

| 组件 | 功能 |
|------|------|
| **Agent接口** | VLNAgent抽象类，参赛者继承实现 |
| **服务端脚手架** | Flask/gRPC服务器模板 |
| **测试客户端** | 本地测试工具 |
| **示例实现** | Random Agent, 示例模型 |

**Agent接口定义**：
```
class VLNAgent:
    def reset(episode: Dict) → None
        # 新回合重置

    def act(observation: Dict) → Dict
        # 根据观测选择动作
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

### 5.2 通信流程

```
┌─────────────┐                    ┌─────────────┐                    ┌─────────────┐
│  仿真器      │                    │  评测系统     │                    │ 参赛者服务   │
│ (Simulator)  │                    │             │                    │             │
└──────┬──────┘                    └──────┬──────┘                    └──────┬──────┘
       │                                  │                                  │
       │                                  │  POST /episode/reset             │
       │                                  │  (episode_data) ────────────────>│
       │                                  │                                  │
       │                                  │  <─────────────────────────  (ready)
       │                                  │                                  │
       │  simulator.reset(episode)        │                                  │
       │  <───────────────────────────────│                                  │
       │                                  │                                  │
       │  (initial_observation)           │                                  │
       │  ────────────────────────────────>│                                  │
       │                                  │                                  │
       │                                  │  POST /agent/act                 │
       │                                  │  (observation) ─────────────────>│
       │                                  │                                  │
       │                                  │  <─────────────────────────  (action)
       │                                  │                                  │
       │  simulator.step(action)          │                                  │
       │  <───────────────────────────────│                                  │
       │                                  │                                  │
       │  (next_observation)              │                                  │
       │  ────────────────────────────────>│                                  │
       │                                  │                                  │
       │            [Repeat step loop until episode over]                   │
       │                                  │                                  │
       │                                  │  POST /episode/reset             │
       │                                  │  (next_episode) ────────────────>│
       │                                  │                                  │
```

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
  measurements: ["success", "spl", "navigation_error", "dtw"]

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

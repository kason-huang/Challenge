# 导航模型评测流程时序图文档

## Overview

此目录包含云机器人评测平台导航模型评测流程的时序图设计文档。时序图详细描述了评测系统各个组件之间的交互流程，包括整体评测生命周期和单个Episode的详细交互。

---

## Document Index

| Document | Description |
|----------|-------------|
| [01_overall_evaluation_flow.md](./01_overall_evaluation_flow.md) | 整体评测流程时序图 - 展示从评测任务启动到结果输出的完整生命周期 |
| [02_single_episode_interaction.md](./02_single_episode_interaction.md) | 单Episode详细交互时序图 - 展示单个Episode内的详细推理-仿真交互循环 |

---

## System Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Evaluation System                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌──────────────┐         ┌──────────────┐         ┌──────────────┐ │
│  │   Evaluator  │───────>│  Inference   │         │  Simulator   │ │
│  │   (主控)     │<───────│  Service     │         │  Service     │ │
│  │              │         │  (模型服务)  │         │  (仿真引擎)  │ │
│  └──────────────┘         └──────────────┘         └──────────────┘ │
│         │                       ▲                       ▲          │
│         │                       │                       │          │
│         └───────────────────────┴───────────────────────┘          │
│                  调用推理获取动作      调用仿真执行动作              │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Role | Responsibilities |
|-----------|------|------------------|
| **Evaluator** | 主控服务 | • 任务编排<br/>• Episode加载<br/>• 指标计算<br/>• 异常处理<br/>• 结果汇总 |
| **Inference Service** | 推理服务 | • 接收Observation<br/>• 执行模型推理<br/>• 返回Action |
| **Simulator Service** | 仿真服务 | • 场景加载与渲染<br/>• 物理仿真<br/>• 传感器数据生成<br/>• 碰撞检测 |

---

## Evaluation Flow Summary

### Phase 1: Initialization

```
User → Evaluator: Start evaluation task
Evaluator → Dataset: Load episodes
Evaluator → Inference: Connect service
Evaluator → Simulator: Connect service
```

### Phase 2: Episode Loop

For each episode in dataset:

```
Evaluator → Simulator: Load scene and robot
Evaluator → Simulator: Set initial state

Loop (until termination):
  Evaluator → Inference: Send observation, get action
  Evaluator → Simulator: Execute action, get new observation
  Check termination conditions

Evaluator: Compute metrics
Evaluator: Record results
```

### Phase 3: Result Aggregation

```
Evaluator: Aggregate all metrics
Evaluator: Generate evaluation report
Evaluator → User: Return results
```

---

## Key Data Structures

### Episode Definition
```json
{
  "episode_id": "ep_001",
  "scene_id": "hm3d_v1_apartment_1",
  "start_state": {"position": [...], "rotation": [...]},
  "goals": [{"object_id": "chair_123"}],
  "instruction": {"text": "Walk to the kitchen"},
  "max_steps": 500
}
```

### Observation
```json
{
  "rgb": "base64_encoded_image",
  "depth": "base64_encoded_depth",
  "instruction": {"text": "..."},
  "gps": [x, y, z],
  "compass": angle
}
```

### Action
```json
{
  "name": "move_forward",  // or "turn_left", "turn_right", "stop"
  "metadata": {}
}
```

### Metrics
```json
{
  "episode_id": "ep_001",
  "success": true,
  "spl": 0.85,
  "distance_to_goal": 0.15,
  "num_steps": 127,
  "execution_time": 12.5
}
```

---

## Action Space

| Action | Description | Default Parameter |
|--------|-------------|-------------------|
| `move_forward` | Move robot forward along current heading | distance = 0.25m |
| `turn_left` | Rotate robot counter-clockwise | angle = 15° |
| `turn_right` | Rotate robot clockwise | angle = 15° |
| `stop` | Stop and terminate episode | - |

---

## Termination Conditions

Priority order (highest first):

1. **Agent Stop**: `action.name == "stop"`
2. **Goal Reached**: `distance_to_goal < threshold` (typically 0.2m)
3. **Max Steps**: `step_count >= max_steps`
4. **Collision**: Robot collides with obstacle (optional)

---

## Evaluation Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| **Success** | Binary | Whether agent reached the goal |
| **SPL** | `success × (shortest_path / actual_path)` | Success weighted by Path Length |
| **Distance to Goal** | Euclidean distance | Final distance to goal |
| **Num Steps** | Count | Total steps taken |
| **Execution Time** | Seconds | Episode duration |

---

## Error Handling

| Error Type | Handling Strategy |
|------------|-------------------|
| **Inference connection failed** | Retry 3 times with exponential backoff, then abort |
| **Simulator connection failed** | Retry 3 times, then abort |
| **Scene load failed** | Skip episode, log error, continue to next |
| **Inference timeout** | Use default action `stop`, terminate episode |
| **Simulation crash** | Mark episode failed, restart simulator, continue |
| **Metrics computation failed** | Use default values, log warning |

---

## Performance Specifications

| Parameter | Value |
|-----------|-------|
| **Action timeout** | 5 seconds |
| **Observation size** | RGB ~30KB, Depth ~15KB (compressed) |
| **Target latency** | < 100ms per action |
| **Max steps per episode** | 500 (configurable) |
| **Episode execution mode** | Serial (parallel planned for future) |

---

## Viewing the Diagrams

### Option 1: Markdown Renderers

The diagrams use Mermaid format and can be rendered in:
- GitHub/GitLab (native support)
- VS Code (with Mermaid preview extension)
- Notion, Confluence, etc.

### Option 2: Online Tools

- [Mermaid Live Editor](https://mermaid.live)
- Copy the diagram code and paste for interactive preview

### Option 3: Command Line

```bash
# Install mermaid-cli
npm install -g @mermaid-js/mermaid-cli

# Render diagram to PNG
mmdc -i 01_overall_evaluation_flow.md -o flow_diagram.png
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-03-11 | Initial design document |

---

## Related Documentation

- [../../RoboVerse-docs/任务执行信息表示方案.md](../../RoboVerse-docs/任务执行信息表示方案.md) - 任务执行信息表示方案
- [../../docs/task_interface_design.md](../../docs/task_interface_design.md) - 任务接口设计

---

## Future Extensions

1. **Parallel Episode Execution**: Support concurrent episode execution for faster evaluation
2. **Multi-Agent Scenarios**: Extend to support multiple agents in collaborative tasks
3. **Real-time Monitoring**: Add progress tracking and intermediate status queries
4. **Checkpoint/Resume**: Support resuming evaluation from interrupted episodes
5. **Custom Metrics**: Allow user-defined evaluation metrics

---

*Last updated: 2026-03-11*

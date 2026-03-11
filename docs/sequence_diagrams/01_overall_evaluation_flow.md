# 整体评测流程时序图

## Overview

此图展示了导航模型评测的完整生命周期，从评测任务启动到结果输出的全流程。

## System Participants

| Participant | Description |
|-------------|-------------|
| **User/Scheduler** | 用户或调度器，发起评测任务 |
| **Evaluator** | 评测器服务，主控服务，负责任务编排和指标计算 |
| **Dataset** | 任务集，包含所有待评测的Episode |
| **Inference** | 推理服务，参赛者部署的模型服务 |
| **Simulator** | 仿真器服务，场景渲染和物理仿真 |
| **Logger** | 日志系统，记录错误和运行日志 |

## Sequence Diagram

```mermaid
sequenceDiagram
    autonumber
    participant User as User/Scheduler
    participant Evaluator as Evaluator
    participant Dataset as Dataset
    participant Inference as Inference Service
    participant Simulator as Simulator Service
    participant Logger as Logger

    rect rgb(220, 240, 255)
        Note over User,Logger: Phase 1: Initialization
        User->>Evaluator: Start evaluation task(config)
        activate Evaluator
        Note right of Evaluator: config contains:<br/>- dataset_path<br/>- inference_url<br/>- simulator_url<br/>- max_episodes

        Evaluator->>Dataset: load_episodes(data_path)
        activate Dataset
        Dataset-->>Evaluator: Episode list
        deactivate Dataset
        Note right of Evaluator: Returns N episodes

        Evaluator->>Inference: connect(inference_url)
        activate Inference
        alt Connection successful
            Inference-->>Evaluator: connected
        else Connection failed
            Inference--xEvaluator: connection error
            Evaluator->>Logger: log("Failed to connect inference service")
            Evaluator->>Logger: log("Retry connection (attempt 1/3)")
            Evaluator->>Inference: reconnect(inference_url)
            alt Still failed after 3 attempts
                Evaluator--xUser: Return error: Cannot connect to inference service
            end
        end
        deactivate Inference

        Evaluator->>Simulator: connect(simulator_url)
        activate Simulator
        Simulator-->>Evaluator: connected
        deactivate Simulator

        Evaluator->>Evaluator: Initialize metrics collector
        Note right of Evaluator: Create metrics storage<br/>for all episodes
    end

    rect rgb(240, 255, 240)
        Note over User,Logger: Phase 2: Episode Loop (Serial Execution)
        loop For each episode in dataset
            Note over Evaluator,Simulator: Episode Initialization
            Evaluator->>Simulator: load_scene(scene_id)
            activate Simulator
            alt Scene load successful
                Simulator-->>Evaluator: scene_loaded
            else Scene load failed
                Simulator--xEvaluator: scene_load_error
                Evaluator->>Logger: log_error(episode_id, error)
                Evaluator->>Evaluator: Mark episode as FAILED
                Note right of Evaluator: Skip to next episode
            end
            deactivate Simulator

            alt Episode not skipped
                Evaluator->>Simulator: load_robot(robot_config)
                Simulator-->>Evaluator: robot_loaded

                Evaluator->>Simulator: set_initial_state(start_state)
                Simulator-->>Evaluator: state_set

                Note over Evaluator,Simulator: Execute Episode (see detailed diagram)
                alt Episode execution successful
                    Evaluator->>Simulator: get_final_state()
                    Simulator-->>Evaluator: final_state
                    Evaluator->>Evaluator: compute_metrics(episode_data)
                    Note right of Evaluator: Calculate:<br/>- Success<br/>- SPL<br/>- Distance to goal<br/>- Num steps
                    Evaluator->>Evaluator: record_results(episode_id, metrics)
                else Episode execution failed
                    Evaluator->>Logger: log_error(episode_id, error_details)
                    Note right of Logger: Error types:<br/>- Inference timeout<br/>- Simulation crash<br/>- Invalid action
                    Evaluator->>Evaluator: Mark episode as FAILED
                    Evaluator->>Evaluator: Record error metadata
                end
            end
        end
    end

    rect rgb(255, 245, 230)
        Note over User,Logger: Phase 3: Result Aggregation
        Evaluator->>Evaluator: aggregate_metrics()
        Note right of Evaluator: Compute:<br/>- Average success rate<br/>- Average SPL<br/>- Success/Failed count<br/>- Total execution time

        Evaluator->>Logger: generate_evaluation_report()
        activate Logger
        Note right of Logger: Report includes:<br/>- Per-episode results<br/>- Aggregate metrics<br/>- Error summary<br/>- Execution statistics
        Logger-->>Evaluator: report_generated
        deactivate Logger

        Evaluator-->>User: Return evaluation results
        deactivate Evaluator
        Note right of User: Results format:<br/>{<br/>  total_episodes: N,<br/>  successful: M,<br/>  failed: N-M,<br/>  metrics: {...},<br/>  report_path: "..."<br/>}
    end
```

## Data Structures

### Input Configuration
```json
{
  "dataset_path": "/data/datasets/objectnav/hm3d_v1/",
  "inference_url": "http://inference-service:8080",
  "simulator_url": "http://simulator-service:9000",
  "max_episodes": 100,
  "max_steps_per_episode": 500,
  "timeout_per_action": 5.0
}
```

### Evaluation Results
```json
{
  "total_episodes": 100,
  "successful": 85,
  "failed": 15,
  "metrics": {
    "success_rate": 0.85,
    "average_spl": 0.72,
    "average_distance_to_goal": 0.35,
    "average_steps": 234.5,
    "total_execution_time": 1250.8
  },
  "failed_episodes": [
    {
      "episode_id": "ep_023",
      "reason": "Inference timeout",
      "step": 127
    },
    {
      "episode_id": "ep_056",
      "reason": "Simulation crash",
      "step": 45
    }
  ],
  "report_path": "/results/evaluation_report_20260311_143052.json"
}
```

## Error Handling Details

| Error Type | Handling Strategy |
|------------|-------------------|
| **Inference connection failed** | Retry 3 times with exponential backoff, then abort |
| **Simulator connection failed** | Retry 3 times, then abort |
| **Scene load failed** | Skip episode, log error, continue to next |
| **Inference timeout** | Use default action (stop), terminate episode |
| **Simulation crash** | Mark episode failed, restart simulator, continue |
| **Metrics computation failed** | Use default values, log warning |

## Key Messages

| Message | Source → Destination | Description |
|---------|---------------------|-------------|
| `load_episodes(path)` | Evaluator → Dataset | Load episode dataset from disk |
| `connect(url)` | Evaluator → Services | Establish connection to services |
| `load_scene(scene_id)` | Evaluator → Simulator | Load 3D scene for episode |
| `load_robot(config)` | Evaluator → Simulator | Load robot configuration |
| `set_initial_state(state)` | Evaluator → Simulator | Set robot start position |
| `compute_metrics(data)` | Evaluator → Evaluator | Internal metrics calculation |
| `aggregate_metrics()` | Evaluator → Evaluator | Aggregate all episode metrics |
| `generate_report()` | Evaluator → Logger | Generate evaluation report |

## Notes

- **Serial Execution**: Episodes are executed one by one to ensure consistent resource usage
- **Fault Tolerance**: Failed episodes do not stop the entire evaluation process
- **Logging**: All errors are logged with sufficient context for debugging
- **Timeout**: Each action has a timeout to prevent hanging
- **Resource Cleanup**: Simulator is reset between episodes

## Related Documents

- [02_single_episode_interaction.md](./02_single_episode_interaction.md) - Detailed single episode interaction
- [README.md](./README.md) - Sequence diagrams overview

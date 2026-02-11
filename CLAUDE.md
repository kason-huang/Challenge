# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains the Habitat ecosystem for embodied AI research, consisting of three main components:

1. **habitat-sim** - High-performance 3D simulator with physics support (C++/Python)
2. **habitat-lab** - High-level library for embodied AI experiments (Python)
3. **habitat-challenge** - Competition framework for navigation tasks (Python)

Habitat is developed by Meta AI Research and focuses on high-speed simulation (thousands of FPS) for tasks like object navigation, image navigation, rearrangement, and human-robot interaction.

## Environment Setup

### Installation

```bash
# Create conda environment
conda create -n habitat python=3.9 cmake=3.14.0
conda activate habitat

# Install habitat-sim with bullet physics (recommended)
conda install habitat-sim withbullet -c conda-forge -c aihabitat

# Install habitat-lab (from the habitat-lab directory)
pip install -e habitat-lab/habitat-lab

# Install habitat-baselines (for RL training)
pip install -e habitat-lab/habitat-baselines
```

### Download Test Data

```bash
# Download test scenes
python -m habitat_sim.utils.datasets_download --uids habitat_test_scenes --data-path data/

# Download example objects
python -m habitat_sim.utils.datasets_download --uids habitat_example_objects --data-path data/

# Download ReplicaCAD dataset for physics interactions
python -m habitat_sim.utils.datasets_download --uids replica_cad_dataset
```

## Architecture

### Three-Component Structure

```
habitat-sim/          # Core simulator (C++ engine, physics, rendering)
  ├── src/            # C++ source code
  ├── src_python/     # Python bindings
  └── examples/       # Example applications

habitat-lab/          # High-level library and training
  ├── habitat-lab/    # Core library (tasks, datasets, gym interface)
  └── habitat-baselines/  # RL algorithms (PPO, imitation learning)

habitat-challenge/    # Competition framework
  ├── agents/         # Agent implementations for challenges
  ├── configs/        # Challenge configurations
  └── docker/         # Docker evaluation setup
```

### Configuration System

Habitat uses **Hydra** for configuration management. Configs are structured hierarchically:

- **Task configs**: `habitat-lab/habitat-lab/habitat/config/habitat/task/` - Define embodied AI tasks
- **Benchmark configs**: `habitat-lab/habitat-lab/habitat/config/benchmark/` - Primary configs for experiments
- **Baselines configs**: `habitat-lab/habitat-baselines/habitat_baselines/config/` - Training configurations

Load and modify configs:
```python
import habitat

# Load config
config = habitat.get_config("benchmark/nav/pointnav/pointnav_gibson.yaml")

# Override via code
from habitat.config.read_write import read_write
with read_write(config):
    config.habitat.simulator.concur_render = False
```

Override via command line:
```bash
python script.py --config-name=config.yaml habitat.environment.max_episode_steps=250
```

### Key Abstractions

- **Agent**: Embodied entity with sensors and actuators (robots like Fetch, Spot Stretch, humanoids)
- **Task**: Defines the objective (ObjectNav, ImageNav, PointNav, Rearrangement)
- **Sensor**: Provides observations (RGB, Depth, Semantic, GPS+Compass)
- **Measurement**: Evaluates performance (SPL, Success, Distance to Goal)
- **Action Space**: Discrete (move_forward, turn_left) or continuous (velocity_control)

## Common Development Commands

### Testing Installation

```bash
# Interactive viewer (habitat-sim directory)
python examples/viewer.py --scene data/scene_datasets/habitat-test-scenes/skokloster-castle.glb

# Run example (habitat-lab directory)
python examples/example.py

# Interactive robot control (requires pygame, pybullet)
python examples/interactive_play.py --never-end
```

### Training with RL

```bash
# Single machine training (from habitat-lab directory)
python -u -m habitat_baselines.run --config-name=pointnav/ppo_pointnav_example.yaml

# DD-PPO training for navigation tasks
python -u -m habitat_baselines.run --config-name=configs/ddppo_objectnav_v2_hm3d_stretch.yaml

# Evaluate trained model
python -u -m habitat_baselines.run \
  --config-name=configs/ddppo_objectnav_v2_hm3d_stretch.yaml \
  habitat_baselines.evaluate=True \
  habitat_baselines.eval_ckpt_path_dir=$PATH_TO_CHECKPOINT
```

### Running Tests

```bash
# Set debug mode for verbose error messages in vectorized environments
export HABITAT_ENV_DEBUG=1

# Run tests (from respective directories)
pytest habitat-lab/test/
```

### Challenge Submission

```bash
# Build docker for challenge
docker build . --file docker/ObjectNav_random_baseline.Dockerfile -t objectnav_submission

# Test locally
./scripts/test_local_objectnav.sh --docker-name objectnav_submission

# Submit to EvalAI
evalai push objectnav_submission:latest --phase <phase-name>
```

## Important Concepts

### Action Spaces

1. **Continuous Velocity** (HelloRobot Stretch):
   - `linear_velocity`, `angular_velocity`, `camera_pitch_velocity`, `velocity_stop`

2. **Waypoint Controller**: Higher-level abstraction for navigation
   - `xyt_waypoint`, `max_duration`, `delta_camera_pitch_angle`

3. **Discrete**: Traditional discrete actions
   - `move_forward`, `turn_left`, `turn_right`, `stop`

### Multi-Agent Training

Habitat supports multi-agent scenarios (Habitat 3.0) for social navigation and rearrangement:
- Multiple agents with different policies
- Human-robot collaboration tasks
- Configured via `benchmark/multi_agent/` configs

### Performance Considerations

- Use `VectorEnv` for fast parallel environments
- Use `ThreadedVectorEnv` (enabled via `HABITAT_ENV_DEBUG=1`) for debugging
- Habitat achieves >10,000 FPS multi-process on single GPU
- Physics simulation at 8,000+ steps per second

### Dataset Locations

- HM3D: `data/scene_datasets/hm3d_v0.2/`
- ReplicaCAD: `data/replica_cad/`
- Episode datasets: `data/datasets/{task_name}/{dataset}/`

## Key File Locations

- Task definitions: `habitat-lab/habitat-lab/habitat/tasks/`
- RL algorithms: `habitat-lab/habitat-baselines/habitat_baselines/rl/`
- Imitation learning: `habitat-lab/habitat-baselines/habitat_baselines/il/`
- Gym wrapper: `habitat-lab/habitat-lab/habitat/gym.py`
- Structured configs: `habitat-lab/habitat-lab/habitat/config/default_structured_configs.py`

## Datasets

Common datasets used with Habitat:
- **HM3D** (Habitat Matterport3D): Photorealistic indoor scenes
- **HM3D-Semantics**: HM3D with semantic annotations
- **ReplicaCAD**: Articulated objects for manipulation tasks
- **Gibson**: Real-world scanned environments
- **MatterPort3D**: Large-scale indoor scans

Download via: `python -m habitat_sim.utils.datasets_download --uids <dataset_name>`

## References

- Documentation: https://aihabitat.org/docs/
- Tutorials: https://aihabitat.org/tutorial/2020/
- Community: https://github.com/facebookresearch/habitat-lab/discussions

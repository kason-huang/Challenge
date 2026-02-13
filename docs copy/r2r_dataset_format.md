# R2R Dataset Format

R2R (Room-to-Room) is a Vision-and-Language Navigation dataset that requires agents to navigate 3D indoor environments following natural language instructions.

## Directory Structure

```
data/r2r/
├── embeddings.json.gz          # Instruction embeddings (2504 x 50)
├── train/                      # Training set
│   ├── train.json.gz
│   └── train_gt.json.gz
├── val_seen/                   # Validation set (seen scenes)
│   ├── val_seen.json           # Uncompressed, 1.4MB
│   └── val_seen_gt.json.gz
├── val_unseen/                 # Validation set (unseen scenes)
│   ├── val_unseen.json.gz
│   └── val_unseen_gt.json.gz
├── test/                       # Test set
│   └── test.json.gz
├── envdrop/                    # Environment augmented dataset
│   ├── envdrop.json.gz
│   └── envdrop_gt.json.gz
└── joint_train_envdrop/        # Joint training data
    ├── joint_train_envdrop
    └── joint_train_envdrop_gt.json.gz
```

---

## 1. Episodes JSON Format (`val_seen.json`, etc.)

### Top-level Structure

```json
{
  "episodes": [...],      // List of episodes
  "instruction_vocab": {}  // Vocabulary for instructions
}
```

### Single Episode Structure

| Field | Type | Description |
|-------|------|-------------|
| `episode_id` | int | Unique episode identifier |
| `trajectory_id` | int | Trajectory ID (multiple instructions can share the same path) |
| `scene_id` | string | Scene file path (e.g., `mp3d/s8pcmisQ38h/s8pcmisQ38h.glb`) |
| `start_position` | [x, y, z] | Starting position in meters |
| `start_rotation` | [w, x, y, z] | Starting rotation as quaternion |
| `info.geodesic_distance` | float | Shortest path distance to goal |
| `goals` | list | Goal positions `[{"position": [x,y,z], "radius": r}]` |
| `instruction.instruction_text` | string | Natural language navigation instruction |
| `instruction.instruction_tokens` | [int] | Token sequence (fixed length 200, 0-padded) |
| `reference_path` | [[x,y,z],...] | List of waypoints for reference path |

### Example

```json
{
  "episode_id": 1,
  "trajectory_id": 37,
  "scene_id": "mp3d/s8pcmisQ38h/s8pcmisQ38h.glb",
  "start_position": [5.582849979400635, -1.6265579462051392, 2.817889928817749],
  "start_rotation": [-0.0, 0.9659258262890683, -0.0, -0.25881904510252063],
  "info": {"geodesic_distance": 10.39185905456543},
  "goals": [{"position": [11.632800102233887, -3.1616740226745605, 1.9992599487304688], "radius": 3.0}],
  "instruction": {
    "instruction_text": "Walk down the stairs, turn right, and walk towards place with a rug. Wait near the bench and piano along the right side of the wall.",
    "instruction_tokens": [2384, 717, 2202, 2058, 2300, 1819, 103, ...]
  },
  "reference_path": [
    [5.582849979400635, -1.6265579462051392, 2.817889928817749],
    [4.822649955749512, -2.2173328399658203, 2.7935900688171387],
    ...
  ]
}
```

---

## 2. Ground Truth JSON Format (`*_gt.json.gz`)

**Structure:** Dictionary with `episode_id` as keys

```json
{
  "457": {
    "locations": [[x,y,z], [x,y,z], ...],  // Trajectory coordinate sequence
    "actions": [2, 2, 2, 1, 1, ...],        // Action sequence
    "forward_steps": 32                     // Forward step count
  },
  "731": {...},
  ...
}
```

### Action Encoding

| Code | Action |
|------|--------|
| `0` | STOP (stop and end episode) |
| `1` | FORWARD (move forward) |
| `2` | LEFT (turn left) |
| `3` | RIGHT (turn right) |

### GT Data Usage

The ground truth files contain:
- **locations**: Full agent trajectory positions at each timestep
- **actions**: Discrete action sequence taken by the reference agent
- **forward_steps**: Total number of forward moves (useful for efficiency metrics)

---

## 3. Embeddings File (`embeddings.json.gz`)

**Format:** 2D list with shape `(2504, 50)`

Each instruction corresponds to a 50-dimensional embedding vector, likely from a pre-trained language model (e.g., DAN, BERT, or other sentence encoders).

```python
embeddings[0]  # [0.0, 0.0, 0.0, 0.0, 0.0, ...]  # 50-dim vector for first instruction
```

---

## Dataset Statistics (val_seen)

| Metric | Value |
|--------|-------|
| Episodes | 778 |
| Scene Source | MP3D (MatterPort3D) |
| Instruction Token Length | 200 (fixed, 0-padded) |
| Coordinate System | 3D cartesian (meters) |
| Rotation Format | Quaternion [w, x, y, z] |

---

## Scene Information

R2R uses **MatterPort3D (MP3D)** scenes:
- Real-world scanned indoor environments
- High-quality 3D meshes with realistic textures
- Paths follow navigable walkable surfaces

Scene file format: `mp3d/<scene_id>/<scene_id>.glb`

---

## Data Splits

| Split | Description | Purpose |
|-------|-------------|---------|
| `train` | Training scenes | Model training |
| `val_seen` | Validation, scenes seen in training | In-scene evaluation |
| `val_unseen` | Validation, new scenes | Generalization evaluation |
| `test` | Test set (no ground truth public) | Competition evaluation |
| `envdrop` | Environment dropout augmentation | Domain randomization |
| `joint_train_envdrop` | Combined training data | Robust training |

---

## Coordinate System

- **Position**: 3D coordinates in meters `[x, y, z]`
  - `y` typically represents height (may be negative in some scenes)
  - Origin depends on scene scan
- **Rotation**: Quaternion `[w, x, y, z]`
  - Represents agent's heading direction
  - Agent typically faces along its local +z axis

---

## Reference Path

The `reference_path` field contains a sequence of waypoints representing the shortest feasible path from start to goal. This is used for:
- Computing success metrics
- Trajectory comparison
- Training supervision (imitation learning)

The path is typically computed using geodesic distance on the navigation mesh.

---

## Related Resources

- [R2R Paper](https://arxiv.org/abs/1811.06931) - "Room-to-Room Navigation via Visual Object Detection"
- [MatterPort3D Dataset](https://niessner.github.io/Matterport/)
- Habitat Lab R2R task implementation

# ManiSkill

A research codebase for bimanual robot manipulation built on top of [openpi](https://github.com/Physical-Intelligence/openpi) (Physical Intelligence). The repository integrates three tightly coupled components covering the full pipeline from teleoperation data collection through Vision-Language-Action (VLA) policy training to real-robot deployment and trajectory visualization.

## Scope

The codebase targets the following research workflow:

1. **Data acquisition.** VR-based bimanual teleoperation produces episodes stored in the Zarr format (end-effector poses, gripper apertures, RGB observations, and tactile signals).
2. **Policy learning.** Fine-tuning of the π₀, π₀-FAST, and π₀.₅ flow-matching / autoregressive VLA checkpoints distributed by Physical Intelligence, using both JAX and PyTorch backends.
3. **Closed-loop deployment.** A client–server runtime streams actions from a remote inference host to the physical robot over WebSocket, decoupling GPU and robot environments.
4. **Qualitative analysis.** A standalone 3D visualizer renders recorded or rolled-out episodes for inspection and failure-mode analysis.

## Repository Layout

| Component | Role |
| --- | --- |
| [robot_server/](robot_server/) | On-robot runtime. Exposes a control interface ([client/robot_client.py](robot_server/client/robot_client.py)), wraps the proprietary driver stack under [real_world/](robot_server/real_world), and provides scripts for data collection, joint jogging, and trajectory replay. |
| [user_client/](user_client/) | Training and inference host. Contains the openpi-derived policy package ([policy/](user_client/policy)), a remote-control entrypoint ([client/interface_client.py](user_client/client/interface_client.py)), and shell wrappers for the main experiments. |
| [user_client/robot_visualization/](user_client/robot_visualization/) | Offline visualizer for `.zarr.zip` teleoperation logs. Supports interactive playback and headless video export. |

Each component is self-contained and is built or executed independently; the top-level directory is a monorepo rather than a single installable package.

## Requirements

Training and fine-tuning follow the upstream openpi envelope:

| Regime | GPU Memory | Reference Hardware |
| --- | --- | --- |
| Inference | ≥ 8 GB | RTX 4090 |
| LoRA fine-tuning | ≥ 22.5 GB | RTX 4090 |
| Full fine-tuning | ≥ 70 GB | A100 80 GB / H100 |

The training stack is validated on Ubuntu 22.04 with Python ≥ 3.11 and CUDA 12.8. On-robot execution additionally depends on the vendor wheels `rb_python` and `hblog`, which are bundled under [robot_server/real_world/robot_api/whl/](robot_server/real_world/robot_api/whl/) and are required only by `robot_server`.

## Installation

`robot_server` and `user_client` are managed with [uv](https://docs.astral.sh/uv/); `robot_visualization` uses a standard virtual environment.

```bash
# Training / inference host
cd user_client
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

# On-robot runtime (requires vendor drivers)
cd ../robot_server
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

# Visualizer
cd ../user_client/robot_visualization
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

`GIT_LFS_SKIP_SMUDGE=1` is required to resolve the LeRobot dependency without materializing its LFS payload.

## Usage

### Policy Learning (`user_client/`)

```bash
# 1. Compute state/action normalization statistics for the chosen config.
bash scripts/compute_norm_stats.sh

# 2. Launch fine-tuning (JAX backend; see script for config and flags).
bash scripts/train.sh

# 3. Sanity-check a single forward pass with a synthetic observation.
bash scripts/test_single_inf.sh

# 4. Serve the trained policy over WebSocket.
bash scripts/infer.sh
```

### Data Collection and Deployment (`robot_server/`)

```bash
# Bimanual teleoperation (V-B configuration).
bash scripts/bimanual_vb.sh

# End-to-end data collection pipeline.
bash scripts/run_data_collection_pipeline.sh

# Deterministic trajectory replay for validation.
bash scripts/replay_data.sh
```

At runtime, [user_client/client/interface_client.py](user_client/client/interface_client.py) issues observations to the policy server and forwards the returned action chunks to [robot_server/client/robot_client.py](robot_server/client/robot_client.py), which executes them on hardware.

### Trajectory Visualization (`user_client/robot_visualization/`)

```bash
# Interactive playback.
python src/viz_3d_enhanced.py data/your_data.zarr.zip

# Headless video export of a single episode.
python src/viz_3d_enhanced.py data/your_data.zarr.zip -r \
    --record_episode 1 --output_video demo.mp4
```

Interactive keybindings: `A`/`D` step frames, `W`/`S` switch episodes, `P` toggles auto-play, `1`–`5` select playback speed (0.25×–5×), `Q` exits.

## Data Format

Episodes are serialized as `.zarr.zip` archives. Each trajectory contains, per arm `robot{0,1}`:

| Key | Description |
| --- | --- |
| `robot{0,1}_eef_pos` | End-effector pose trajectory. |
| `robot{0,1}_gripper_width` | Gripper aperture (continuous). |
| `robot{0,1}_visual` | RGB observations from the wrist/scene cameras. |
| `robot{0,1}_left_tactile`, `robot{0,1}_right_tactile` | Per-finger tactile signals. |

## License and Attribution

`robot_server` and `user_client` are derivative works of [openpi](https://github.com/Physical-Intelligence/openpi) and inherit its license together with the upstream notices; see [robot_server/LICENSE](robot_server/LICENSE), [robot_server/NOTICE_1.md](robot_server/NOTICE_1.md), [robot_server/NOTICE_2.md](robot_server/NOTICE_2.md), and [user_client/LICENSE](user_client/LICENSE). Model checkpoints (π₀, π₀-FAST, π₀.₅, and the DROID / ALOHA / LIBERO fine-tunes) are distributed by Physical Intelligence under `gs://openpi-assets/` and are subject to their original terms.

## References

- Physical Intelligence. *π₀: A Vision-Language-Action Flow Model for General Robot Control*. [[blog](https://www.physicalintelligence.company/blog/pi0)]
- Physical Intelligence. *FAST: Efficient Action Tokenization for Vision-Language-Action Models*. [[page](https://www.physicalintelligence.company/research/fast)]
- Physical Intelligence. *π₀.₅: Open-World Generalization via Knowledge Insulation*. [[blog](https://www.physicalintelligence.company/blog/pi05)]
- Detailed component documentation: [robot_server/README.md](robot_server/README.md) and [user_client/robot_visualization/README.md](user_client/robot_visualization/README.md).

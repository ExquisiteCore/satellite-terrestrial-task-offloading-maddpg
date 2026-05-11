# Satellite-Terrestrial Task Offloading MADDPG

This project simulates task offloading in a satellite-terrestrial MEC network.
Each ground user splits its task among local execution, base-station MEC, and
LEO satellite MEC. The main comparison is All Local, Random, DQN, and MADDPG.

## Setup

```bash
uv sync
```

## Run

```bash
uv run python train_maddpg.py
uv run python train_dqn.py
uv run python evaluate.py
uv run python plot_results.py
```

Training and evaluation default to CUDA. Use `--device cpu` only when running on
a machine without a compatible NVIDIA GPU.

For a quick smoke test:

```bash
uv run python train_maddpg.py --episodes 3
uv run python train_dqn.py --episodes 3
uv run python evaluate.py --episodes 3
uv run python plot_results.py
```

## Outputs

```text
results/models/maddpg_best.pt
results/models/dqn.pt
results/csv/maddpg_train_log.csv
results/csv/dqn_train_log.csv
results/csv/evaluation_summary.csv
results/figures/reward_curve.png
results/figures/avg_delay_comparison.png
results/figures/avg_energy_comparison.png
results/figures/success_rate_comparison.png
results/figures/offload_ratio_maddpg.png
```

## Default Simulation Parameters

The default task deadlines are sampled from 5-20 seconds. This keeps the task
model strict enough for failed users to appear, while avoiding all-zero success
rates in short default evaluations and preserving visible differences between
local-only and offloading-capable policies.

The environment follows the proposal model: each user observes task size,
channel quality, distance to the base station and LEO satellite, and MEC
resource indicators. Actions are continuous task split ratios over local
execution, ground base-station MEC, and LEO satellite MEC. Rewards use a shared
team cost based on average delay, average energy, and the user deadline
violation rate.

## Scope

The model intentionally uses one base station MEC server, one LEO satellite MEC
server, and multiple users. The LEO satellite follows a simplified linear
motion model, and the channel rate is updated from distance-based path loss. It
does not include multi-satellite routing, inter-satellite links, cloud centers,
real ephemeris data, or cross-slot queues.

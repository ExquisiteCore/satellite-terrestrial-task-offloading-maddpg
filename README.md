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

## Scope

The model intentionally uses one base station MEC server, one LEO satellite MEC
server, and multiple users. It does not include multi-satellite routing,
inter-satellite links, cloud centers, real ephemeris data, or cross-slot queues.

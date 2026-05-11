from __future__ import annotations

import argparse
from dataclasses import replace
from collections.abc import Callable
from pathlib import Path

import numpy as np

from algorithms.baselines import all_local_policy, random_policy
from algorithms.dqn import DQNAgent
from algorithms.maddpg import MADDPG
from config import CSV_DIR, MODELS_DIR, EnvConfig
from envs.offloading_env import OffloadingEnv
from utils.logger import ensure_result_dirs, write_rows_csv
from utils.seed import set_seed


Policy = Callable[[np.ndarray, int], np.ndarray]


def aggregate_episode(infos: list[dict], num_users: int, baseline_delay: float | None = None) -> dict[str, float]:
    completed_per_step = [item["success_rate"] * num_users for item in infos]
    unfinished_per_step = [num_users - completed for completed in completed_per_step]
    avg_delay = sum(item["avg_delay"] for item in infos) / len(infos)
    return {
        "avg_reward": sum(item["avg_reward"] for item in infos) / len(infos),
        "avg_delay": avg_delay,
        "avg_energy": sum(item["avg_energy"] for item in infos) / len(infos),
        "success_rate": sum(item["success_rate"] for item in infos) / len(infos),
        "avg_completed_tasks": float(sum(completed_per_step)),
        "avg_unfinished_tasks": float(sum(unfinished_per_step)),
        "benefit_user_ratio": 0.0 if baseline_delay is None else float(avg_delay < baseline_delay),
        "avg_local_ratio": sum(item["avg_local_ratio"] for item in infos) / len(infos),
        "avg_bs_ratio": sum(item["avg_bs_ratio"] for item in infos) / len(infos),
        "avg_sat_ratio": sum(item["avg_sat_ratio"] for item in infos) / len(infos),
    }


def run_policy(
    name: str,
    env: OffloadingEnv,
    policy: Policy,
    episodes: int,
    baseline_delays: list[float] | None = None,
) -> dict:
    metrics = []
    for episode in range(episodes):
        obs = env.reset()
        done = False
        infos = []
        while not done:
            actions = policy(obs, episode)
            obs, _, done, info = env.step(actions)
            infos.append(info)
        baseline_delay = baseline_delays[episode] if baseline_delays is not None else None
        metrics.append(aggregate_episode(infos, env.num_users, baseline_delay=baseline_delay))
    avg_delay = float(np.mean([item["avg_delay"] for item in metrics]))
    return {
        "algorithm": name,
        "avg_reward": float(np.mean([item["avg_reward"] for item in metrics])),
        "avg_delay": avg_delay,
        "avg_energy": float(np.mean([item["avg_energy"] for item in metrics])),
        "success_rate": float(np.mean([item["success_rate"] for item in metrics])),
        "avg_completed_tasks": float(np.mean([item["avg_completed_tasks"] for item in metrics])),
        "avg_unfinished_tasks": float(np.mean([item["avg_unfinished_tasks"] for item in metrics])),
        "benefit_user_ratio": float(np.mean([item["benefit_user_ratio"] for item in metrics])),
        "avg_local_ratio": float(np.mean([item["avg_local_ratio"] for item in metrics])),
        "avg_bs_ratio": float(np.mean([item["avg_bs_ratio"] for item in metrics])),
        "avg_sat_ratio": float(np.mean([item["avg_sat_ratio"] for item in metrics])),
    }


def collect_baseline_delays(env: OffloadingEnv, episodes: int) -> list[float]:
    delays: list[float] = []
    for _ in range(episodes):
        obs = env.reset()
        done = False
        infos = []
        while not done:
            obs, _, done, info = env.step(all_local_policy(env.num_users))
            infos.append(info)
        delays.append(sum(item["avg_delay"] for item in infos) / len(infos))
    return delays


def maybe_load_model(model, path: str | Path, load_models: bool) -> dict[str, str | bool]:
    if not load_models:
        return {"checkpoint_loaded": False, "checkpoint_status": "not_requested", "checkpoint_path": ""}
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        return {
            "checkpoint_loaded": False,
            "checkpoint_status": "missing",
            "checkpoint_path": str(checkpoint_path),
        }
    model.load(checkpoint_path)
    return {
        "checkpoint_loaded": True,
        "checkpoint_status": "loaded",
        "checkpoint_path": str(checkpoint_path),
    }


def build_policies(env: OffloadingEnv, config: EnvConfig, dqn: DQNAgent, maddpg: MADDPG) -> list[tuple[str, Policy]]:
    return [
        ("All Local", lambda obs, episode: all_local_policy(env.num_users)),
        ("Random", lambda obs, episode: random_policy(env.num_users, seed=config.seed + episode)),
        ("DQN", lambda obs, episode: dqn.act(obs, epsilon=0.0)[0]),
        ("MADDPG", lambda obs, episode: maddpg.act(obs, noise_std=0.0)),
    ]


def evaluate(
    episodes: int,
    load_models: bool = True,
    dqn_checkpoint: str | Path = MODELS_DIR / "dqn.pt",
    maddpg_checkpoint: str | Path = MODELS_DIR / "maddpg_best.pt",
    device: str = "cuda",
) -> list[dict]:
    config = EnvConfig(seed=99)
    set_seed(config.seed)
    ensure_result_dirs()

    env = OffloadingEnv(config)
    maddpg = MADDPG(num_users=env.num_users, obs_dim=env.obs_dim, action_dim=env.action_dim, seed=config.seed, device=device)
    dqn = DQNAgent(obs_dim=env.obs_dim, seed=config.seed, device=device)
    dqn_status = maybe_load_model(dqn, dqn_checkpoint, load_models)
    maddpg_status = maybe_load_model(maddpg, maddpg_checkpoint, load_models)

    baseline_env = OffloadingEnv(config)
    baseline_delays = collect_baseline_delays(baseline_env, episodes)
    rows = [
        run_policy(
            name,
            env,
            policy,
            episodes,
            baseline_delays=None if name == "All Local" else baseline_delays,
        )
        for name, policy in build_policies(env, config, dqn, maddpg)
    ]
    for row in rows:
        status = {
            "DQN": dqn_status,
            "MADDPG": maddpg_status,
        }.get(
            row["algorithm"],
            {"checkpoint_loaded": False, "checkpoint_status": "not_applicable", "checkpoint_path": ""},
        )
        row.update(status)
    write_rows_csv(CSV_DIR / "evaluation_summary.csv", rows)
    return rows


def evaluate_sensitivity(
    episodes: int,
    load_factors: tuple[float, ...] = (0.6, 1.0, 1.4, 1.8),
    load_models: bool = True,
    dqn_checkpoint: str | Path = MODELS_DIR / "dqn.pt",
    maddpg_checkpoint: str | Path = MODELS_DIR / "maddpg_best.pt",
    device: str = "cuda",
) -> list[dict]:
    rows: list[dict] = []
    base_config = EnvConfig(seed=199)
    for factor in load_factors:
        config = replace(
            base_config,
            task_data_min_mb=base_config.task_data_min_mb * factor,
            task_data_max_mb=base_config.task_data_max_mb * factor,
        )
        set_seed(config.seed)
        ensure_result_dirs()
        env = OffloadingEnv(config)
        maddpg = MADDPG(
            num_users=env.num_users,
            obs_dim=env.obs_dim,
            action_dim=env.action_dim,
            seed=config.seed,
            device=device,
        )
        dqn = DQNAgent(obs_dim=env.obs_dim, seed=config.seed, device=device)
        maybe_load_model(dqn, dqn_checkpoint, load_models)
        maybe_load_model(maddpg, maddpg_checkpoint, load_models)
        baseline_env = OffloadingEnv(config)
        baseline_delays = collect_baseline_delays(baseline_env, episodes)
        for name, policy in build_policies(env, config, dqn, maddpg):
            row = run_policy(
                name,
                env,
                policy,
                episodes,
                baseline_delays=None if name == "All Local" else baseline_delays,
            )
            row["task_load_factor"] = factor
            rows.append(row)
    write_rows_csv(CSV_DIR / "sensitivity_summary.csv", rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--no-load-models", action="store_true")
    parser.add_argument("--dqn-checkpoint", type=Path, default=MODELS_DIR / "dqn.pt")
    parser.add_argument("--maddpg-checkpoint", type=Path, default=MODELS_DIR / "maddpg_best.pt")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--sensitivity", action="store_true")
    args = parser.parse_args()
    evaluate(
        args.episodes,
        load_models=not args.no_load_models,
        dqn_checkpoint=args.dqn_checkpoint,
        maddpg_checkpoint=args.maddpg_checkpoint,
        device=args.device,
    )
    if args.sensitivity:
        evaluate_sensitivity(
            args.episodes,
            load_models=not args.no_load_models,
            dqn_checkpoint=args.dqn_checkpoint,
            maddpg_checkpoint=args.maddpg_checkpoint,
            device=args.device,
        )


if __name__ == "__main__":
    main()

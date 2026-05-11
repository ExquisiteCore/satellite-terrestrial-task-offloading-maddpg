from __future__ import annotations

import argparse
from collections.abc import Callable

import numpy as np

from algorithms.baselines import all_local_policy, random_policy
from algorithms.dqn import DQNAgent
from algorithms.maddpg import MADDPG
from config import CSV_DIR, EnvConfig
from envs.offloading_env import OffloadingEnv
from utils.logger import ensure_result_dirs, write_rows_csv
from utils.seed import set_seed


Policy = Callable[[np.ndarray, int], np.ndarray]


def run_policy(name: str, env: OffloadingEnv, policy: Policy, episodes: int) -> dict:
    metrics = []
    for episode in range(episodes):
        obs = env.reset()
        done = False
        infos = []
        while not done:
            actions = policy(obs, episode)
            obs, _, done, info = env.step(actions)
            infos.append(info)
        metrics.append(
            {
                "avg_delay": sum(item["avg_delay"] for item in infos) / len(infos),
                "avg_energy": sum(item["avg_energy"] for item in infos) / len(infos),
                "success_rate": sum(item["success_rate"] for item in infos) / len(infos),
                "avg_local_ratio": sum(item["avg_local_ratio"] for item in infos) / len(infos),
                "avg_bs_ratio": sum(item["avg_bs_ratio"] for item in infos) / len(infos),
                "avg_sat_ratio": sum(item["avg_sat_ratio"] for item in infos) / len(infos),
            }
        )
    return {
        "algorithm": name,
        "avg_delay": float(np.mean([item["avg_delay"] for item in metrics])),
        "avg_energy": float(np.mean([item["avg_energy"] for item in metrics])),
        "success_rate": float(np.mean([item["success_rate"] for item in metrics])),
        "avg_local_ratio": float(np.mean([item["avg_local_ratio"] for item in metrics])),
        "avg_bs_ratio": float(np.mean([item["avg_bs_ratio"] for item in metrics])),
        "avg_sat_ratio": float(np.mean([item["avg_sat_ratio"] for item in metrics])),
    }


def evaluate(episodes: int) -> list[dict]:
    config = EnvConfig(seed=99)
    set_seed(config.seed)
    ensure_result_dirs()

    env = OffloadingEnv(config)
    maddpg = MADDPG(num_users=env.num_users, obs_dim=env.obs_dim, action_dim=env.action_dim, seed=config.seed)
    dqn = DQNAgent(obs_dim=env.obs_dim, seed=config.seed)

    rows = [
        run_policy("All Local", env, lambda obs, episode: all_local_policy(env.num_users), episodes),
        run_policy("Random", env, lambda obs, episode: random_policy(env.num_users, seed=config.seed + episode), episodes),
        run_policy("DQN", env, lambda obs, episode: dqn.act(obs, epsilon=0.0)[0], episodes),
        run_policy("MADDPG", env, lambda obs, episode: maddpg.act(obs, noise_std=0.0), episodes),
    ]
    write_rows_csv(CSV_DIR / "evaluation_summary.csv", rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()
    evaluate(args.episodes)


if __name__ == "__main__":
    main()

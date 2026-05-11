from __future__ import annotations

import argparse

from config import CSV_DIR, MODELS_DIR, EnvConfig, TrainConfig
from envs.offloading_env import OffloadingEnv
from algorithms.maddpg import MADDPG
from utils.logger import ensure_result_dirs, write_rows_csv
from utils.seed import set_seed


def train(episodes: int) -> list[dict]:
    env_config = EnvConfig()
    train_config = TrainConfig(episodes=episodes)
    set_seed(env_config.seed)
    ensure_result_dirs()

    env = OffloadingEnv(env_config)
    maddpg = MADDPG(
        num_users=env.num_users,
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        hidden_dim=train_config.hidden_dim,
        seed=env_config.seed,
    )

    rows: list[dict] = []
    best_reward = float("-inf")
    for episode in range(1, episodes + 1):
        obs = env.reset()
        done = False
        total_reward = 0.0
        infos = []
        while not done:
            actions = maddpg.act(obs, noise_std=train_config.exploration_noise)
            obs, rewards, done, info = env.step(actions)
            total_reward += float(rewards.mean())
            infos.append(info)

        avg_reward = total_reward / env_config.episode_steps
        row = {
            "episode": episode,
            "avg_reward": avg_reward,
            "avg_delay": sum(item["avg_delay"] for item in infos) / len(infos),
            "avg_energy": sum(item["avg_energy"] for item in infos) / len(infos),
            "success_rate": sum(item["success_rate"] for item in infos) / len(infos),
            "avg_local_ratio": sum(item["avg_local_ratio"] for item in infos) / len(infos),
            "avg_bs_ratio": sum(item["avg_bs_ratio"] for item in infos) / len(infos),
            "avg_sat_ratio": sum(item["avg_sat_ratio"] for item in infos) / len(infos),
        }
        rows.append(row)
        if avg_reward > best_reward:
            best_reward = avg_reward
            maddpg.save(MODELS_DIR / "maddpg_best.pt")

    write_rows_csv(CSV_DIR / "maddpg_train_log.csv", rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=TrainConfig().episodes)
    args = parser.parse_args()
    train(args.episodes)


if __name__ == "__main__":
    main()

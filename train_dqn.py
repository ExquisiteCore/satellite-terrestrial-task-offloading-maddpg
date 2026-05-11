from __future__ import annotations

import argparse

from config import CSV_DIR, MODELS_DIR, EnvConfig, TrainConfig
from envs.offloading_env import OffloadingEnv
from algorithms.dqn import DQNAgent
from utils.logger import ensure_result_dirs, write_rows_csv
from utils.seed import set_seed


def train(episodes: int) -> list[dict]:
    env_config = EnvConfig(seed=43)
    set_seed(env_config.seed)
    ensure_result_dirs()

    env = OffloadingEnv(env_config)
    agent = DQNAgent(obs_dim=env.obs_dim, hidden_dim=TrainConfig().hidden_dim, seed=env_config.seed)
    rows: list[dict] = []

    for episode in range(1, episodes + 1):
        obs = env.reset()
        done = False
        total_reward = 0.0
        infos = []
        epsilon = max(0.05, 0.3 * (1.0 - episode / max(episodes, 1)))
        while not done:
            actions, _ = agent.act(obs, epsilon=epsilon)
            obs, rewards, done, info = env.step(actions)
            total_reward += float(rewards.mean())
            infos.append(info)

        rows.append(
            {
                "episode": episode,
                "avg_reward": total_reward / env_config.episode_steps,
                "avg_delay": sum(item["avg_delay"] for item in infos) / len(infos),
                "avg_energy": sum(item["avg_energy"] for item in infos) / len(infos),
                "success_rate": sum(item["success_rate"] for item in infos) / len(infos),
            }
        )

    agent.save(MODELS_DIR / "dqn.pt")
    write_rows_csv(CSV_DIR / "dqn_train_log.csv", rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=TrainConfig().episodes)
    args = parser.parse_args()
    train(args.episodes)


if __name__ == "__main__":
    main()

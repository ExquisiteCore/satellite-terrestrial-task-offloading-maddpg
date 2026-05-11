from __future__ import annotations

import argparse

from config import CSV_DIR, MODELS_DIR, EnvConfig, TrainConfig
from envs.offloading_env import OffloadingEnv
from algorithms.dqn import DQNAgent
from algorithms.dqn.replay_buffer import DQNReplayBuffer
from utils.logger import ensure_result_dirs, write_rows_csv
from utils.seed import set_seed


def store_user_transitions(
    buffer: DQNReplayBuffer,
    obs,
    action_indices,
    rewards,
    next_obs,
    done: bool,
) -> None:
    for user_idx, action_idx in enumerate(action_indices):
        buffer.add(
            obs[user_idx],
            int(action_idx),
            float(rewards[user_idx]),
            next_obs[user_idx],
            done,
        )


def train(episodes: int, device: str = "cuda") -> list[dict]:
    env_config = EnvConfig(seed=43)
    train_config = TrainConfig()
    set_seed(env_config.seed)
    ensure_result_dirs()

    env = OffloadingEnv(env_config)
    agent = DQNAgent(obs_dim=env.obs_dim, hidden_dim=train_config.hidden_dim, seed=env_config.seed, device=device)
    buffer = DQNReplayBuffer(capacity=train_config.buffer_capacity, obs_dim=env.obs_dim, seed=env_config.seed)
    rows: list[dict] = []
    global_step = 0
    target_sync_interval = env_config.episode_steps

    for episode in range(1, episodes + 1):
        obs = env.reset()
        done = False
        total_reward = 0.0
        infos = []
        losses: list[float] = []
        epsilon = max(0.05, 0.3 * (1.0 - episode / max(episodes, 1)))
        while not done:
            actions, action_indices = agent.act(obs, epsilon=epsilon)
            current_obs = obs
            next_obs, rewards, done, info = env.step(actions)
            store_user_transitions(buffer, current_obs, action_indices, rewards, next_obs, done)
            if len(buffer) >= train_config.batch_size:
                metrics = agent.update(buffer.sample(train_config.batch_size), gamma=train_config.gamma)
                losses.append(metrics["loss"])
            global_step += 1
            if global_step % target_sync_interval == 0:
                agent.sync_target()
            obs = next_obs
            total_reward += float(rewards.mean())
            infos.append(info)

        rows.append(
            {
                "episode": episode,
                "avg_reward": total_reward / env_config.episode_steps,
                "avg_loss": sum(losses) / len(losses) if losses else 0.0,
                "num_updates": len(losses),
                "buffer_size": len(buffer),
                "avg_delay": sum(item["avg_delay"] for item in infos) / len(infos),
                "avg_energy": sum(item["avg_energy"] for item in infos) / len(infos),
                "success_rate": sum(item["success_rate"] for item in infos) / len(infos),
                "avg_local_ratio": sum(item["avg_local_ratio"] for item in infos) / len(infos),
                "avg_bs_ratio": sum(item["avg_bs_ratio"] for item in infos) / len(infos),
                "avg_sat_ratio": sum(item["avg_sat_ratio"] for item in infos) / len(infos),
                "epsilon": epsilon,
            }
        )

    agent.save(MODELS_DIR / "dqn.pt")
    write_rows_csv(CSV_DIR / "dqn_train_log.csv", rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=TrainConfig().episodes)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    train(args.episodes, device=args.device)


if __name__ == "__main__":
    main()

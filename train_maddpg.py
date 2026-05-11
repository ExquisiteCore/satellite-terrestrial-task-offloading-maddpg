from __future__ import annotations

import argparse

from config import CSV_DIR, MODELS_DIR, EnvConfig, TrainConfig
from envs.offloading_env import OffloadingEnv
from algorithms.maddpg import MADDPG
from algorithms.maddpg.replay_buffer import MultiAgentReplayBuffer
from utils.logger import ensure_result_dirs, write_rows_csv
from utils.seed import set_seed


def train(episodes: int, device: str = "cuda") -> list[dict]:
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
        actor_lr=train_config.actor_lr,
        critic_lr=train_config.critic_lr,
        device=device,
    )
    buffer = MultiAgentReplayBuffer(
        capacity=train_config.buffer_capacity,
        num_users=env.num_users,
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        seed=env_config.seed,
    )

    rows: list[dict] = []
    best_reward = float("-inf")
    for episode in range(1, episodes + 1):
        obs = env.reset()
        done = False
        total_reward = 0.0
        infos = []
        actor_losses: list[float] = []
        critic_losses: list[float] = []
        while not done:
            actions = maddpg.act(obs, noise_std=train_config.exploration_noise)
            current_obs = obs
            next_obs, rewards, done, info = env.step(actions)
            buffer.add(current_obs, actions, rewards, next_obs, done)
            if len(buffer) >= train_config.batch_size:
                metrics = maddpg.update(
                    buffer.sample(train_config.batch_size),
                    gamma=train_config.gamma,
                    tau=train_config.tau,
                )
                actor_losses.append(metrics["actor_loss"])
                critic_losses.append(metrics["critic_loss"])
            obs = next_obs
            total_reward += float(rewards.mean())
            infos.append(info)

        avg_reward = total_reward / env_config.episode_steps
        row = {
            "episode": episode,
            "avg_reward": avg_reward,
            "avg_actor_loss": sum(actor_losses) / len(actor_losses) if actor_losses else 0.0,
            "avg_critic_loss": sum(critic_losses) / len(critic_losses) if critic_losses else 0.0,
            "num_updates": len(actor_losses),
            "buffer_size": len(buffer),
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
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    train(args.episodes, device=args.device)


if __name__ == "__main__":
    main()

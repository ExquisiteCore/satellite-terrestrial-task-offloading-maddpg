from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from algorithms.maddpg.agent import MaddpgAgent


class MADDPG:
    def __init__(self, num_users: int, obs_dim: int, action_dim: int, hidden_dim: int = 128, seed: int = 42):
        self.num_users = num_users
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.agents = [
            MaddpgAgent(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim, seed=seed + i)
            for i in range(num_users)
        ]

    def act(self, obs: np.ndarray, noise_std: float = 0.0) -> np.ndarray:
        actions = []
        for i, agent in enumerate(self.agents):
            actions.append(agent.act(obs[i : i + 1], noise_std=noise_std)[0])
        return np.asarray(actions, dtype=np.float32)

    def save(self, path: str | Path) -> None:
        payload = {"actors": [agent.actor.state_dict() for agent in self.agents]}
        torch.save(payload, path)

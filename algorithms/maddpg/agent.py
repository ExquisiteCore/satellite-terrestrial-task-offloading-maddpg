from __future__ import annotations

import numpy as np
import torch

from algorithms.maddpg.networks import Actor


class MaddpgAgent:
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int, seed: int = 42):
        torch.manual_seed(seed)
        self.actor = Actor(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim)

    def act(self, obs: np.ndarray, noise_std: float = 0.0) -> np.ndarray:
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
            action = self.actor(obs_tensor).cpu().numpy()
        if noise_std > 0.0:
            action = action + np.random.default_rng().normal(0.0, noise_std, size=action.shape)
            action = np.clip(action, 0.0, None)
            row_sum = action.sum(axis=-1, keepdims=True)
            action = np.divide(action, row_sum, out=np.full_like(action, 1.0 / action.shape[-1]), where=row_sum > 0.0)
        return action

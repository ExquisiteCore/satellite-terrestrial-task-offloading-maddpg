from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from algorithms.dqn.dqn_network import DQNNetwork


DISCRETE_ACTIONS = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
        [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
    ],
    dtype=np.float32,
)


class DQNAgent:
    def __init__(self, obs_dim: int, hidden_dim: int = 128, seed: int = 42):
        torch.manual_seed(seed)
        self.network = DQNNetwork(obs_dim=obs_dim, num_actions=len(DISCRETE_ACTIONS), hidden_dim=hidden_dim)

    def act(self, obs: np.ndarray, epsilon: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng()
        if epsilon > 0.0:
            random_mask = rng.random(obs.shape[0]) < epsilon
        else:
            random_mask = np.zeros(obs.shape[0], dtype=bool)

        with torch.no_grad():
            q_values = self.network(torch.as_tensor(obs, dtype=torch.float32)).cpu().numpy()
        indices = np.argmax(q_values, axis=1)
        if np.any(random_mask):
            indices[random_mask] = rng.integers(0, len(DISCRETE_ACTIONS), size=int(np.sum(random_mask)))
        return DISCRETE_ACTIONS[indices], indices

    def save(self, path: str | Path) -> None:
        torch.save({"network": self.network.state_dict()}, path)

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DQNBatch:
    obs: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_obs: np.ndarray
    dones: np.ndarray


class DQNReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, seed: int = 42):
        self.capacity = capacity
        self.rng = np.random.default_rng(seed)
        self.position = 0
        self.size = 0
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def add(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, done: bool) -> None:
        self.obs[self.position] = obs
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_obs[self.position] = next_obs
        self.dones[self.position] = float(done)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def __len__(self) -> int:
        return self.size

    def sample(self, batch_size: int) -> DQNBatch:
        if self.size < batch_size:
            raise ValueError(f"cannot sample {batch_size} transitions from buffer of size {self.size}")
        indices = self.rng.choice(self.size, size=batch_size, replace=False)
        return DQNBatch(
            obs=self.obs[indices],
            actions=self.actions[indices],
            rewards=self.rewards[indices],
            next_obs=self.next_obs[indices],
            dones=self.dones[indices],
        )

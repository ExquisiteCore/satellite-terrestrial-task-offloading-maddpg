from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MultiAgentBatch:
    obs: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_obs: np.ndarray
    dones: np.ndarray


class MultiAgentReplayBuffer:
    def __init__(self, capacity: int, num_users: int, obs_dim: int, action_dim: int, seed: int = 42):
        self.capacity = capacity
        self.num_users = num_users
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.rng = np.random.default_rng(seed)
        self.position = 0
        self.size = 0

        self.obs = np.zeros((capacity, num_users, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, num_users, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, num_users), dtype=np.float32)
        self.next_obs = np.zeros((capacity, num_users, obs_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def add(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        self.obs[self.position] = obs
        self.actions[self.position] = actions
        self.rewards[self.position] = rewards
        self.next_obs[self.position] = next_obs
        self.dones[self.position] = float(done)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> MultiAgentBatch:
        if self.size < batch_size:
            raise ValueError(f"cannot sample {batch_size} transitions from buffer of size {self.size}")
        indices = self.rng.choice(self.size, size=batch_size, replace=False)
        return MultiAgentBatch(
            obs=self.obs[indices],
            actions=self.actions[indices],
            rewards=self.rewards[indices],
            next_obs=self.next_obs[indices],
            dones=self.dones[indices],
        )

    def __len__(self) -> int:
        return self.size

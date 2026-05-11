from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn, optim

from algorithms.dqn.dqn_network import DQNNetwork
from algorithms.dqn.replay_buffer import DQNBatch


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
    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int = 128,
        seed: int = 42,
        lr: float = 1e-3,
        device: str | torch.device | None = None,
    ):
        torch.manual_seed(seed)
        self.device = torch.device("cuda" if device is None else device)
        self.network = DQNNetwork(obs_dim=obs_dim, num_actions=len(DISCRETE_ACTIONS), hidden_dim=hidden_dim).to(self.device)
        self.target_network = DQNNetwork(obs_dim=obs_dim, num_actions=len(DISCRETE_ACTIONS), hidden_dim=hidden_dim).to(
            self.device
        )
        self.rng = np.random.default_rng(seed)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.sync_target()
        self.target_network.eval()

    def act(self, obs: np.ndarray, epsilon: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        if epsilon > 0.0:
            random_mask = self.rng.random(obs.shape[0]) < epsilon
        else:
            random_mask = np.zeros(obs.shape[0], dtype=bool)

        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            q_values = self.network(obs_tensor).cpu().numpy()
        indices = np.argmax(q_values, axis=1)
        if np.any(random_mask):
            indices[random_mask] = self.rng.integers(0, len(DISCRETE_ACTIONS), size=int(np.sum(random_mask)))
        return DISCRETE_ACTIONS[indices], indices

    def update(self, batch: DQNBatch, gamma: float) -> dict[str, float]:
        obs = torch.as_tensor(batch.obs, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch.actions, dtype=torch.long, device=self.device)
        rewards = torch.as_tensor(batch.rewards, dtype=torch.float32, device=self.device)
        next_obs = torch.as_tensor(batch.next_obs, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(batch.dones, dtype=torch.float32, device=self.device)

        q_values = self.network(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_values = self.target_network(next_obs).max(dim=1).values
            targets = rewards + gamma * (1.0 - dones) * next_q_values

        loss = self.loss_fn(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": float(loss.detach().cpu().item())}

    def sync_target(self) -> None:
        self.target_network.load_state_dict(self.network.state_dict())

    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "network": self.network.state_dict(),
                "target_network": self.target_network.state_dict(),
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        payload = torch.load(path, map_location=self.device, weights_only=True)
        if not isinstance(payload, dict):
            raise ValueError("DQN checkpoint must be a dictionary")
        required_keys = {"network", "target_network"}
        missing = sorted(required_keys.difference(payload))
        if missing:
            raise ValueError(f"DQN checkpoint missing keys: {', '.join(missing)}")
        self.network.load_state_dict(payload["network"])
        self.target_network.load_state_dict(payload["target_network"])
        self.target_network.eval()

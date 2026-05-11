from __future__ import annotations

import numpy as np
import torch
from torch import nn, optim

from algorithms.maddpg.networks import Actor, Critic


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.mul_(1.0 - tau)
            target_param.data.add_(source_param.data, alpha=tau)


class MaddpgAgent:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int,
        global_obs_dim: int | None = None,
        global_action_dim: int | None = None,
        seed: int = 42,
        actor_lr: float = 1e-3,
        critic_lr: float = 1e-3,
        device: str | torch.device | None = None,
        num_users: int | None = None,
        agent_index: int | None = None,
    ):
        torch.manual_seed(seed)
        self.device = torch.device("cuda" if device is None else device)
        self.rng = np.random.default_rng(seed)
        self.agent_index = agent_index
        if global_obs_dim is None or global_action_dim is None:
            if num_users is None:
                raise ValueError("num_users is required when global critic dimensions are not provided")
            global_obs_dim = num_users * obs_dim
            global_action_dim = num_users * action_dim

        self.actor = Actor(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(self.device)
        self.critic = Critic(
            global_obs_dim=global_obs_dim,
            global_action_dim=global_action_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)
        self.target_actor = Actor(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(self.device)
        self.target_critic = Critic(
            global_obs_dim=global_obs_dim,
            global_action_dim=global_action_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.sync_targets()
        self.target_actor.eval()
        self.target_critic.eval()

    def act(self, obs: np.ndarray, noise_std: float = 0.0) -> np.ndarray:
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            action = self.actor(obs_tensor).cpu().numpy()
        if noise_std > 0.0:
            action = action + self.rng.normal(0.0, noise_std, size=action.shape)
            action = np.clip(action, 0.0, None)
            row_sum = action.sum(axis=-1, keepdims=True)
            action = np.divide(action, row_sum, out=np.full_like(action, 1.0 / action.shape[-1]), where=row_sum > 0.0)
        return action

    def sync_targets(self) -> None:
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

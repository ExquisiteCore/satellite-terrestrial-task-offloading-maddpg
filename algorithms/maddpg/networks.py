from __future__ import annotations

import torch
from torch import nn


class Actor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int = 3, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        logits = self.net(obs)
        return torch.softmax(logits, dim=-1)


class Critic(nn.Module):
    def __init__(self, global_obs_dim: int, global_action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_obs_dim + global_action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, global_obs: torch.Tensor, global_action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([global_obs, global_action], dim=-1)
        return self.net(x)

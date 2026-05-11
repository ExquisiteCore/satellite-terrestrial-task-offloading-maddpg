from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from algorithms.maddpg.agent import MaddpgAgent, soft_update
from algorithms.maddpg.replay_buffer import MultiAgentBatch


class MADDPG:
    def __init__(
        self,
        num_users: int,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        seed: int = 42,
        actor_lr: float = 1e-3,
        critic_lr: float = 1e-3,
        device: str | torch.device | None = None,
    ):
        self.num_users = num_users
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = torch.device("cpu" if device is None else device)
        global_obs_dim = num_users * obs_dim
        global_action_dim = num_users * action_dim
        self.agents = [
            MaddpgAgent(
                obs_dim=obs_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                global_obs_dim=global_obs_dim,
                global_action_dim=global_action_dim,
                seed=seed + i,
                actor_lr=actor_lr,
                critic_lr=critic_lr,
                device=self.device,
            )
            for i in range(num_users)
        ]

    def act(self, obs: np.ndarray, noise_std: float = 0.0) -> np.ndarray:
        actions = []
        for i, agent in enumerate(self.agents):
            actions.append(agent.act(obs[i : i + 1], noise_std=noise_std)[0])
        return np.asarray(actions, dtype=np.float32)

    def update(self, batch: MultiAgentBatch, gamma: float, tau: float) -> dict[str, float]:
        obs = torch.as_tensor(batch.obs, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch.actions, dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(batch.rewards, dtype=torch.float32, device=self.device)
        next_obs = torch.as_tensor(batch.next_obs, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(batch.dones, dtype=torch.float32, device=self.device)

        batch_size = obs.shape[0]
        global_obs = obs.reshape(batch_size, -1)
        global_actions = actions.reshape(batch_size, -1)
        next_global_obs = next_obs.reshape(batch_size, -1)

        with torch.no_grad():
            target_next_actions = [
                agent.target_actor(next_obs[:, agent_idx, :]) for agent_idx, agent in enumerate(self.agents)
            ]
            target_global_actions = torch.cat(target_next_actions, dim=1)

        actor_losses: list[float] = []
        critic_losses: list[float] = []
        for agent_idx, agent in enumerate(self.agents):
            with torch.no_grad():
                target_q = agent.target_critic(next_global_obs, target_global_actions).squeeze(1)
                target = rewards[:, agent_idx] + gamma * (1.0 - dones) * target_q

            q_values = agent.critic(global_obs, global_actions).squeeze(1)
            critic_loss = nn.functional.mse_loss(q_values, target)
            agent.critic_optimizer.zero_grad()
            critic_loss.backward()
            agent.critic_optimizer.step()
            critic_losses.append(float(critic_loss.detach().cpu().item()))

            current_actions = []
            for other_idx, other_agent in enumerate(self.agents):
                actor_action = other_agent.actor(obs[:, other_idx, :])
                if other_idx != agent_idx:
                    actor_action = actor_action.detach()
                current_actions.append(actor_action)
            current_global_actions = torch.cat(current_actions, dim=1)
            actor_loss = -agent.critic(global_obs, current_global_actions).mean()
            agent.actor_optimizer.zero_grad()
            actor_loss.backward()
            agent.actor_optimizer.step()
            actor_losses.append(float(actor_loss.detach().cpu().item()))

            soft_update(agent.target_actor, agent.actor, tau=tau)
            soft_update(agent.target_critic, agent.critic, tau=tau)

        return {
            "actor_loss": float(np.mean(actor_losses)),
            "critic_loss": float(np.mean(critic_losses)),
        }

    def save(self, path: str | Path) -> None:
        payload = {
            "actors": [agent.actor.state_dict() for agent in self.agents],
            "critics": [agent.critic.state_dict() for agent in self.agents],
            "target_actors": [agent.target_actor.state_dict() for agent in self.agents],
            "target_critics": [agent.target_critic.state_dict() for agent in self.agents],
        }
        torch.save(payload, path)

    def load(self, path: str | Path) -> None:
        payload = torch.load(path, map_location=self.device, weights_only=True)
        if not isinstance(payload, dict):
            raise ValueError("MADDPG checkpoint must be a dictionary")
        self._validate_checkpoint_list(payload, "actors")
        self._validate_checkpoint_list(payload, "critics")
        self._validate_checkpoint_list(payload, "target_actors")
        self._validate_checkpoint_list(payload, "target_critics")

        for agent, actor, critic, target_actor, target_critic in zip(
            self.agents,
            payload["actors"],
            payload["critics"],
            payload["target_actors"],
            payload["target_critics"],
        ):
            agent.actor.load_state_dict(actor)
            agent.critic.load_state_dict(critic)
            agent.target_actor.load_state_dict(target_actor)
            agent.target_critic.load_state_dict(target_critic)
            agent.target_actor.eval()
            agent.target_critic.eval()

    def _validate_checkpoint_list(self, payload: dict, key: str) -> None:
        if key not in payload:
            raise ValueError(f"MADDPG checkpoint missing {key}")
        if len(payload[key]) != len(self.agents):
            raise ValueError(
                f"MADDPG checkpoint {key} has {len(payload[key])} entries; expected {len(self.agents)}"
            )

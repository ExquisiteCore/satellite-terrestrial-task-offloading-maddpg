import numpy as np
import torch

from algorithms.dqn.dqn_agent import DISCRETE_ACTIONS
from algorithms.dqn.dqn_network import DQNNetwork
from algorithms.maddpg.networks import Actor, Critic
from algorithms.maddpg.replay_buffer import MultiAgentReplayBuffer


def test_actor_outputs_simplex_actions():
    actor = Actor(obs_dim=10, action_dim=3, hidden_dim=32)
    obs = torch.zeros((4, 10), dtype=torch.float32)

    actions = actor(obs)

    assert actions.shape == (4, 3)
    assert torch.all(actions >= 0.0)
    torch.testing.assert_close(actions.sum(dim=1), torch.ones(4))


def test_critic_outputs_one_q_value_per_batch_row():
    critic = Critic(global_obs_dim=40, global_action_dim=12, hidden_dim=32)
    obs = torch.zeros((5, 40), dtype=torch.float32)
    actions = torch.zeros((5, 12), dtype=torch.float32)

    q_values = critic(obs, actions)

    assert q_values.shape == (5, 1)


def test_multi_agent_replay_buffer_samples_expected_shapes():
    buffer = MultiAgentReplayBuffer(capacity=10, num_users=2, obs_dim=10, action_dim=3, seed=2)
    obs = np.zeros((2, 10), dtype=np.float32)
    actions = np.zeros((2, 3), dtype=np.float32)
    rewards = np.zeros(2, dtype=np.float32)

    for _ in range(4):
        buffer.add(obs, actions, rewards, obs + 1.0, False)

    batch = buffer.sample(batch_size=3)

    assert batch.obs.shape == (3, 2, 10)
    assert batch.actions.shape == (3, 2, 3)
    assert batch.rewards.shape == (3, 2)
    assert batch.next_obs.shape == (3, 2, 10)
    assert batch.dones.shape == (3,)


def test_dqn_network_outputs_one_value_per_discrete_action():
    network = DQNNetwork(obs_dim=10, num_actions=len(DISCRETE_ACTIONS), hidden_dim=32)
    obs = torch.zeros((6, 10), dtype=torch.float32)

    q_values = network(obs)

    assert q_values.shape == (6, len(DISCRETE_ACTIONS))
    assert DISCRETE_ACTIONS.shape == (7, 3)
    np.testing.assert_allclose(DISCRETE_ACTIONS.sum(axis=1), np.ones(7), atol=1e-8)

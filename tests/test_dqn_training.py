import math

import numpy as np
import torch

from algorithms.dqn.dqn_agent import DQNAgent
from algorithms.dqn.replay_buffer import DQNBatch, DQNReplayBuffer
from train_dqn import store_user_transitions


def test_dqn_agent_initializes_identical_target_network():
    agent = DQNAgent(obs_dim=4, hidden_dim=16, seed=7)

    for online_param, target_param in zip(agent.network.parameters(), agent.target_network.parameters()):
        torch.testing.assert_close(online_param, target_param)
        assert online_param.data_ptr() != target_param.data_ptr()


def test_dqn_agent_update_changes_online_parameters_and_returns_finite_loss():
    agent = DQNAgent(obs_dim=3, hidden_dim=8, seed=3, lr=1e-2)
    batch = DQNBatch(
        obs=np.array(
            [
                [0.0, 0.2, 0.4],
                [0.1, 0.3, 0.5],
                [0.2, 0.4, 0.6],
                [0.3, 0.5, 0.7],
            ],
            dtype=np.float32,
        ),
        actions=np.array([0, 1, 2, 3], dtype=np.int64),
        rewards=np.array([1.0, 0.5, -0.25, 0.75], dtype=np.float32),
        next_obs=np.array(
            [
                [0.4, 0.2, 0.0],
                [0.5, 0.3, 0.1],
                [0.6, 0.4, 0.2],
                [0.7, 0.5, 0.3],
            ],
            dtype=np.float32,
        ),
        dones=np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
    )
    before = [param.detach().clone() for param in agent.network.parameters()]
    target_before = [param.detach().clone() for param in agent.target_network.parameters()]

    metrics = agent.update(batch, gamma=0.95)

    assert "loss" in metrics
    assert math.isfinite(metrics["loss"])
    assert metrics["loss"] >= 0.0
    assert any(not torch.allclose(old, new) for old, new in zip(before, agent.network.parameters()))
    for old_target, new_target in zip(target_before, agent.target_network.parameters()):
        torch.testing.assert_close(old_target, new_target)


def test_dqn_agent_update_uses_bellman_target_with_done_mask_and_action_indices():
    agent = DQNAgent(obs_dim=2, hidden_dim=4, seed=17, lr=0.0)
    with torch.no_grad():
        for param in agent.network.parameters():
            param.zero_()
        for param in agent.target_network.parameters():
            param.zero_()
        agent.network.net[-1].bias.copy_(torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
        agent.target_network.net[-1].bias.copy_(torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]))

    batch = DQNBatch(
        obs=np.zeros((2, 2), dtype=np.float32),
        actions=np.array([2, 5], dtype=np.int64),
        rewards=np.array([1.0, -2.0], dtype=np.float32),
        next_obs=np.ones((2, 2), dtype=np.float32),
        dones=np.array([0.0, 1.0], dtype=np.float32),
    )

    metrics = agent.update(batch, gamma=0.5)

    expected_targets = np.array([1.0 + 0.5 * 70.0, -2.0], dtype=np.float32)
    selected_q = np.array([2.0, 5.0], dtype=np.float32)
    expected_loss = float(np.mean(np.square(selected_q - expected_targets)))
    assert math.isclose(metrics["loss"], expected_loss, rel_tol=1e-6)


def test_sync_target_copies_online_parameters_to_target_network():
    agent = DQNAgent(obs_dim=4, hidden_dim=16, seed=11)
    with torch.no_grad():
        for param in agent.network.parameters():
            param.add_(1.0)

    agent.sync_target()

    for online_param, target_param in zip(agent.network.parameters(), agent.target_network.parameters()):
        torch.testing.assert_close(online_param, target_param)


def test_dqn_replay_buffer_reports_current_length():
    buffer = DQNReplayBuffer(capacity=3, obs_dim=2, seed=5)

    assert len(buffer) == 0
    for index in range(5):
        obs = np.full(2, index, dtype=np.float32)
        buffer.add(obs, action=1, reward=0.5, next_obs=obs + 1.0, done=False)

    assert len(buffer) == 3


def test_store_user_transitions_adds_one_transition_per_user():
    buffer = DQNReplayBuffer(capacity=10, obs_dim=2, seed=5)
    obs = np.array([[1.0, 1.1], [2.0, 2.2], [3.0, 3.3]], dtype=np.float32)
    next_obs = obs + 10.0
    action_indices = np.array([0, 4, 6], dtype=np.int64)
    rewards = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    store_user_transitions(buffer, obs, action_indices, rewards, next_obs, done=True)

    assert len(buffer) == 3
    np.testing.assert_allclose(buffer.obs[:3], obs)
    np.testing.assert_array_equal(buffer.actions[:3], action_indices)
    np.testing.assert_allclose(buffer.rewards[:3], rewards)
    np.testing.assert_allclose(buffer.next_obs[:3], next_obs)
    np.testing.assert_allclose(buffer.dones[:3], np.ones(3))


def test_dqn_epsilon_actions_are_reproducible_for_same_seed():
    obs = np.zeros((6, 4), dtype=np.float32)
    agent_a = DQNAgent(obs_dim=4, hidden_dim=16, seed=13)
    agent_b = DQNAgent(obs_dim=4, hidden_dim=16, seed=13)

    _, indices_a = agent_a.act(obs, epsilon=1.0)
    _, indices_b = agent_b.act(obs, epsilon=1.0)

    np.testing.assert_array_equal(indices_a, indices_b)

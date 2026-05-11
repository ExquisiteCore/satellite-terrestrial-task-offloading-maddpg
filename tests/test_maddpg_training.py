import math
from pathlib import Path

import numpy as np
import torch

from algorithms.maddpg import MADDPG
from algorithms.maddpg.agent import MaddpgAgent
from algorithms.maddpg.maddpg import soft_update
from algorithms.maddpg.replay_buffer import MultiAgentBatch, MultiAgentReplayBuffer
from train_maddpg import train


def test_maddpg_defaults_to_cuda():
    maddpg = MADDPG(num_users=2, obs_dim=4, action_dim=3, hidden_dim=16, seed=9)

    assert maddpg.device.type == "cuda"
    assert all(agent.device.type == "cuda" for agent in maddpg.agents)


def make_batch(batch_size: int = 5, num_users: int = 3, obs_dim: int = 4, action_dim: int = 3) -> MultiAgentBatch:
    rng = np.random.default_rng(123)
    raw_actions = rng.uniform(0.0, 1.0, size=(batch_size, num_users, action_dim)).astype(np.float32)
    actions = raw_actions / raw_actions.sum(axis=2, keepdims=True)
    return MultiAgentBatch(
        obs=rng.uniform(0.0, 1.0, size=(batch_size, num_users, obs_dim)).astype(np.float32),
        actions=actions.astype(np.float32),
        rewards=rng.normal(0.0, 1.0, size=(batch_size, num_users)).astype(np.float32),
        next_obs=rng.uniform(0.0, 1.0, size=(batch_size, num_users, obs_dim)).astype(np.float32),
        dones=np.array([0.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float32)[:batch_size],
    )


def test_maddpg_agent_initializes_identical_target_networks():
    agent = MaddpgAgent(
        obs_dim=4,
        action_dim=3,
        hidden_dim=16,
        global_obs_dim=8,
        global_action_dim=6,
        seed=5,
        device="cpu",
    )

    for online_param, target_param in zip(agent.actor.parameters(), agent.target_actor.parameters()):
        torch.testing.assert_close(online_param, target_param)
        assert online_param.data_ptr() != target_param.data_ptr()
    for online_param, target_param in zip(agent.critic.parameters(), agent.target_critic.parameters()):
        torch.testing.assert_close(online_param, target_param)
        assert online_param.data_ptr() != target_param.data_ptr()


def test_soft_update_tau_zero_leaves_target_and_tau_one_copies_source():
    source = torch.nn.Linear(2, 2)
    target = torch.nn.Linear(2, 2)
    with torch.no_grad():
        source.weight.fill_(2.0)
        source.bias.fill_(3.0)
        target.weight.fill_(-1.0)
        target.bias.fill_(-2.0)
    before = [param.detach().clone() for param in target.parameters()]

    soft_update(target, source, tau=0.0)

    for old, new in zip(before, target.parameters()):
        torch.testing.assert_close(old, new)

    soft_update(target, source, tau=1.0)

    for source_param, target_param in zip(source.parameters(), target.parameters()):
        torch.testing.assert_close(source_param, target_param)


def test_maddpg_update_changes_online_parameters_and_returns_losses():
    maddpg = MADDPG(
        num_users=3,
        obs_dim=4,
        action_dim=3,
        hidden_dim=16,
        seed=9,
        actor_lr=1e-2,
        critic_lr=1e-2,
        device="cpu",
    )
    batch = make_batch()
    actor_before = [param.detach().clone() for param in maddpg.agents[0].actor.parameters()]
    critic_before = [param.detach().clone() for param in maddpg.agents[0].critic.parameters()]
    target_before = [param.detach().clone() for param in maddpg.agents[0].target_actor.parameters()]

    metrics = maddpg.update(batch, gamma=0.95, tau=0.0)

    assert math.isfinite(metrics["actor_loss"])
    assert math.isfinite(metrics["critic_loss"])
    assert any(not torch.allclose(old, new) for old, new in zip(actor_before, maddpg.agents[0].actor.parameters()))
    assert any(not torch.allclose(old, new) for old, new in zip(critic_before, maddpg.agents[0].critic.parameters()))
    for old_target, new_target in zip(target_before, maddpg.agents[0].target_actor.parameters()):
        torch.testing.assert_close(old_target, new_target)


def test_maddpg_act_noise_is_reproducible_for_same_seed_and_preserves_simplex():
    obs = np.zeros((3, 4), dtype=np.float32)
    maddpg_a = MADDPG(num_users=3, obs_dim=4, action_dim=3, hidden_dim=16, seed=21, device="cpu")
    maddpg_b = MADDPG(num_users=3, obs_dim=4, action_dim=3, hidden_dim=16, seed=21, device="cpu")

    actions_a = maddpg_a.act(obs, noise_std=0.3)
    actions_b = maddpg_b.act(obs, noise_std=0.3)

    assert actions_a.shape == (3, 3)
    assert actions_a.dtype == np.float32
    assert np.all(actions_a >= 0.0)
    np.testing.assert_allclose(actions_a.sum(axis=1), np.ones(3), atol=1e-6)
    np.testing.assert_allclose(actions_a, actions_b)


def test_multi_agent_replay_buffer_rejects_undersized_sample():
    buffer = MultiAgentReplayBuffer(capacity=4, num_users=2, obs_dim=3, action_dim=3, seed=1)

    try:
        buffer.sample(batch_size=1)
    except ValueError as exc:
        assert "cannot sample" in str(exc)
    else:
        raise AssertionError("expected undersized sample to raise ValueError")


def test_maddpg_save_contains_full_training_state():
    maddpg = MADDPG(num_users=2, obs_dim=4, action_dim=3, hidden_dim=16, seed=10, device="cpu")
    path = Path("maddpg_training_payload_test.pt")

    try:
        maddpg.save(path)
        payload = torch.load(path, weights_only=False)

        assert set(payload) >= {"actors", "critics", "target_actors", "target_critics"}
        assert len(payload["actors"]) == 2
        assert len(payload["critics"]) == 2
    finally:
        path.unlink(missing_ok=True)


def test_train_maddpg_rows_include_training_diagnostics():
    rows = train(episodes=2)

    assert len(rows) == 2
    row = rows[-1]
    assert set(row) >= {
        "avg_actor_loss",
        "avg_critic_loss",
        "num_updates",
        "buffer_size",
        "avg_reward",
        "avg_delay",
        "avg_energy",
        "success_rate",
        "avg_local_ratio",
        "avg_bs_ratio",
        "avg_sat_ratio",
    }
    assert row["buffer_size"] > 0
    assert row["num_updates"] > 0

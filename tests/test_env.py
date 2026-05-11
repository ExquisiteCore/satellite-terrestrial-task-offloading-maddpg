import numpy as np

from config import EnvConfig
from envs.offloading_env import OffloadingEnv, StarGroundEnv, normalize_actions


def test_normalize_actions_returns_valid_simplex_rows():
    raw = np.array([[0.2, 0.3, 0.5], [-1.0, 0.0, 3.0], [0.0, 0.0, 0.0]])
    actions = normalize_actions(raw)

    assert actions.shape == (3, 3)
    assert np.all(actions >= 0.0)
    np.testing.assert_allclose(actions.sum(axis=1), np.ones(3), atol=1e-8)
    np.testing.assert_allclose(actions[2], np.array([1 / 3, 1 / 3, 1 / 3]), atol=1e-8)


def test_environment_reset_and_step_shapes():
    config = EnvConfig(num_users=4, episode_steps=5, seed=7)
    env = OffloadingEnv(config)

    obs = env.reset()
    assert obs.shape == (4, env.obs_dim)

    actions = np.tile(np.array([[0.2, 0.5, 0.3]]), (4, 1))
    next_obs, rewards, done, info = env.step(actions)

    assert next_obs.shape == (4, env.obs_dim)
    assert rewards.shape == (4,)
    assert done is False
    assert info["avg_delay"] >= 0.0
    assert info["avg_energy"] >= 0.0
    assert 0.0 <= info["success_rate"] <= 1.0


def test_environment_done_after_episode_steps():
    config = EnvConfig(num_users=2, episode_steps=2, seed=3)
    env = OffloadingEnv(config)
    env.reset()

    _, _, done_1, _ = env.step(np.ones((2, 3)))
    _, _, done_2, _ = env.step(np.ones((2, 3)))

    assert done_1 is False
    assert done_2 is True


def test_environment_exposes_proposal_compatible_action_helpers():
    config = EnvConfig(num_users=3, seed=11)
    env = OffloadingEnv(config)

    random_actions = env.sample_random_actions()
    local_actions = env.all_local_actions()

    assert StarGroundEnv is OffloadingEnv
    assert random_actions.shape == (3, env.action_dim)
    assert local_actions.shape == (3, env.action_dim)
    assert np.all(random_actions >= 0.0)
    np.testing.assert_allclose(random_actions.sum(axis=1), np.ones(3), atol=1e-6)
    np.testing.assert_allclose(local_actions, np.tile(np.array([[1.0, 0.0, 0.0]]), (3, 1)))


def test_rewards_use_proposal_aligned_team_cost():
    config = EnvConfig(num_users=3, seed=13)
    env = OffloadingEnv(config)
    env.reset()

    _, rewards, _, info = env.step(env.all_local_actions())

    assert rewards.shape == (3,)
    np.testing.assert_allclose(rewards, np.full(3, info["team_reward"], dtype=np.float32))
    assert info["team_reward"] == info["avg_reward"]
    assert info["total_delay_cost"] >= 0.0
    assert info["total_energy_cost"] >= 0.0
    assert info["deadline_violation"] in {0.0, 1.0}


def test_deadline_violation_penalty_reduces_team_reward():
    loose = OffloadingEnv(EnvConfig(num_users=2, seed=17, deadline_min_s=1000.0, deadline_max_s=1000.0))
    strict = OffloadingEnv(EnvConfig(num_users=2, seed=17, deadline_min_s=0.001, deadline_max_s=0.001))

    loose.reset()
    strict.reset()
    actions = loose.all_local_actions()
    _, loose_rewards, _, loose_info = loose.step(actions)
    _, strict_rewards, _, strict_info = strict.step(actions)

    assert loose_info["deadline_violation"] == 0.0
    assert strict_info["deadline_violation"] == 1.0
    assert float(strict_rewards[0]) < float(loose_rewards[0])

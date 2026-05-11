import numpy as np

from config import EnvConfig
from envs.offloading_env import OffloadingEnv, normalize_actions


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

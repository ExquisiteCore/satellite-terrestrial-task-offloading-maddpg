import numpy as np

from algorithms.baselines import all_local_policy, random_policy, uniform_policy


def test_all_local_policy_uses_only_local_execution():
    actions = all_local_policy(num_users=3)
    np.testing.assert_allclose(actions, np.array([[1.0, 0.0, 0.0]] * 3))


def test_uniform_policy_splits_work_equally():
    actions = uniform_policy(num_users=2)
    np.testing.assert_allclose(actions, np.array([[1 / 3, 1 / 3, 1 / 3]] * 2))


def test_random_policy_returns_valid_action_rows():
    actions = random_policy(num_users=5, seed=11)
    assert actions.shape == (5, 3)
    assert np.all(actions >= 0.0)
    np.testing.assert_allclose(actions.sum(axis=1), np.ones(5), atol=1e-8)

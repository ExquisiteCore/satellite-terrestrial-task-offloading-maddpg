import numpy as np


def all_local_policy(num_users: int) -> np.ndarray:
    return np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float64), (num_users, 1))


def uniform_policy(num_users: int) -> np.ndarray:
    return np.tile(np.array([[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]], dtype=np.float64), (num_users, 1))


def random_policy(num_users: int, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.dirichlet(alpha=np.ones(3), size=num_users)

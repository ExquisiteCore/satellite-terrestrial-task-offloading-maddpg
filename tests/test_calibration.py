import numpy as np

from algorithms.baselines import all_local_policy, random_policy, uniform_policy
from config import EnvConfig
from envs.offloading_env import OffloadingEnv


def _rollout(policy_fn, *, seed: int = 99) -> dict[str, float]:
    config = EnvConfig(seed=seed)
    env = OffloadingEnv(config)
    obs = env.reset()
    done = False
    infos = []
    step = 0

    while not done:
        actions = policy_fn(env.num_users, step)
        obs, _, done, info = env.step(actions)
        infos.append(info)
        step += 1

    return {
        "avg_delay": float(np.mean([info["avg_delay"] for info in infos])),
        "avg_energy": float(np.mean([info["avg_energy"] for info in infos])),
        "success_rate": float(np.mean([info["success_rate"] for info in infos])),
    }


def test_default_environment_has_nontrivial_baseline_success_rate():
    metrics = {
        "all_local": _rollout(lambda num_users, step: all_local_policy(num_users)),
        "uniform": _rollout(lambda num_users, step: uniform_policy(num_users)),
        "random": _rollout(lambda num_users, step: random_policy(num_users, seed=10_000 + step)),
    }

    nontrivial_success_rates = [
        metrics[name]["success_rate"]
        for name in ("uniform", "random")
        if 0.0 < metrics[name]["success_rate"] < 1.0
    ]

    assert nontrivial_success_rates, metrics


def test_all_local_policy_differs_from_offloading_capable_policy():
    local_metrics = _rollout(lambda num_users, step: all_local_policy(num_users))
    offloading_metrics = {
        "uniform": _rollout(lambda num_users, step: uniform_policy(num_users)),
        "random": _rollout(lambda num_users, step: random_policy(num_users, seed=20_000 + step)),
    }

    for policy_name, metrics in offloading_metrics.items():
        delay_gap = abs(local_metrics["avg_delay"] - metrics["avg_delay"])
        energy_gap = abs(local_metrics["avg_energy"] - metrics["avg_energy"])

        assert (
            delay_gap >= 0.1 or energy_gap >= 0.1
        ), f"policy={policy_name}, local={local_metrics}, offloading={metrics}"

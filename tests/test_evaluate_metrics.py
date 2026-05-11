import numpy as np

from config import EnvConfig
from envs.offloading_env import OffloadingEnv
from evaluate import evaluate_sensitivity, run_policy


def test_run_policy_reports_reward_completion_and_benefit_metrics():
    env = OffloadingEnv(EnvConfig(num_users=3, episode_steps=4, seed=51))
    row = run_policy("All Local", env, lambda obs, episode: env.all_local_actions(), episodes=2)

    assert row["algorithm"] == "All Local"
    assert "avg_reward" in row
    assert "avg_completed_tasks" in row
    assert "avg_unfinished_tasks" in row
    assert "benefit_user_ratio" in row
    assert row["avg_completed_tasks"] + row["avg_unfinished_tasks"] == env.num_users * env.config.episode_steps
    assert 0.0 <= row["benefit_user_ratio"] <= 1.0


def test_evaluate_sensitivity_records_task_load_rows(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    rows = evaluate_sensitivity(
        episodes=1,
        load_factors=(0.6, 1.2),
        load_models=False,
        device="cpu",
    )

    assert len(rows) == 8
    assert {row["task_load_factor"] for row in rows} == {0.6, 1.2}
    for row in rows:
        assert row["avg_completed_tasks"] >= 0.0
        assert row["avg_unfinished_tasks"] >= 0.0

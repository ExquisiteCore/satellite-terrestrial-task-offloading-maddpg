import numpy as np

from visualization.trace import DEFAULT_POLICY_NAMES, generate_rollout_trace


def test_generate_rollout_trace_contains_all_default_policies():
    trace = generate_rollout_trace(steps=3, seed=123, load_models=False, device="cpu")

    assert trace["schema_version"] == 1
    assert list(trace["policies"].keys()) == list(DEFAULT_POLICY_NAMES)
    for policy_name in DEFAULT_POLICY_NAMES:
        assert len(trace["policies"][policy_name]["steps"]) == 3


def test_generate_rollout_trace_records_user_actions_and_metrics():
    trace = generate_rollout_trace(steps=2, seed=123, load_models=False, device="cpu")
    step = trace["policies"]["MADDPG"]["steps"][0]

    assert step["step"] == 0
    assert set(step["metrics"]).issuperset(
        {
            "avg_delay",
            "avg_energy",
            "success_rate",
            "avg_reward",
            "avg_local_ratio",
            "avg_bs_ratio",
            "avg_sat_ratio",
        }
    )
    assert "base_station" in step
    assert "satellite" in step
    assert len(step["users"]) == trace["config"]["num_users"]

    for user in step["users"]:
        action = np.array(user["action"], dtype=float)
        assert action.shape == (3,)
        assert np.all(action >= 0.0)
        np.testing.assert_allclose(action.sum(), 1.0, atol=1e-6)
        assert set(user).issuperset(
            {
                "id",
                "x_m",
                "y_m",
                "task_data_mb",
                "deadline_s",
                "bs_distance_m",
                "sat_distance_m",
                "bs_rate_bps",
                "sat_rate_bps",
                "delay_s",
                "energy_j",
                "success",
            }
        )


def test_default_policy_traces_share_initial_geometry_for_comparison():
    trace = generate_rollout_trace(steps=1, seed=321, load_models=False, device="cpu")

    first_users_by_policy = {
        policy_name: policy_trace["steps"][0]["users"]
        for policy_name, policy_trace in trace["policies"].items()
    }
    reference = first_users_by_policy["MADDPG"]
    for policy_name, users in first_users_by_policy.items():
        assert len(users) == len(reference), policy_name
        for actual, expected in zip(users, reference):
            assert actual["x_m"] == expected["x_m"], policy_name
            assert actual["y_m"] == expected["y_m"], policy_name
            assert actual["task_data_mb"] == expected["task_data_mb"], policy_name


def test_first_recorded_satellite_position_is_initial_position():
    trace = generate_rollout_trace(steps=1, seed=321, load_models=False, device="cpu")

    expected_x = trace["config"]["sat_initial_x_m"]
    for policy_name, policy_trace in trace["policies"].items():
        actual_x = policy_trace["steps"][0]["satellite"]["x_m"]
        assert actual_x == expected_x, policy_name

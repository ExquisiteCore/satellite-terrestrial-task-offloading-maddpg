from pathlib import Path

import pandas as pd

import plot_results


def test_plot_generates_chinese_presentation_figures(tmp_path, monkeypatch):
    csv_dir = tmp_path / "csv"
    figures_dir = tmp_path / "figures"
    csv_dir.mkdir()
    figures_dir.mkdir()

    pd.DataFrame(
        [
            {"episode": 1, "avg_reward": -100.0},
            {"episode": 2, "avg_reward": -80.0},
            {"episode": 3, "avg_reward": -60.0},
            {"episode": 4, "avg_reward": -40.0},
            {"episode": 5, "avg_reward": -20.0},
        ]
    ).to_csv(csv_dir / "maddpg_train_log.csv", index=False)
    pd.DataFrame(
        [
            {
                "algorithm": "All Local",
                "avg_reward": -90.0,
                "avg_delay": 25.0,
                "avg_energy": 30.0,
                "success_rate": 0.2,
                "avg_completed_tasks": 5.0,
                "avg_unfinished_tasks": 15.0,
                "benefit_user_ratio": 0.0,
                "avg_local_ratio": 1.0,
                "avg_bs_ratio": 0.0,
                "avg_sat_ratio": 0.0,
            },
            {
                "algorithm": "Random",
                "avg_reward": -110.0,
                "avg_delay": 40.0,
                "avg_energy": 35.0,
                "success_rate": 0.1,
                "avg_completed_tasks": 2.0,
                "avg_unfinished_tasks": 18.0,
                "benefit_user_ratio": 0.25,
                "avg_local_ratio": 0.3,
                "avg_bs_ratio": 0.4,
                "avg_sat_ratio": 0.3,
            },
            {
                "algorithm": "DQN",
                "avg_reward": -55.0,
                "avg_delay": 18.0,
                "avg_energy": 12.0,
                "success_rate": 0.6,
                "avg_completed_tasks": 12.0,
                "avg_unfinished_tasks": 8.0,
                "benefit_user_ratio": 0.5,
                "avg_local_ratio": 0.0,
                "avg_bs_ratio": 1.0,
                "avg_sat_ratio": 0.0,
            },
            {
                "algorithm": "MADDPG",
                "avg_reward": -35.0,
                "avg_delay": 14.0,
                "avg_energy": 10.0,
                "success_rate": 0.7,
                "avg_completed_tasks": 14.0,
                "avg_unfinished_tasks": 6.0,
                "benefit_user_ratio": 0.75,
                "avg_local_ratio": 0.2,
                "avg_bs_ratio": 0.6,
                "avg_sat_ratio": 0.2,
            },
        ]
    ).to_csv(csv_dir / "evaluation_summary.csv", index=False)
    pd.DataFrame(
        [
            {"task_load_factor": 0.6, "algorithm": "All Local", "avg_completed_tasks": 6.0},
            {"task_load_factor": 0.6, "algorithm": "Random", "avg_completed_tasks": 3.0},
            {"task_load_factor": 0.6, "algorithm": "DQN", "avg_completed_tasks": 10.0},
            {"task_load_factor": 0.6, "algorithm": "MADDPG", "avg_completed_tasks": 12.0},
            {"task_load_factor": 1.2, "algorithm": "All Local", "avg_completed_tasks": 4.0},
            {"task_load_factor": 1.2, "algorithm": "Random", "avg_completed_tasks": 2.0},
            {"task_load_factor": 1.2, "algorithm": "DQN", "avg_completed_tasks": 8.0},
            {"task_load_factor": 1.2, "algorithm": "MADDPG", "avg_completed_tasks": 11.0},
        ]
    ).to_csv(csv_dir / "sensitivity_summary.csv", index=False)

    monkeypatch.setattr(plot_results, "CSV_DIR", csv_dir)
    monkeypatch.setattr(plot_results, "FIGURES_DIR", figures_dir)

    plot_results.plot()

    expected = [
        "reward_curve.png",
        "avg_delay_comparison.png",
        "avg_energy_comparison.png",
        "success_rate_comparison.png",
        "offload_ratio_maddpg.png",
        "avg_reward_comparison.png",
        "task_completion_summary.png",
        "benefit_user_ratio_comparison.png",
        "task_load_sensitivity.png",
    ]
    for filename in expected:
        path = figures_dir / filename
        assert path.exists(), filename
        assert path.stat().st_size > 0

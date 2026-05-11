from __future__ import annotations

import pandas as pd
import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt

from config import CSV_DIR, FIGURES_DIR
from utils.logger import ensure_result_dirs


def save_bar(df: pd.DataFrame, column: str, ylabel: str, output_name: str) -> None:
    plt.figure(figsize=(7, 4))
    plt.bar(df["algorithm"], df[column], color=["#4b5563", "#0f766e", "#2563eb", "#b45309"])
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / output_name, dpi=300)
    plt.close()


def plot() -> None:
    ensure_result_dirs()
    train_path = CSV_DIR / "maddpg_train_log.csv"
    eval_path = CSV_DIR / "evaluation_summary.csv"

    if train_path.exists():
        train_df = pd.read_csv(train_path)
        plt.figure(figsize=(7, 4))
        plt.plot(train_df["episode"], train_df["avg_reward"], label="MADDPG")
        if len(train_df) >= 5:
            plt.plot(
                train_df["episode"],
                train_df["avg_reward"].rolling(window=5, min_periods=1).mean(),
                label="Moving Avg",
            )
        plt.xlabel("Episode")
        plt.ylabel("Average Reward")
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "reward_curve.png", dpi=300)
        plt.close()

    if eval_path.exists():
        eval_df = pd.read_csv(eval_path)
        save_bar(eval_df, "avg_delay", "Average Delay (s)", "avg_delay_comparison.png")
        save_bar(eval_df, "avg_energy", "Average Energy (J)", "avg_energy_comparison.png")
        save_bar(eval_df, "success_rate", "Success Rate", "success_rate_comparison.png")

        maddpg_row = eval_df[eval_df["algorithm"] == "MADDPG"].iloc[0]
        plt.figure(figsize=(6, 4))
        plt.bar(
            ["Local", "Base Station", "Satellite"],
            [maddpg_row["avg_local_ratio"], maddpg_row["avg_bs_ratio"], maddpg_row["avg_sat_ratio"]],
            color=["#4b5563", "#2563eb", "#b45309"],
        )
        plt.ylabel("Average Offloading Ratio")
        plt.ylim(0.0, 1.0)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "offload_ratio_maddpg.png", dpi=300)
        plt.close()


def main() -> None:
    plot()


if __name__ == "__main__":
    main()

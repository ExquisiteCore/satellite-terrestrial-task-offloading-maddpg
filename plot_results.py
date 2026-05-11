from __future__ import annotations

import pandas as pd
import matplotlib

matplotlib.use("Agg")

from matplotlib import font_manager
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from config import CSV_DIR, FIGURES_DIR
from utils.logger import ensure_result_dirs


TITLE_COLOR = "#0f2f4a"
GRID_COLOR = "#d9e2ec"
TEXT_COLOR = "#1f2937"
BAR_COLOR = "#1f78b4"
SECONDARY_COLOR = "#f59e0b"
TERTIARY_COLOR = "#2ca02c"
QUATERNARY_COLOR = "#7c3aed"

ALGORITHM_LABELS = {
    "All Local": "仅本地执行",
    "Random": "随机卸载",
    "DQN": "DQN",
    "MADDPG": "MADDPG",
}


def configure_chinese_fonts() -> None:
    available = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in ("Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC"):
        if font_name in available:
            plt.rcParams["font.sans-serif"] = [font_name]
            break
    plt.rcParams["axes.unicode_minus"] = False


def begin_figure(title: str, subtitle: str | None = None, figsize: tuple[float, float] = (9, 5)):
    configure_chinese_fonts()
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#f7f9fc")
    fig.subplots_adjust(top=0.78, left=0.12, right=0.96, bottom=0.16)
    fig.add_artist(Rectangle((0, 0.88), 1, 0.12, transform=fig.transFigure, color=TITLE_COLOR, zorder=-1))
    fig.text(0.035, 0.925, title, color="white", fontsize=16, weight="bold", va="center")
    if subtitle:
        fig.text(0.035, 0.842, subtitle, color="#64748b", fontsize=10, va="center")
    ax.set_facecolor("white")
    ax.grid(True, axis="y", color=GRID_COLOR, linewidth=0.8, alpha=0.75)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#94a3b8")
    ax.spines["bottom"].set_color("#94a3b8")
    ax.tick_params(colors=TEXT_COLOR, labelsize=10)
    return fig, ax


def add_bar_labels(ax, bars, fmt: str = "{:.2f}") -> None:
    ymin, ymax = ax.get_ylim()
    offset = (ymax - ymin) * 0.018
    for bar in bars:
        value = bar.get_height()
        va = "bottom" if value >= 0 else "top"
        y = value + offset if value >= 0 else value - offset
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            fmt.format(value),
            ha="center",
            va=va,
            fontsize=9,
            color=TEXT_COLOR,
        )


def localized_algorithms(df: pd.DataFrame) -> list[str]:
    return [ALGORITHM_LABELS.get(name, name) for name in df["algorithm"]]


def save_bar(df: pd.DataFrame, column: str, title: str, ylabel: str, output_name: str, fmt: str = "{:.2f}") -> None:
    fig, ax = begin_figure(title, "不同卸载策略下的系统性能对比")
    labels = localized_algorithms(df)
    colors = [BAR_COLOR if name == "MADDPG" else "#6b9ecf" for name in df["algorithm"]]
    bars = ax.bar(labels, df[column], color=colors, width=0.56)
    ax.set_ylabel(ylabel, fontsize=11, color=TEXT_COLOR)
    ax.set_xlabel("对比算法", fontsize=11, color=TEXT_COLOR)
    ax.margins(y=0.15)
    add_bar_labels(ax, bars, fmt=fmt)
    fig.savefig(FIGURES_DIR / output_name, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_reward_curve(train_df: pd.DataFrame) -> None:
    fig, ax = begin_figure("训练收敛曲线", "展示 MADDPG 训练过程中平均奖励的变化趋势")
    ax.plot(
        train_df["episode"],
        train_df["avg_reward"],
        color="#9cc9de",
        linewidth=1.2,
        alpha=0.65,
        label="单回合平均奖励",
    )
    if len(train_df) >= 5:
        ax.plot(
            train_df["episode"],
            train_df["avg_reward"].rolling(window=5, min_periods=1).mean(),
            color=SECONDARY_COLOR,
            linewidth=2.0,
            label="5回合滑动平均",
        )
    if len(train_df) >= 10:
        ax.plot(
            train_df["episode"],
            train_df["avg_reward"].rolling(window=10, min_periods=1).mean(),
            color=TERTIARY_COLOR,
            linewidth=2.0,
            label="10回合滑动平均",
        )
    ax.set_xlabel("训练回合", fontsize=11, color=TEXT_COLOR)
    ax.set_ylabel("平均奖励", fontsize=11, color=TEXT_COLOR)
    ax.legend(frameon=False, loc="best", fontsize=10)
    fig.savefig(FIGURES_DIR / "reward_curve.png", dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_offload_ratio(eval_df: pd.DataFrame) -> None:
    maddpg_row = eval_df[eval_df["algorithm"] == "MADDPG"].iloc[0]
    fig, ax = begin_figure("MADDPG 卸载比例", "展示训练后策略在本地、地面基站和低轨卫星之间的任务分配")
    labels = ["本地计算", "地面基站MEC", "低轨卫星MEC"]
    values = [maddpg_row["avg_local_ratio"], maddpg_row["avg_bs_ratio"], maddpg_row["avg_sat_ratio"]]
    bars = ax.bar(labels, values, color=[SECONDARY_COLOR, BAR_COLOR, QUATERNARY_COLOR], width=0.56)
    ax.set_ylabel("平均卸载比例", fontsize=11, color=TEXT_COLOR)
    ax.set_xlabel("卸载目的地", fontsize=11, color=TEXT_COLOR)
    ax.set_ylim(0.0, max(1.0, max(values) * 1.2))
    add_bar_labels(ax, bars, fmt="{:.2%}")
    fig.savefig(FIGURES_DIR / "offload_ratio_maddpg.png", dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot() -> None:
    ensure_result_dirs()
    train_path = CSV_DIR / "maddpg_train_log.csv"
    eval_path = CSV_DIR / "evaluation_summary.csv"

    if train_path.exists():
        train_df = pd.read_csv(train_path)
        plot_reward_curve(train_df)

    if eval_path.exists():
        eval_df = pd.read_csv(eval_path)
        save_bar(eval_df, "avg_delay", "平均时延对比", "平均时延 / s", "avg_delay_comparison.png")
        save_bar(eval_df, "avg_energy", "平均能耗对比", "平均能耗 / J", "avg_energy_comparison.png")
        save_bar(eval_df, "success_rate", "任务完成率对比", "任务完成率", "success_rate_comparison.png", fmt="{:.2%}")
        plot_offload_ratio(eval_df)


def main() -> None:
    plot()


if __name__ == "__main__":
    main()

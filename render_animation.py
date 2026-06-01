from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

from matplotlib import animation
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from config import VISUALIZATION_DIR


COLORS = {
    "local": "#e6a23c",
    "bs": "#2077b4",
    "sat": "#5b6fb5",
    "user": "#17885b",
    "failed": "#bc3b3b",
    "grid": "#d8e1ec",
    "text": "#172033",
    "muted": "#637083",
}


def load_trace(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def bounds_for_trace(trace: dict[str, Any], policy_names: list[str]) -> tuple[float, float, float, float]:
    cfg = trace["config"]
    xs = [-cfg["area_size_m"] / 2, cfg["area_size_m"] / 2, cfg["bs_position_x_m"]]
    ys = [-cfg["area_size_m"] / 2, cfg["area_size_m"] / 2, cfg["bs_position_y_m"]]
    for policy_name in policy_names:
        for step in trace["policies"][policy_name]["steps"]:
            xs.append(step["satellite"]["ground_x_m"])
            ys.append(step["satellite"]["ground_y_m"])
            for user in step["users"]:
                xs.append(user["x_m"])
                ys.append(user["y_m"])
    padding = cfg["area_size_m"] * 0.25
    return min(xs) - padding, max(xs) + padding, min(ys) - padding, max(ys) + padding


def draw_policy_step(ax: Axes, trace: dict[str, Any], policy_name: str, frame_index: int, bounds) -> None:
    steps = trace["policies"][policy_name]["steps"]
    step = steps[min(frame_index, len(steps) - 1)]
    min_x, max_x, min_y, max_y = bounds

    ax.clear()
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor("#f8fbfe")
    ax.grid(True, color=COLORS["grid"], linewidth=0.7, alpha=0.7)
    ax.tick_params(labelsize=8, colors=COLORS["muted"])
    ax.set_title(policy_name, fontsize=12, color=COLORS["text"], pad=8)

    sat_track_x = [item["satellite"]["ground_x_m"] for item in steps]
    sat_track_y = [item["satellite"]["ground_y_m"] for item in steps]
    ax.plot(sat_track_x, sat_track_y, color=COLORS["sat"], linestyle="--", linewidth=1.3, alpha=0.45)

    bs_x = step["base_station"]["x_m"]
    bs_y = step["base_station"]["y_m"]
    sat_x = step["satellite"]["ground_x_m"]
    sat_y = step["satellite"]["ground_y_m"]

    for user in step["users"]:
        if user["action"][1] > 0.02:
            ax.plot(
                [user["x_m"], bs_x],
                [user["y_m"], bs_y],
                color=COLORS["bs"],
                linewidth=0.8 + user["action"][1] * 3.0,
                alpha=0.2 + user["action"][1] * 0.65,
            )
        if user["action"][2] > 0.02:
            ax.plot(
                [user["x_m"], sat_x],
                [user["y_m"], sat_y],
                color=COLORS["sat"],
                linewidth=0.8 + user["action"][2] * 3.0,
                alpha=0.2 + user["action"][2] * 0.65,
            )

    ax.scatter([bs_x], [bs_y], s=120, color=COLORS["bs"], edgecolor="white", linewidth=1.2, zorder=4)
    ax.text(bs_x, bs_y, " BS", color=COLORS["text"], fontsize=9, va="bottom")
    ax.scatter([sat_x], [sat_y], s=130, color=COLORS["sat"], edgecolor="white", linewidth=1.2, zorder=4)
    ax.text(sat_x, sat_y, " LEO", color=COLORS["text"], fontsize=9, va="bottom")

    for user in step["users"]:
        color = COLORS["user"] if user["success"] else COLORS["failed"]
        size = 45 + user["task_data_mb"] * 15
        ax.scatter([user["x_m"]], [user["y_m"]], s=size, color=color, edgecolor="white", linewidth=1.0, zorder=5)
        ax.text(user["x_m"], user["y_m"], f" U{user['id'] + 1}", color=COLORS["text"], fontsize=8, va="center")

    metrics = step["metrics"]
    summary = (
        f"step {step['step'] + 1}/{len(steps)}\n"
        f"delay {metrics['avg_delay']:.2f}s | energy {metrics['avg_energy']:.2f}J\n"
        f"success {metrics['success_rate'] * 100:.1f}% | reward {metrics['avg_reward']:.2f}"
    )
    ax.text(
        0.02,
        0.02,
        summary,
        transform=ax.transAxes,
        fontsize=8,
        color=COLORS["text"],
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": COLORS["grid"], "alpha": 0.9, "boxstyle": "round,pad=0.35"},
    )


def selected_policies(trace: dict[str, Any], policy: str) -> list[str]:
    if policy.lower() == "all":
        return list(trace["policies"].keys())
    if policy not in trace["policies"]:
        available = ", ".join(trace["policies"].keys())
        raise ValueError(f"unknown policy {policy!r}; available: {available}")
    return [policy]


def render_animation(
    trace: dict[str, Any],
    policy: str,
    output_path: str | Path,
    fps: int = 5,
    dpi: int = 150,
) -> Path:
    policies = selected_policies(trace, policy)
    max_frames = max(len(trace["policies"][name]["steps"]) for name in policies)
    bounds = bounds_for_trace(trace, policies)

    if len(policies) == 1:
        fig, axes = plt.subplots(1, 1, figsize=(8, 5.6))
        axes_list = [axes]
    else:
        fig, axes_grid = plt.subplots(2, 2, figsize=(11, 8.5))
        axes_list = list(axes_grid.flat)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.05, right=0.98, top=0.94, bottom=0.06, hspace=0.22, wspace=0.16)

    def update(frame_index: int):
        for ax, policy_name in zip(axes_list, policies):
            draw_policy_step(ax, trace, policy_name, frame_index, bounds)
        for ax in axes_list[len(policies) :]:
            ax.axis("off")
        fig.suptitle("Satellite-Terrestrial Task Offloading Rollout", fontsize=14, color=COLORS["text"])
        return axes_list

    anim = animation.FuncAnimation(fig, update, frames=max_frames, interval=1000 / fps, blit=False)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.suffix.lower() == ".gif":
        anim.save(target, writer=animation.PillowWriter(fps=fps), dpi=dpi)
    elif target.suffix.lower() == ".mp4":
        anim.save(target, writer=animation.FFMpegWriter(fps=fps), dpi=dpi)
    else:
        raise ValueError("output path must end with .gif or .mp4")
    plt.close(fig)
    return target


def default_output_path(policy: str, output_format: str) -> Path:
    safe_policy = "all_policies" if policy.lower() == "all" else policy.replace(" ", "_")
    return VISUALIZATION_DIR / f"{safe_policy}_rollout.{output_format}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Render rollout trace animations.")
    parser.add_argument("--trace", type=Path, default=VISUALIZATION_DIR / "rollout_trace.json")
    parser.add_argument("--policy", default="MADDPG")
    parser.add_argument("--format", choices=["gif", "mp4"], default="gif")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    output = args.output or default_output_path(args.policy, args.format)
    trace = load_trace(args.trace)
    output_path = render_animation(trace, args.policy, output, fps=args.fps, dpi=args.dpi)
    print(f"Wrote animation to {output_path}")


if __name__ == "__main__":
    main()

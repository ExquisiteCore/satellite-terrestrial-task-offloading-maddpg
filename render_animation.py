from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

from matplotlib import animation
from matplotlib import font_manager
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
    "earth": "#1d5f8f",
    "earth_dark": "#123f64",
    "land": "#2b8c67",
    "orbit": "#9fb3c8",
    "text": "#172033",
    "muted": "#637083",
}


def configure_chinese_fonts() -> None:
    available = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in ("Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC"):
        if font_name in available:
            plt.rcParams["font.sans-serif"] = [font_name]
            break
    plt.rcParams["axes.unicode_minus"] = False


def load_trace(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def orbit_extent(trace: dict[str, Any], policy_names: list[str]) -> float:
    max_radius = 1.2
    for policy_name in policy_names:
        for step in trace["policies"][policy_name]["steps"]:
            max_radius = max(max_radius, float(step["orbit_view"]["satellite"]["orbit_radius_norm"]))
    return max_radius + 0.18


def draw_earth(ax: Axes) -> None:
    earth = plt.Circle((0.0, 0.0), 1.0, color=COLORS["earth"], ec="white", lw=1.4, zorder=2)
    ax.add_patch(earth)
    land_specs = [
        (-0.34, -0.22, 0.28, 0.12, 18),
        (0.22, -0.08, 0.32, 0.16, -12),
        (-0.08, 0.28, 0.34, 0.12, 8),
    ]
    for x, y, width, height, angle in land_specs:
        patch = matplotlib.patches.Ellipse(
            (x, y),
            width,
            height,
            angle=angle,
            color=COLORS["land"],
            alpha=0.35,
            zorder=3,
        )
        ax.add_patch(patch)
    for scale in (0.32, 0.55, 0.78):
        ax.add_patch(
            matplotlib.patches.Ellipse(
                (0.0, 0.0),
                2.0,
                2.0 * scale,
                fill=False,
                ec="white",
                lw=0.6,
                alpha=0.22,
                zorder=4,
            )
        )


def annotate_radial(ax: Axes, x: float, y: float, label: str, fontsize: int = 8) -> None:
    length = max(1e-6, (x**2 + y**2) ** 0.5)
    label_x = x + (x / length) * 0.08
    label_y = y + (y / length) * 0.08
    ha = "left" if x >= 0 else "right"
    ax.text(label_x, label_y, label, color=COLORS["text"], fontsize=fontsize, va="center", ha=ha)


def draw_policy_step(ax: Axes, trace: dict[str, Any], policy_name: str, frame_index: int, extent: float) -> None:
    steps = trace["policies"][policy_name]["steps"]
    step = steps[min(frame_index, len(steps) - 1)]
    orbit = step["orbit_view"]

    ax.clear()
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor("#f8fbfe")
    ax.grid(True, color=COLORS["grid"], linewidth=0.7, alpha=0.7)
    ax.tick_params(labelsize=8, colors=COLORS["muted"])
    ax.set_title(policy_name, fontsize=12, color=COLORS["text"], pad=8)
    ax.set_xticks([])
    ax.set_yticks([])

    orbit_radius_norm = orbit["satellite"]["orbit_radius_norm"]
    ax.add_patch(
        plt.Circle((0.0, 0.0), orbit_radius_norm, fill=False, ec=COLORS["orbit"], lw=1.4, ls="--", zorder=1)
    )
    draw_earth(ax)

    bs_x = orbit["base_station"]["x_norm"]
    bs_y = orbit["base_station"]["y_norm"]
    sat_x = orbit["satellite"]["x_norm"]
    sat_y = orbit["satellite"]["y_norm"]
    orbit_users = {user["id"]: user for user in orbit["users"]}

    for user in step["users"]:
        orbit_user = orbit_users[user["id"]]
        user_x = orbit_user["x_norm"]
        user_y = orbit_user["y_norm"]
        if user["action"][1] > 0.02:
            ax.plot(
                [user_x, bs_x],
                [user_y, bs_y],
                color=COLORS["bs"],
                linewidth=0.8 + user["action"][1] * 3.0,
                alpha=0.2 + user["action"][1] * 0.65,
                zorder=5,
            )
        if user["action"][2] > 0.02:
            ax.plot(
                [user_x, sat_x],
                [user_y, sat_y],
                color=COLORS["sat"],
                linewidth=0.8 + user["action"][2] * 3.0,
                alpha=0.2 + user["action"][2] * 0.65,
                zorder=5,
            )

    ax.scatter([bs_x], [bs_y], s=120, color=COLORS["bs"], edgecolor="white", linewidth=1.2, zorder=4)
    annotate_radial(ax, bs_x, bs_y, "BS", fontsize=9)
    ax.scatter([sat_x], [sat_y], s=130, color=COLORS["sat"], edgecolor="white", linewidth=1.2, zorder=4)
    annotate_radial(ax, sat_x, sat_y, "LEO", fontsize=9)

    for user in step["users"]:
        orbit_user = orbit_users[user["id"]]
        color = COLORS["user"] if user["success"] else COLORS["failed"]
        size = 45 + user["task_data_mb"] * 15
        ax.scatter([orbit_user["x_norm"]], [orbit_user["y_norm"]], s=size, color=color, edgecolor="white", linewidth=1.0, zorder=6)
        annotate_radial(ax, orbit_user["x_norm"], orbit_user["y_norm"], f"U{user['id'] + 1}", fontsize=8)

    metrics = step["metrics"]
    summary = (
        f"step {step['step'] + 1}/{len(steps)} | altitude {orbit['satellite']['altitude_km']:.0f} km\n"
        f"delay {metrics['avg_delay']:.2f}s | energy {metrics['avg_energy']:.2f}J\n"
        f"success {metrics['success_rate'] * 100:.1f}% | reward {metrics['avg_reward']:.2f}"
    )
    ax.text(
        0.02,
        0.03,
        summary,
        transform=ax.transAxes,
        fontsize=8,
        color=COLORS["text"],
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": COLORS["grid"], "alpha": 0.9, "boxstyle": "round,pad=0.35"},
    )
    ax.text(
        0.02,
        0.96,
        "动画展示完整绕行；指标仍使用仿真距离模型",
        transform=ax.transAxes,
        fontsize=7,
        color=COLORS["muted"],
        va="top",
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
    configure_chinese_fonts()
    policies = selected_policies(trace, policy)
    max_frames = max(len(trace["policies"][name]["steps"]) for name in policies)
    extent = orbit_extent(trace, policies)

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
            draw_policy_step(ax, trace, policy_name, frame_index, extent)
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

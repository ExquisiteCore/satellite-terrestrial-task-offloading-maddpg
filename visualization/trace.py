from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from algorithms.baselines import all_local_policy, random_policy
from algorithms.dqn import DQNAgent
from algorithms.maddpg import MADDPG
from config import EnvConfig, MODELS_DIR, VISUALIZATION_DIR
from envs.offloading_env import OffloadingEnv, normalize_actions
from evaluate import maybe_load_model


DEFAULT_POLICY_NAMES = ("MADDPG", "DQN", "Random", "All Local")
EARTH_RADIUS_KM = 6371.0
Policy = Callable[[np.ndarray, int], np.ndarray]


def _to_float(value: Any) -> float:
    return float(np.asarray(value).item())


def _to_bool(value: Any) -> bool:
    return bool(np.asarray(value).item())


def _surface_angle_from_xy(x_m: float, y_m: float) -> float:
    angle = float(np.arctan2(y_m, x_m))
    if angle < 0.0:
        angle += float(2.0 * np.pi)
    return angle


def _unit_point(angle_rad: float, radius_scale: float = 1.0) -> dict[str, float]:
    return {
        "x_norm": float(np.cos(angle_rad) * radius_scale),
        "y_norm": float(np.sin(angle_rad) * radius_scale),
    }


def _satellite_orbit_angle(env: OffloadingEnv, step: int) -> float:
    total_steps = max(1, env.config.episode_steps)
    initial_angle = -0.55 * np.pi
    presentation_sweep = 2.0 * np.pi
    return float((initial_angle + presentation_sweep * step / total_steps) % (2.0 * np.pi))


def _orbit_view(env: OffloadingEnv, step: int) -> dict[str, Any]:
    cfg = env.config
    orbit_radius_km = EARTH_RADIUS_KM + cfg.sat_altitude_m / 1000.0
    orbit_scale = orbit_radius_km / EARTH_RADIUS_KM
    sat_angle = _satellite_orbit_angle(env, step)
    satellite = {
        **_unit_point(sat_angle, orbit_scale),
        "orbit_angle_rad": sat_angle,
        "altitude_km": float(cfg.sat_altitude_m / 1000.0),
        "orbit_radius_norm": float(orbit_scale),
    }
    bs_angle = _surface_angle_from_xy(cfg.bs_position_x_m, cfg.bs_position_y_m)
    users = []
    for user_idx in range(env.num_users):
        angle = _surface_angle_from_xy(
            _to_float(env.state["user_xy_m"][user_idx, 0]),
            _to_float(env.state["user_xy_m"][user_idx, 1]),
        )
        users.append(
            {
                "id": user_idx,
                "surface_angle_rad": angle,
                **_unit_point(angle, 1.0),
            }
        )
    return {
        "earth_radius_km": EARTH_RADIUS_KM,
        "orbit_radius_km": float(orbit_radius_km),
        "model_note": "简化轨道展示视图：卫星动画按一轮仿真展示完整绕行，通信指标仍使用仿真器距离模型计算。",
        "satellite": satellite,
        "base_station": {
            "surface_angle_rad": bs_angle,
            **_unit_point(bs_angle, 1.0),
        },
        "users": users,
    }


def _policy_map(
    env: OffloadingEnv,
    config: EnvConfig,
    load_models: bool,
    device: str,
    dqn_checkpoint: str | Path,
    maddpg_checkpoint: str | Path,
) -> dict[str, Policy]:
    dqn = DQNAgent(obs_dim=env.obs_dim, seed=config.seed, device=device)
    maddpg = MADDPG(
        num_users=env.num_users,
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        seed=config.seed,
        device=device,
    )
    maybe_load_model(dqn, dqn_checkpoint, load_models)
    maybe_load_model(maddpg, maddpg_checkpoint, load_models)
    return {
        "MADDPG": lambda obs, step: maddpg.act(obs, noise_std=0.0),
        "DQN": lambda obs, step: dqn.act(obs, epsilon=0.0)[0],
        "Random": lambda obs, step: random_policy(env.num_users, seed=config.seed + step),
        "All Local": lambda obs, step: all_local_policy(env.num_users),
    }


def _record_step(
    env: OffloadingEnv,
    step: int,
    actions: np.ndarray,
    metrics: dict[str, np.ndarray],
    info: dict[str, Any],
) -> dict[str, Any]:
    cfg = env.config
    split = normalize_actions(actions)
    users = []
    for user_idx in range(env.num_users):
        users.append(
            {
                "id": user_idx,
                "x_m": _to_float(env.state["user_xy_m"][user_idx, 0]),
                "y_m": _to_float(env.state["user_xy_m"][user_idx, 1]),
                "task_data_mb": _to_float(env.state["task_data_mb"][user_idx]),
                "cycles_per_bit": _to_float(env.state["cycles_per_bit"][user_idx]),
                "deadline_s": _to_float(env.state["deadline_s"][user_idx]),
                "local_freq_ghz": _to_float(env.state["local_freq_ghz"][user_idx]),
                "bs_distance_m": _to_float(env.state["bs_distance_m"][user_idx]),
                "sat_distance_m": _to_float(env.state["sat_distance_m"][user_idx]),
                "bs_rate_bps": _to_float(env.state["bs_rate_bps"][user_idx]),
                "sat_rate_bps": _to_float(env.state["sat_rate_bps"][user_idx]),
                "action": [_to_float(value) for value in split[user_idx]],
                "delay_s": _to_float(metrics["delay"][user_idx]),
                "energy_j": _to_float(metrics["energy"][user_idx]),
                "local_delay_s": _to_float(metrics["local_delay"][user_idx]),
                "bs_delay_s": _to_float(metrics["bs_delay"][user_idx]),
                "sat_delay_s": _to_float(metrics["sat_delay"][user_idx]),
                "sat_propagation_delay_s": _to_float(metrics["sat_propagation_delay"][user_idx]),
                "success": not _to_bool(metrics["failed"][user_idx]),
            }
        )
    return {
        "step": step,
        "base_station": {
            "x_m": float(cfg.bs_position_x_m),
            "y_m": float(cfg.bs_position_y_m),
        },
        "satellite": {
            "x_m": _to_float(env.state["sat_position_m"][0]),
            "y_m": _to_float(env.state["sat_position_m"][1]),
            "z_m": _to_float(env.state["sat_position_m"][2]),
            "ground_x_m": _to_float(env.state["sat_position_m"][0]),
            "ground_y_m": _to_float(env.state["sat_position_m"][1]),
        },
        "orbit_view": _orbit_view(env, step),
        "metrics": {
            "avg_delay": float(info["avg_delay"]),
            "avg_energy": float(info["avg_energy"]),
            "success_rate": float(info["success_rate"]),
            "avg_reward": float(info["avg_reward"]),
            "avg_local_ratio": float(info["avg_local_ratio"]),
            "avg_bs_ratio": float(info["avg_bs_ratio"]),
            "avg_sat_ratio": float(info["avg_sat_ratio"]),
            "deadline_violation": float(info["deadline_violation"]),
        },
        "users": users,
    }


def _rollout_policy(config: EnvConfig, policy_name: str, policy: Policy, steps: int) -> dict[str, Any]:
    env = OffloadingEnv(config)
    obs = env.reset(seed=config.seed)
    recorded_steps = []
    for step in range(steps):
        actions = normalize_actions(policy(obs, step))
        metrics = env._compute_metrics(actions)
        total_delay_cost = float(np.mean(metrics["delay"]))
        total_energy_cost = float(np.mean(metrics["energy"]))
        deadline_violation = float(np.mean(metrics["failed"]))
        team_reward = -(
            env.config.reward_delay_weight * total_delay_cost
            + env.config.reward_energy_weight * total_energy_cost
            + env.config.deadline_penalty * deadline_violation
        )
        info = {
            "avg_delay": total_delay_cost,
            "avg_energy": total_energy_cost,
            "avg_reward": team_reward,
            "deadline_violation": deadline_violation,
            "success_rate": float(np.mean(1.0 - metrics["failed"])),
            "avg_local_ratio": float(np.mean(actions[:, 0])),
            "avg_bs_ratio": float(np.mean(actions[:, 1])),
            "avg_sat_ratio": float(np.mean(actions[:, 2])),
        }
        recorded_steps.append(_record_step(env, step, actions, metrics, info))
        next_obs, _, done, _ = env.step(actions)
        obs = next_obs
        if done:
            break
    return {
        "name": policy_name,
        "steps": recorded_steps,
    }


def generate_rollout_trace(
    steps: int = 50,
    seed: int = 99,
    load_models: bool = True,
    device: str = "cpu",
    dqn_checkpoint: str | Path = MODELS_DIR / "dqn.pt",
    maddpg_checkpoint: str | Path = MODELS_DIR / "maddpg_best.pt",
) -> dict[str, Any]:
    if steps <= 0:
        raise ValueError("steps must be positive")

    config = EnvConfig(seed=seed, episode_steps=steps)
    policy_env = OffloadingEnv(config)
    policies = _policy_map(
        policy_env,
        config,
        load_models=load_models,
        device=device,
        dqn_checkpoint=dqn_checkpoint,
        maddpg_checkpoint=maddpg_checkpoint,
    )
    policy_traces = {
        name: _rollout_policy(config, name, policies[name], steps)
        for name in DEFAULT_POLICY_NAMES
    }
    return {
        "schema_version": 1,
        "config": asdict(config),
        "policies": policy_traces,
    }


def write_rollout_trace(trace: dict[str, Any], output_path: str | Path = VISUALIZATION_DIR / "rollout_trace.json") -> Path:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
    return target

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from config import EnvConfig


def normalize_actions(actions: np.ndarray) -> np.ndarray:
    """Project raw per-user actions onto the local/base-station/satellite simplex."""
    arr = np.asarray(actions, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] != 3:
        raise ValueError(f"actions must have shape (num_users, 3), got {arr.shape}")

    arr = np.clip(arr, 0.0, None)
    row_sums = arr.sum(axis=1, keepdims=True)
    zero_rows = row_sums.squeeze(axis=1) <= 0.0
    safe = np.divide(arr, row_sums, out=np.zeros_like(arr), where=row_sums > 0.0)
    if np.any(zero_rows):
        safe[zero_rows] = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
    return safe


class OffloadingEnv:
    """Satellite-terrestrial MEC task offloading simulator.

    Each user chooses a split among local execution, base-station MEC, and
    satellite MEC. The model is intentionally compact so the thesis project can
    focus on DRL behavior and reproducible comparisons.
    """

    obs_dim = 10
    action_dim = 3

    def __init__(self, config: EnvConfig | None = None):
        self.config = config or EnvConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.step_count = 0
        self.state: dict[str, np.ndarray] = {}

    @property
    def num_users(self) -> int:
        return self.config.num_users

    def reset(self) -> np.ndarray:
        self.rng = np.random.default_rng(self.config.seed)
        self.step_count = 0
        self._sample_state()
        return self._get_obs()

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool, dict[str, Any]]:
        if not self.state:
            self.reset()

        split = normalize_actions(actions)
        if split.shape[0] != self.num_users:
            raise ValueError(f"expected {self.num_users} user actions, got {split.shape[0]}")

        metrics = self._compute_metrics(split)
        self.step_count += 1
        done = self.step_count >= self.config.episode_steps
        self._advance_state()

        rewards = -(
            self.config.reward_delay_weight * metrics["delay"]
            + self.config.reward_energy_weight * metrics["energy"]
            + self.config.deadline_penalty * metrics["failed"]
        )
        info = {
            "avg_delay": float(np.mean(metrics["delay"])),
            "avg_energy": float(np.mean(metrics["energy"])),
            "success_rate": float(np.mean(1.0 - metrics["failed"])),
            "avg_local_ratio": float(np.mean(split[:, 0])),
            "avg_bs_ratio": float(np.mean(split[:, 1])),
            "avg_sat_ratio": float(np.mean(split[:, 2])),
            "config": asdict(self.config),
        }
        return self._get_obs(), rewards.astype(np.float32), done, info

    def _sample_state(self) -> None:
        cfg = self.config
        self.state = {
            "task_data_mb": self.rng.uniform(cfg.task_data_min_mb, cfg.task_data_max_mb, cfg.num_users),
            "cycles_per_bit": self.rng.uniform(cfg.task_cycles_min, cfg.task_cycles_max, cfg.num_users),
            "deadline_s": self.rng.uniform(cfg.deadline_min_s, cfg.deadline_max_s, cfg.num_users),
            "local_freq_ghz": self.rng.uniform(cfg.local_freq_min_ghz, cfg.local_freq_max_ghz, cfg.num_users),
            "bs_distance_m": self.rng.uniform(cfg.bs_distance_min_m, cfg.bs_distance_max_m, cfg.num_users),
            "sat_distance_m": self.rng.uniform(cfg.sat_distance_min_m, cfg.sat_distance_max_m, cfg.num_users),
        }
        self._update_rates()

    def _advance_state(self) -> None:
        cfg = self.config
        self.state["task_data_mb"] = self.rng.uniform(cfg.task_data_min_mb, cfg.task_data_max_mb, cfg.num_users)
        self.state["cycles_per_bit"] = self.rng.uniform(cfg.task_cycles_min, cfg.task_cycles_max, cfg.num_users)
        self.state["deadline_s"] = self.rng.uniform(cfg.deadline_min_s, cfg.deadline_max_s, cfg.num_users)

        bs_delta = self.rng.normal(0.0, 35.0, cfg.num_users)
        sat_delta = self.rng.normal(0.0, 25_000.0, cfg.num_users)
        self.state["bs_distance_m"] = np.clip(
            self.state["bs_distance_m"] + bs_delta,
            cfg.bs_distance_min_m,
            cfg.bs_distance_max_m,
        )
        self.state["sat_distance_m"] = np.clip(
            self.state["sat_distance_m"] + sat_delta,
            cfg.sat_distance_min_m,
            cfg.sat_distance_max_m,
        )
        self._update_rates()

    def _update_rates(self) -> None:
        cfg = self.config
        bs_gain = 1.0 / np.square(self.state["bs_distance_m"])
        sat_gain = 1.0 / np.square(self.state["sat_distance_m"])
        bs_snr = cfg.tx_power_w * bs_gain / cfg.noise_power_w
        sat_snr = cfg.tx_power_w * sat_gain / cfg.noise_power_w
        self.state["bs_rate_bps"] = np.maximum(cfg.bandwidth_hz * np.log2(1.0 + bs_snr), cfg.min_rate_bps)
        self.state["sat_rate_bps"] = np.maximum(cfg.bandwidth_hz * np.log2(1.0 + sat_snr), cfg.min_rate_bps)

    def _compute_metrics(self, split: np.ndarray) -> dict[str, np.ndarray]:
        cfg = self.config
        data_bits = self.state["task_data_mb"] * 8.0e6
        cycles = data_bits * self.state["cycles_per_bit"]

        local_cycles = split[:, 0] * cycles
        bs_cycles = split[:, 1] * cycles
        sat_cycles = split[:, 2] * cycles

        local_freq = self.state["local_freq_ghz"] * 1.0e9
        local_delay = np.divide(local_cycles, local_freq, out=np.zeros_like(local_cycles), where=local_cycles > 0.0)
        local_energy = cfg.local_energy_coeff * local_cycles * np.square(local_freq)

        bs_total = np.sum(bs_cycles)
        bs_freq = np.zeros(self.num_users)
        if bs_total > 0.0:
            bs_freq = (bs_cycles / bs_total) * cfg.bs_freq_ghz * 1.0e9
        bs_tx_delay = split[:, 1] * data_bits / self.state["bs_rate_bps"]
        bs_compute_delay = np.divide(bs_cycles, bs_freq, out=np.zeros_like(bs_cycles), where=bs_freq > 0.0)
        bs_delay = bs_tx_delay + bs_compute_delay
        bs_energy = cfg.transmit_energy_coeff * cfg.tx_power_w * bs_tx_delay

        sat_total = np.sum(sat_cycles)
        sat_freq = np.zeros(self.num_users)
        if sat_total > 0.0:
            sat_freq = (sat_cycles / sat_total) * cfg.sat_freq_ghz * 1.0e9
        sat_tx_delay = split[:, 2] * data_bits / self.state["sat_rate_bps"]
        sat_compute_delay = np.divide(sat_cycles, sat_freq, out=np.zeros_like(sat_cycles), where=sat_freq > 0.0)
        sat_delay = sat_tx_delay + sat_compute_delay
        sat_energy = cfg.transmit_energy_coeff * cfg.tx_power_w * sat_tx_delay

        delay = np.maximum.reduce([local_delay, bs_delay, sat_delay])
        energy = local_energy + bs_energy + sat_energy
        failed = (delay > self.state["deadline_s"]).astype(np.float64)
        return {"delay": delay, "energy": energy, "failed": failed}

    def _get_obs(self) -> np.ndarray:
        cfg = self.config
        obs = np.column_stack(
            [
                self.state["task_data_mb"] / cfg.task_data_max_mb,
                self.state["cycles_per_bit"] / cfg.task_cycles_max,
                self.state["deadline_s"] / cfg.deadline_max_s,
                self.state["bs_distance_m"] / cfg.bs_distance_max_m,
                self.state["sat_distance_m"] / cfg.sat_distance_max_m,
                self.state["bs_rate_bps"] / np.max(self.state["bs_rate_bps"]),
                self.state["sat_rate_bps"] / np.max(self.state["sat_rate_bps"]),
                self.state["local_freq_ghz"] / cfg.local_freq_max_ghz,
                np.full(self.num_users, cfg.bs_freq_ghz / cfg.bs_freq_ghz),
                np.full(self.num_users, cfg.sat_freq_ghz / cfg.bs_freq_ghz),
            ]
        )
        return np.clip(obs, 0.0, 1.0).astype(np.float32)

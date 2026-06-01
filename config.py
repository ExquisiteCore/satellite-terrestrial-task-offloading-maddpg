from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EnvConfig:
    num_users: int = 6
    episode_steps: int = 50
    seed: int = 42

    task_data_min_mb: float = 0.5
    task_data_max_mb: float = 5.0
    task_cycles_min: float = 500.0
    task_cycles_max: float = 1500.0
    deadline_min_s: float = 5.0
    deadline_max_s: float = 20.0

    local_freq_min_ghz: float = 0.8
    local_freq_max_ghz: float = 1.5
    bs_freq_ghz: float = 12.0
    sat_freq_ghz: float = 8.0
    bs_capacity_cycles_per_slot: float = 45.0e9
    sat_capacity_cycles_per_slot: float = 45.0e9
    mec_overload_penalty: float = 3.0

    bs_distance_min_m: float = 100.0
    bs_distance_max_m: float = 1000.0
    sat_distance_min_m: float = 500_000.0
    sat_distance_max_m: float = 1_200_000.0
    area_size_m: float = 2_000.0
    bs_position_x_m: float = 0.0
    bs_position_y_m: float = 0.0
    sat_altitude_m: float = 600_000.0
    sat_initial_x_m: float = -1_000_000.0
    sat_velocity_mps: float = 7_500.0
    slot_duration_s: float = 1.0
    reference_channel_gain: float = 1e-2
    bs_path_loss_exponent: float = 2.2
    sat_path_loss_exponent: float = 2.0
    light_speed_mps: float = 3.0e8

    bandwidth_hz: float = 10e6
    noise_power_w: float = 1e-13
    tx_power_w: float = 0.5
    local_energy_coeff: float = 1e-27
    transmit_energy_coeff: float = 1.0
    min_rate_bps: float = 1e5

    reward_delay_weight: float = 0.7
    reward_energy_weight: float = 0.3
    deadline_penalty: float = 2.0


@dataclass(frozen=True)
class TrainConfig:
    episodes: int = 100
    batch_size: int = 64
    gamma: float = 0.95
    tau: float = 0.01
    actor_lr: float = 1e-3
    critic_lr: float = 1e-3
    buffer_capacity: int = 50_000
    hidden_dim: int = 128
    exploration_noise: float = 0.15


RESULTS_DIR = Path("results")
MODELS_DIR = RESULTS_DIR / "models"
CSV_DIR = RESULTS_DIR / "csv"
FIGURES_DIR = RESULTS_DIR / "figures"
VISUALIZATION_DIR = RESULTS_DIR / "visualization"

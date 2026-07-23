"""问题上下文对象：保存所有预处理结果，所有算法共享"""
import numpy as np


def validate_global_ptx_max(value) -> float:
    """Validate the global hardware TX-power limit."""
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(
            "Global maximum transmit power must be finite and strictly positive; "
            f"got {value!r}"
        )
    return value


def resolve_global_ptx_max(config: dict) -> float:
    """Resolve the one required global TX-power limit from configuration."""
    channel = (config or {}).get("channel", {}) or {}
    if "p_tx_max" not in channel:
        raise ValueError("channel.p_tx_max is required")
    return validate_global_ptx_max(channel["p_tx_max"])


class ProblemContext:
    def __init__(self, config):
        self.num_candidates: int = 0
        self.num_targets: int = 0
        self.candidate_coords: np.ndarray = None
        self.candidate_points_full: np.ndarray = None
        self.target_coords: np.ndarray = None
        self.T_r: np.ndarray = None
        self.temperature_stats: dict = None
        self.delta_T: np.ndarray = None
        self.P_grid: np.ndarray = None
        self.coverage_matrix: np.ndarray = None
        self.distance_matrix: np.ndarray = None
        self.channel_gain_matrix: np.ndarray = None
        self.ptx_min_matrix: np.ndarray = None
        self.link_feasible_matrix: np.ndarray = None
        self.potential_rate_matrix: np.ndarray = None
        self.p_tx_max: float = resolve_global_ptx_max(config)
        self.neighbor_sets: list = None
        self.nmax: np.ndarray = None
        self.sensor_mask: np.ndarray = None
        self.ap_mask: np.ndarray = None
        self.index_mapping = None
        self.config: dict = config
        self.Rs: float = 1.5
        self.constraint_evaluation_spec = None
        self.constraint_context_signature = None
        self.context_revision = 0

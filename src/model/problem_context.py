"""问题上下文对象：保存所有预处理结果，所有算法共享"""
import numpy as np


class ProblemContext:
    def __init__(self):
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
        self.optimistic_ptx_up_matrix: np.ndarray = None
        self.neighbor_sets: list = None
        self.nmax: np.ndarray = None
        self.sensor_mask: np.ndarray = None
        self.ap_mask: np.ndarray = None
        self.index_mapping = None
        self.config: dict = None
        self.Rs: float = 1.5
        self.constraint_evaluation_spec = None
        self.constraint_context_signature = None
        self.context_revision = 0

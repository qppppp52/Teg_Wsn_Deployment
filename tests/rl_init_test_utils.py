import numpy as np
from src.model.problem_context import ProblemContext
from src.utils.index_mapping import IndexMapping


def make_dummy_ctx():
    ctx = ProblemContext()
    ctx.num_candidates = 5
    ctx.num_targets = 3
    ctx.candidate_coords = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
    ], dtype=float)
    ctx.candidate_points_full = np.column_stack([ctx.candidate_coords, np.arange(5) % 6])
    ctx.target_coords = np.array([[0.1, 0.1, 0.0], [0.8, 0.1, 0.0], [0.2, 0.9, 0.0]], dtype=float)
    ctx.T_r = np.array([340.0, 342.0, 341.0, 339.0, 343.0])
    ctx.P_grid = np.array([0.01, 0.02, 0.015, 0.012, 0.018])
    ctx.coverage_matrix = np.array([
        [1, 0, 1],
        [1, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [0, 1, 0],
    ], dtype=int)
    mat = np.ones((5, 5), dtype=float)
    ctx.channel_gain_matrix = mat
    ctx.potential_rate_matrix = mat * 1.0e6
    ctx.distance_matrix = mat
    ctx.link_feasible_matrix = np.ones((5, 5), dtype=bool)
    ctx.neighbor_sets = [[1, 2], [0, 2], [0, 1], [4], [3]]
    sensor_mask = np.array([True, True, True, True, False])
    ap_mask = np.array([False, True, True, False, True])
    ctx.sensor_mask = sensor_mask
    ctx.ap_mask = ap_mask
    ctx.index_mapping = IndexMapping(ctx.num_candidates, sensor_mask, ap_mask)
    ctx.config = {
        "deployment": {"max_sensors": 3, "max_aps": 1, "min_sensors": 1, "min_aps": 1},
        "temperature": {"source_position": [0.5, 0.5, 0.5]},
        "constraints": {"max_repair_iter": 1},
        "drl_init": {
            "seed": 1,
            "max_steps_per_episode": 6,
            "max_selected_sensors": 3,
            "max_selected_aps": 1,
            "min_selected_sensors": 1,
            "min_selected_aps": 1,
            "priority_noise_std": 0.0,
            "normalization": {"rsum_ref_max": 1.0e6, "cv_ref": 10.0, "repair_iter_ref": 5},
            "reward": {"invalid_action_penalty": 0.2},
            "network": {"hidden_dim": 16, "dropout": 0.0},
        },
    }
    return ctx

"""预处理总控"""
import numpy as np
from src.physics.temperature_field import compute_temperature
from src.physics.teg_model import compute_delta_T, compute_grid_power
from src.physics.channel_model import compute_distance_matrix
from src.preprocessing.coverage_matrix import compute_coverage_matrix
from src.preprocessing.link_matrix import compute_all_link_matrices
from src.preprocessing.energy_filter import filter_sensor_candidates, filter_ap_candidates
from src.preprocessing.iterative_candidate_filter import iterative_filter
from src.heatsink.neighborhood import compute_neighborhoods
from src.utils.index_mapping import IndexMapping
from src.model.problem_context import ProblemContext


def run_preprocessing(scenario, config: dict, seed: int) -> ProblemContext:
    cand = scenario.candidate_points
    cand_coords = cand[:, :3].astype(float)
    K = len(cand)

    T_r = compute_temperature(scenario, config)

    teg_cfg = config.get("teg", {})
    delta_T = compute_delta_T(T_r, config)
    P_grid = compute_grid_power(delta_T, config)

    tcfg = config.get("targets", {})
    Rs = tcfg.get("sensing_radius", 1.5)
    coverage_mat = compute_coverage_matrix(cand_coords, scenario.target_points, Rs)

    link_mats = compute_all_link_matrices(cand_coords, config, seed)

    hs_cfg = config.get("heatsink", {})
    q = hs_cfg.get("neighbor_order", 1)
    gs = config.get("discretization", {}).get("grid_spacing", 1.0)
    neighbor_sets, nmax = compute_neighborhoods(cand, q, grid_spacing=gs)

    sensor_mask = filter_sensor_candidates(P_grid, nmax,
                                            link_mats["ptx_min"],
                                            link_mats["link_feasible"], config)
    ap_mask = filter_ap_candidates(P_grid, nmax,
                                    link_mats["link_feasible"], config)
    sensor_mask, ap_mask = iterative_filter(sensor_mask, ap_mask,
                                             link_mats["link_feasible"])

    idx_map = IndexMapping(K, sensor_mask, ap_mask)

    ctx = ProblemContext()
    ctx.num_candidates = K
    ctx.num_targets = scenario.num_targets
    ctx.candidate_coords = cand_coords
    ctx.candidate_points_full = cand
    ctx.target_coords = scenario.target_points
    ctx.T_r = T_r
    ctx.delta_T = delta_T
    ctx.P_grid = P_grid
    ctx.coverage_matrix = coverage_mat
    ctx.distance_matrix = link_mats["distance"]
    ctx.channel_gain_matrix = link_mats["channel_gain"]
    ctx.ptx_min_matrix = link_mats["ptx_min"]
    ctx.link_feasible_matrix = link_mats["link_feasible"]
    ctx.potential_rate_matrix = link_mats["potential_rate"]
    ctx.optimistic_ptx_up_matrix = link_mats["optimistic_ptx_up"]
    ctx.neighbor_sets = neighbor_sets
    ctx.nmax = nmax
    ctx.sensor_mask = sensor_mask
    ctx.ap_mask = ap_mask
    ctx.index_mapping = idx_map
    ctx.config = config
    ctx.Rs = Rs
    return ctx

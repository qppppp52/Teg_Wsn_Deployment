
import numpy as np


def filter_sensor_candidates(P_grid, nmax, ptx_min_matrix, link_feasible, config):
    scfg = config.get("sensor", {})
    P_min_fixed = scfg.get("P_sens", 0.01) + scfg.get("P_proc", 0.005)
    K = len(P_grid)
    mask = np.ones(K, dtype=bool)
    for r in range(K):
        max_harv = P_grid[r] * nmax[r]
        feasible_aps = np.where(link_feasible[r])[0]
        if len(feasible_aps) > 0:
            best_min_ptx = np.min(ptx_min_matrix[r, feasible_aps])
        else:
            best_min_ptx = float("inf")
        if max_harv < P_min_fixed + best_min_ptx:
            mask[r] = False
    return mask


def filter_ap_candidates(P_grid, nmax, link_feasible, config):
    acfg = config.get("ap", {})
    P_idle = acfg.get("P_idle", 0.05)
    P_proc_ap = acfg.get("P_proc", 0.02)
    P_rx = acfg.get("P_rx", 0.003)
    N_load_ref = 2
    P_ap_ref = P_idle + P_proc_ap + P_rx * N_load_ref
    K = len(P_grid)
    mask = np.ones(K, dtype=bool)
    for r in range(K):
        max_harv = P_grid[r] * nmax[r]
        has_sensor = np.any(link_feasible[:, r])
        if max_harv < P_ap_ref or not has_sensor:
            mask[r] = False
    return mask

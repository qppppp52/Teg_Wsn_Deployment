import numpy as np

def calculate_actual_ptx_upper_bound(solution, sensor_idx, ap_idx, ctx):
    ch = ctx.config["channel"]
    ptx_max = ch["p_tx_max"]
    scfg = ctx.config["sensor"]
    P_fixed = scfg["P_sens"] + scfg["P_proc"]
    n_available = ctx.nmax[sensor_idx]
    n_used = solution.n_sink_sensor[sensor_idx]
    n_free = max(0, n_available - n_used)
    max_energy = n_free * ctx.P_grid[sensor_idx] if n_free > 0 else 0.0
    ptx_up_energy = max_energy - P_fixed
    return min(ptx_max, max(0.0, ptx_up_energy))

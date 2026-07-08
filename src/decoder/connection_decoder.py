import numpy as np

def assign_connections(x, y, ctx):
    K = ctx.num_candidates
    c = np.zeros((K, K), dtype=np.int8)
    ap_capacity = ctx.config["ap"]["C_max"]
    ap_loads = np.zeros(K, dtype=int)
    sensor_ids = np.where(x == 1)[0]
    ap_ids = np.where(y == 1)[0]
    for si in sensor_ids:
        feasible_aps = [(aj, ctx.ptx_min_matrix[si, aj])
                        for aj in ap_ids
                        if ctx.link_feasible_matrix[si, aj] == 1
                        and ap_loads[aj] < ap_capacity]
        if not feasible_aps:
            continue
        feasible_aps.sort(key=lambda t: t[1])
        best_ap = feasible_aps[0][0]
        c[si, best_ap] = 1
        ap_loads[best_ap] += 1
    return c

import numpy as np
from src.physics.node_power_model import compute_sensor_consumption, compute_ap_consumption

def compute_sink_requirements(solution, ctx):
    K = ctx.num_candidates
    for si in range(K):
        if solution.x[si] == 1:
            aps = np.where(solution.c[si] == 1)[0]
            p_tx = solution.p_tx[si, aps[0]] if len(aps) > 0 else 0.0
            P_cons = compute_sensor_consumption(si, p_tx, ctx.config)
            solution.sensor_power_consumption[si] = P_cons
            req = int(np.ceil(P_cons / max(ctx.P_grid[si], 1e-12)))
            solution.n_sink_sensor[si] = min(req, ctx.nmax[si])
    for ai in range(K):
        if solution.y[ai] == 1:
            num_conn = int(np.sum(solution.c[:, ai]))
            P_cons = compute_ap_consumption(num_conn, ctx.config)
            solution.ap_power_consumption[ai] = P_cons
            req = int(np.ceil(P_cons / max(ctx.P_grid[ai], 1e-12)))
            solution.n_sink_ap[ai] = min(req, ctx.nmax[ai])

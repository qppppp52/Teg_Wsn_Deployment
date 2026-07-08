import numpy as np
from src.power.power_allocator import allocate_power

def decode_power(solution, ctx):
    K = ctx.num_candidates
    sensor_ids = np.where(solution.x == 1)[0]
    for si in sensor_ids:
        connected_aps = np.where(solution.c[si] == 1)[0]
        if len(connected_aps) == 0:
            continue
        aj = connected_aps[0]
        ptx_min = ctx.ptx_min_matrix[si, aj]
        ptx = allocate_power(si, aj, solution, ctx)
        solution.p_tx[si, aj] = max(ptx, ptx_min)

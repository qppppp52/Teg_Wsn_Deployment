import numpy as np
from src.physics.rate_model import compute_snr, compute_rate

def compute_throughput(solution, ctx):
    total = 0.0
    for si in range(ctx.num_candidates):
        if solution.x[si] == 0:
            continue
        for aj in range(ctx.num_candidates):
            if solution.c[si, aj] == 1:
                snr = compute_snr(
                    np.array(solution.p_tx[si, aj]),
                    np.array(ctx.channel_gain_matrix[si, aj]),
                    np.array(ctx.distance_matrix[si, aj]),
                    ctx.config)
                total += compute_rate(snr, ctx.config)
    return float(total)

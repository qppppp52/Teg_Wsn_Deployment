import numpy as np

def compute_coverage(solution, ctx):
    sids = np.where(solution.x == 1)[0]
    if len(sids) == 0:
        return 0.0
    covered = np.any(ctx.coverage_matrix[sids], axis=0)
    return float(np.mean(covered))

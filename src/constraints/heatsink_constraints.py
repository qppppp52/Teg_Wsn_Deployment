"""散热片约束检查 — 按建模文件实现：空间不重叠 + 邻域 + 等式约束"""
import numpy as np
from src.heatsink.sink_conflict import check_sink_conflicts, check_spatial_non_overlap


def check_heatsink_constraints(solution, ctx):
    """
    建模文件约束：
    1) 空间不重叠约束：x_k + y_k + Σ_{i≠k}z_ik + Σ_{j≠k}z_jk ≤ 1
    2) 散热片在邻域内
    3) 等式约束：已部署节点 → n_sink = 所需数量；未部署 → 不得占用
    """
    cv = 0.0
    K = ctx.num_candidates

    # 1) 空间不重叠约束
    cv += float(check_spatial_non_overlap(solution, ctx))

    # 2) 散热片必须在邻域内 + 3) 等式约束
    for si in range(K):
        nb = set(ctx.neighbor_sets[si])
        n_alloc = len(solution.z_sink_sensor[si])
        for g in solution.z_sink_sensor[si]:
            if g not in nb:
                cv += 1.0

        if solution.x[si] == 1:
            if n_alloc != solution.n_sink_sensor[si]:
                cv += abs(n_alloc - solution.n_sink_sensor[si])
        else:
            if n_alloc > 0:
                cv += n_alloc

    for ai in range(K):
        nb = set(ctx.neighbor_sets[ai])
        n_alloc = len(solution.z_sink_ap[ai])
        for g in solution.z_sink_ap[ai]:
            if g not in nb:
                cv += 1.0

        if solution.y[ai] == 1:
            if n_alloc != solution.n_sink_ap[ai]:
                cv += abs(n_alloc - solution.n_sink_ap[ai])
        else:
            if n_alloc > 0:
                cv += n_alloc

    return cv

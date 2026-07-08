import numpy as np

def check_link_constraints(solution, ctx):
    """
    建模文件约束检查：
    1. c_ij ≤ x_i, c_ij ≤ y_j  (连接存在性)
    2. Σ_j c_ij == x_i  (一对一关联)
    3. γ_ij ≥ γ_th (链路可达性/SNR)
    4. p_tx_ij ≤ p_tx_max (功率上限)
    """
    cv = 0.0
    K = ctx.num_candidates
    ptx_max = float(ctx.config.get("channel", {}).get("p_tx_max", 0.5))

    for i in range(K):
        for j in range(K):
            if solution.c[i, j] == 1:
                # 连接存在性: c_ij ≤ x_i
                if solution.x[i] != 1:
                    cv += 1.0
                # 连接存在性: c_ij ≤ y_j
                if solution.y[j] != 1:
                    cv += 1.0
                # SNR 约束
                if ctx.link_feasible_matrix[i, j] == 0:
                    cv += 1.0
                # 发射功率上限
                if solution.p_tx[i, j] > ptx_max:
                    cv += 1.0

    # 一对一关联约束: 每个已部署传感器必须且只能连接一个AP
    for si in np.where(solution.x == 1)[0]:
        n_conn = int(np.sum(solution.c[si]))
        if n_conn < 1:
            cv += 1.0
        elif n_conn > 1:
            cv += n_conn - 1

    return cv


def repair_link_constraints(solution, ctx):
    """修复链路约束"""
    sids = np.where(solution.x == 1)[0]
    aids = np.where(solution.y == 1)[0]
    Cmax = ctx.config["ap"]["C_max"]

    # 清理不存在的连接 (c_ij=1但x_i=0或y_j=0)
    for i in range(ctx.num_candidates):
        for j in range(ctx.num_candidates):
            if solution.c[i, j] == 1:
                if solution.x[i] != 1 or solution.y[j] != 1:
                    solution.c[i, j] = 0

    # 对传感器一对一修复
    for si in sids:
        conn = np.where(solution.c[si] == 1)[0]
        if len(conn) > 1:
            # 保留一个最好的
            best = min(conn, key=lambda aj: ctx.ptx_min_matrix[si, aj])
            for aj in conn:
                if aj != best:
                    solution.c[si, aj] = 0

        # 无连接时尝试分配
        needs_reconnect = (np.sum(solution.c[si]) == 0)
        if needs_reconnect:
            loads = np.sum(solution.c, axis=0)
            feasible = [(aj, ctx.ptx_min_matrix[si, aj])
                       for aj in aids
                       if ctx.link_feasible_matrix[si, aj] == 1 and loads[aj] < Cmax]
            if feasible:
                feasible.sort(key=lambda t: t[1])
                solution.c[si, feasible[0][0]] = 1

    # 功率上限修复
    ptx_max = float(ctx.config.get("channel", {}).get("p_tx_max", 0.5))
    for i in range(ctx.num_candidates):
        for j in range(ctx.num_candidates):
            if solution.c[i, j] == 1 and solution.p_tx[i, j] > ptx_max:
                solution.p_tx[i, j] = ptx_max

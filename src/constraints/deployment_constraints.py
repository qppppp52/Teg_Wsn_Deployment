"""部署约束：数量上限 + 传感器/AP 互斥 x_r + y_r <= 1"""
import numpy as np


def check_deployment_constraints(solution, ctx):
    cv = 0.0
    scfg = ctx.config.get("deployment", {})
    mx_s = scfg.get("max_sensors", 20)
    mx_a = scfg.get("max_aps", 4)
    ns = int(np.sum(solution.x))
    na = int(np.sum(solution.y))
    if ns > mx_s:
        cv += ns - mx_s
    if na > mx_a:
        cv += na - mx_a
    cv += float(np.sum(solution.x & solution.y))
    return cv


def _sensor_coverage_contribution(gid, ctx):
    """候选点gid作为传感器的覆盖贡献：能覆盖多少目标点"""
    return int(np.sum(ctx.coverage_matrix[gid]))


def _ap_communication_contribution(gid, ctx):
    """候选点gid作为AP的通信贡献：可达传感器数量 × 平均潜在速率"""
    n_reachable = int(np.sum(ctx.link_feasible_matrix[:, gid]))
    if n_reachable == 0:
        return 0.0
    avg_rate = np.mean(ctx.potential_rate_matrix[:, gid][ctx.link_feasible_matrix[:, gid] == 1])
    return n_reachable * avg_rate


def repair_deployment(solution, ctx):
    """修复部署约束"""
    scfg = ctx.config.get("deployment", {})
    mx_s = scfg.get("max_sensors", 20)
    mx_a = scfg.get("max_aps", 4)

    # 互斥修复：比较传感器覆盖贡献 vs AP通信贡献，保留贡献大者
    overlap = np.where(solution.x & solution.y)[0]
    for gid in overlap:
        cov_contrib = _sensor_coverage_contribution(gid, ctx)
        ap_contrib = _ap_communication_contribution(gid, ctx)
        if cov_contrib >= ap_contrib:
            solution.y[gid] = 0  # 保留传感器
        else:
            solution.x[gid] = 0  # 保留AP

    # 传感器数量超限：按覆盖贡献排序，保留高贡献的
    sids = np.where(solution.x == 1)[0]
    if len(sids) > mx_s:
        cc = np.array([_sensor_coverage_contribution(s, ctx) for s in sids])
        order = np.argsort(cc)
        remove = sids[order[:len(sids) - mx_s]]
        for gid in remove:
            solution.x[gid] = 0

    # AP数量超限：按通信贡献排序，保留高贡献的
    aids = np.where(solution.y == 1)[0]
    if len(aids) > mx_a:
        ac = np.array([_ap_communication_contribution(a, ctx) for a in aids])
        order = np.argsort(ac)
        remove = aids[order[:len(aids) - mx_a]]
        for gid in remove:
            solution.y[gid] = 0

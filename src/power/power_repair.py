"""功率修复：能量不足时降低p_tx，能量富余时提升p_tx以最大化吞吐量"""
import numpy as np
from src.power.power_bounds import calculate_actual_ptx_upper_bound
from src.power.adaptive_power import adaptive_power


def repair_power(solution, ctx):
    """
    修复发射功率：对每个传感器，根据当前实际散热片分配情况重新计算p_tx。
    - 能量不足：降低p_tx到能量可支撑的水平
    - 能量富余：在能量允许范围内提高p_tx以增大吞吐量
    - 更新 sensor_power_consumption 以反映新功率
    """
    K = ctx.num_candidates
    strategy = getattr(ctx, "current_power_policy", None)
    if strategy is None:
        strategy = solution.metadata.get("power_policy")
    if strategy is None:
        strategy = ctx.config.get("mode", {}).get("power_strategy", "energy_balanced")
    if strategy == "balanced":
        strategy = "energy_balanced"
    if strategy == "sink_limited":
        strategy = "conservative"
    scfg = ctx.config.get("sensor", {})
    P_fixed = scfg.get("P_sens", 0.01) + scfg.get("P_proc", 0.005)
    acfg = ctx.config.get("ap", {})
    P_rx = acfg.get("P_rx", 0.003)

    for si in range(K):
        if solution.x[si] == 0:
            continue
        aps = np.where(solution.c[si] == 1)[0]
        if len(aps) == 0:
            continue
        aj = aps[0]

        ptx_min = ctx.ptx_min_matrix[si, aj]

        # 根据实际已分配的散热片计算实际可用能量
        n_actual = solution.n_sink_sensor[si]
        actual_harvest = n_actual * ctx.P_grid[si]

        # 能量允许的最大发射功率（基于实际散热片）
        ptx_up_actual = actual_harvest - P_fixed
        ptx_up_actual = max(0.0, ptx_up_actual)

        if ptx_up_actual < ptx_min:
            # 不可行：即使降到最小功率也撑不住
            solution.p_tx[si, aj] = ptx_min
        elif strategy == "conservative":
            solution.p_tx[si, aj] = ptx_min
        else:
            # energy_balanced 或 rsum_capacity_priority：在能量允许范围内提升功率
            if strategy == "rsum_capacity_priority":
                eta = 0.8
            else:
                eta = 0.5
            ptx_target = ptx_min + eta * (ptx_up_actual - ptx_min)
            # 但不能超过p_tx_max
            ptx_max = float(ctx.config.get("channel", {}).get("p_tx_max", 0.5))
            solution.p_tx[si, aj] = min(ptx_target, ptx_max)

        # 更新传感器功耗
        solution.sensor_power_consumption[si] = P_fixed + solution.p_tx[si, aj]

    # 更新AP功耗
    for ai in range(K):
        if solution.y[ai] == 1:
            n_conn = int(np.sum(solution.c[:, ai]))
            P_idle = acfg.get("P_idle", 0.05)
            P_proc = acfg.get("P_proc", 0.02)
            solution.ap_power_consumption[ai] = P_idle + P_proc + P_rx * n_conn

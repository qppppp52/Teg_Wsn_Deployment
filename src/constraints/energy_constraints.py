"""能量中性约束 — P_harvest = n_sink * P_grid >= P_cons"""
import numpy as np


def check_energy_constraints(solution, ctx):
    cv = 0.0
    for si in np.where(solution.x == 1)[0]:
        Ph = solution.n_sink_sensor[si] * ctx.P_grid[si]
        solution.sensor_harvest_power[si] = Ph
        if Ph < solution.sensor_power_consumption[si]:
            deficit = solution.sensor_power_consumption[si] - Ph
            cv += deficit / max(ctx.P_grid[si], 1e-12)
    for ai in np.where(solution.y == 1)[0]:
        Ph = solution.n_sink_ap[ai] * ctx.P_grid[ai]
        solution.ap_harvest_power[ai] = Ph
        if Ph < solution.ap_power_consumption[ai]:
            deficit = solution.ap_power_consumption[ai] - Ph
            cv += deficit / max(ctx.P_grid[ai], 1e-12)
    return cv


def repair_energy_constraints(solution, ctx):
    """
    修复能量约束。闭环逻辑：
    1. 先重新分配发射功率（基于实际散热片状态）
    2. 重新计算散热片需求（基于新功耗）
    3. 尝试增加散热片
    4. 仍不够则删除节点
    """
    from src.power.power_repair import repair_power
    from src.heatsink.sink_requirement import compute_sink_requirements
    from src.heatsink.sink_allocator import allocate_all_sinks

    # 第1遍：重算功率（基于当前散热片）
    repair_power(solution, ctx)

    # 重算散热片需求（基于当前功耗）
    compute_sink_requirements(solution, ctx)

    # 重新分配散热片
    allocate_all_sinks(solution, ctx)

    # 第2遍：散热片分配后可能改变了n_sink，再微调功率
    repair_power(solution, ctx)

    # 尝试给能量不足的传感器增加散热片
    for si in np.where(solution.x == 1)[0]:
        Ph = solution.n_sink_sensor[si] * ctx.P_grid[si]
        while (Ph < solution.sensor_power_consumption[si] and
               solution.n_sink_sensor[si] < ctx.nmax[si]):
            solution.n_sink_sensor[si] += 1
            Ph = solution.n_sink_sensor[si] * ctx.P_grid[si]
        if Ph < solution.sensor_power_consumption[si]:
            # 最后一招：降功率到最小，再试
            aps = np.where(solution.c[si] == 1)[0]
            if len(aps) > 0:
                solution.p_tx[si, aps[0]] = ctx.ptx_min_matrix[si, aps[0]]
                scfg = ctx.config.get("sensor", {})
                solution.sensor_power_consumption[si] = (
                    scfg.get("P_sens", 0.01) + scfg.get("P_proc", 0.005) +
                    solution.p_tx[si, aps[0]]
                )
            compute_sink_requirements(solution, ctx)
            allocate_all_sinks(solution, ctx)
            Ph = solution.n_sink_sensor[si] * ctx.P_grid[si]
            if Ph < solution.sensor_power_consumption[si]:
                solution.x[si] = 0
                solution.c[si] = 0

    # 同样对AP
    for ai in np.where(solution.y == 1)[0]:
        Ph = solution.n_sink_ap[ai] * ctx.P_grid[ai]
        while (Ph < solution.ap_power_consumption[ai] and
               solution.n_sink_ap[ai] < ctx.nmax[ai]):
            solution.n_sink_ap[ai] += 1
            Ph = solution.n_sink_ap[ai] * ctx.P_grid[ai]

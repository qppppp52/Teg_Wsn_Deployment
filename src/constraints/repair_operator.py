"""固定顺序约束修复 — 部署→通信→功率+能耗→散热片→AP服务"""
from src.constraints.deployment_constraints import repair_deployment
from src.constraints.link_constraints import repair_link_constraints
from src.constraints.capacity_constraints import repair_capacity
from src.constraints.energy_constraints import repair_energy_constraints
from src.constraints.ap_service_constraints import repair_empty_aps
from src.heatsink.sink_allocator import allocate_all_sinks


def repair_solution(solution, ctx):
    """建模文件修复顺序：部署修复 → 通信修复 → 容量修复 → 能量修复
    （含功率重新分配+散热片重新分配）→ AP非空修复"""
    repair_deployment(solution, ctx)
    repair_link_constraints(solution, ctx)
    repair_capacity(solution, ctx)
    # 能量修复内部已包含功率重新分配和散热片重新分配
    repair_energy_constraints(solution, ctx)
    repair_empty_aps(solution, ctx)
    # 最终确保散热片与当前部署一致
    allocate_all_sinks(solution, ctx)
    return solution

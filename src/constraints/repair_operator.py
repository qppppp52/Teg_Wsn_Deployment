"""Constraint-repair operators.

The original fixed-order repair is kept for CR-MODE baseline comparability. The
strategy-aware wrapper is used by DQN-CR-MODE to change repair order and power
policy without replacing the underlying repair functions.
"""
from src.constraints.deployment_constraints import repair_deployment
from src.constraints.link_constraints import repair_link_constraints
from src.constraints.capacity_constraints import repair_capacity
from src.constraints.energy_constraints import repair_energy_constraints
from src.constraints.ap_service_constraints import repair_empty_aps
from src.heatsink.sink_allocator import allocate_all_sinks


DEFAULT_REPAIR_ORDER = ["deploy", "link", "capacity", "energy", "service", "sink"]


def repair_solution(solution, ctx):
    """Baseline fixed-order repair used by plain CR-MODE."""
    repair_deployment(solution, ctx)
    repair_link_constraints(solution, ctx)
    repair_capacity(solution, ctx)
    repair_energy_constraints(solution, ctx)
    repair_empty_aps(solution, ctx)
    allocate_all_sinks(solution, ctx)
    return solution


def repair_solution_with_strategy(
    solution,
    ctx,
    repair_order=None,
    power_policy=None,
    strategy_name=None,
):
    """Repair a solution with configurable repair order and power policy."""
    order = list(repair_order or DEFAULT_REPAIR_ORDER)
    previous_policy = getattr(ctx, "current_power_policy", None)
    if power_policy is not None:
        ctx.current_power_policy = power_policy
        solution.metadata["power_policy"] = power_policy
    if strategy_name is not None:
        solution.metadata["repair_strategy"] = strategy_name

    try:
        for step in order:
            step_key = _normalize_step(step)
            if step_key == "deploy":
                repair_deployment(solution, ctx)
            elif step_key == "link":
                repair_link_constraints(solution, ctx)
            elif step_key == "capacity":
                repair_capacity(solution, ctx)
            elif step_key == "energy":
                repair_energy_constraints(solution, ctx)
            elif step_key == "service":
                repair_empty_aps(solution, ctx)
            elif step_key == "sink":
                allocate_all_sinks(solution, ctx)
            else:
                raise ValueError(f"Unknown repair step: {step}")
    finally:
        if previous_policy is None and hasattr(ctx, "current_power_policy"):
            delattr(ctx, "current_power_policy")
        elif previous_policy is not None:
            ctx.current_power_policy = previous_policy
    return solution


def _normalize_step(step):
    mapping = {
        "deployment": "deploy",
        "deploy": "deploy",
        "communication": "link",
        "link": "link",
        "power": "energy",
        "energy": "energy",
        "heatsink": "sink",
        "sink": "sink",
        "ap_service": "service",
        "service": "service",
        "capacity": "capacity",
    }
    return mapping.get(str(step).lower(), str(step).lower())

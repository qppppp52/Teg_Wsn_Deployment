"""Constraint-repair operators for the six-component CV contract."""
from src.constraints.deployment_constraints import repair_deployment
from src.constraints.link_constraints import repair_link_constraints
from src.constraints.energy_constraints import repair_energy_constraints
from src.constraints.ap_service_constraints import repair_empty_aps
from src.heatsink.sink_allocator import allocate_all_sinks


DEFAULT_REPAIR_ORDER = ["deploy", "link", "energy", "service", "sink"]


def repair_solution(solution, ctx):
    """Baseline fixed-order repair used by plain CR-MODE."""
    repair_deployment(solution, ctx)
    repair_link_constraints(solution, ctx)
    repair_energy_constraints(solution, ctx)
    repair_empty_aps(solution, ctx)
    allocate_all_sinks(solution, ctx)
    return solution


def repair_solution_with_strategy(
    solution,
    ctx,
    repair_order=None,
    strategy_name=None,
):
    """Apply an action-specific order without capacity or power-policy branches."""
    order = list(repair_order or DEFAULT_REPAIR_ORDER)
    if strategy_name is not None:
        solution.metadata["repair_strategy"] = strategy_name
    for step in order:
        step_key = _normalize_step(step)
        if step_key == "deploy":
            repair_deployment(solution, ctx)
        elif step_key == "link":
            repair_link_constraints(solution, ctx)
        elif step_key == "energy":
            repair_energy_constraints(solution, ctx)
        elif step_key == "service":
            repair_empty_aps(solution, ctx)
        elif step_key == "sink":
            allocate_all_sinks(solution, ctx)
        else:
            raise ValueError(f"Unknown repair step: {step}")
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
    }
    return mapping.get(str(step).lower(), str(step).lower())
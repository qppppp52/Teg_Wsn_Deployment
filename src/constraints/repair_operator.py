"""Constraint-repair operators for the six-component CV contract."""
from src.constraints.ap_service_constraints import repair_empty_aps
from src.constraints.deployment_constraints import repair_deployment
from src.constraints.energy_constraints import repair_energy_constraints
from src.constraints.heatsink_constraints import repair_sink_hard_constraints
from src.constraints.link_constraints import repair_link_constraints
from src.heatsink.sink_allocator import allocate_all_sinks
from src.power.power_repair import repair_power_legality


# Kept for existing action-specific DQN strategies. The fixed baseline below is
# intentionally separate so DQN remains unchanged until its repair skeleton is reviewed.
DEFAULT_REPAIR_ORDER = ("deploy", "link", "energy", "service", "sink")
BASELINE_REPAIR_ORDER = (
    "deploy",
    "link",
    "sink_hard",
    "power",
    "energy",
    "service",
)


def repair_solution(solution, ctx):
    """Run the deterministic baseline CR-MODE repair pipeline."""
    repair_deployment(solution, ctx)
    repair_link_constraints(solution, ctx)
    repair_sink_hard_constraints(solution, ctx)
    repair_power_legality(solution, ctx)
    repair_energy_constraints(solution, ctx)
    repair_empty_aps(solution, ctx)
    solution.metadata["baseline_repair_order"] = list(BASELINE_REPAIR_ORDER)
    return solution


def repair_solution_with_strategy(
    solution,
    ctx,
    repair_order=None,
    strategy_name=None,
):
    """Apply an existing action-specific order without changing DQN semantics."""
    order = list(repair_order or DEFAULT_REPAIR_ORDER)
    if strategy_name is not None:
        solution.metadata["repair_strategy"] = strategy_name
    for step in order:
        step_key = _normalize_step(step)
        if step_key == "deploy":
            repair_deployment(solution, ctx)
        elif step_key == "link":
            repair_link_constraints(solution, ctx)
        elif step_key == "sink_hard":
            repair_sink_hard_constraints(solution, ctx)
        elif step_key == "power":
            repair_power_legality(solution, ctx)
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
        "power": "power",
        "energy": "energy",
        "sink_hard": "sink_hard",
        "heatsink_hard": "sink_hard",
        "heatsink": "sink",
        "sink": "sink",
        "ap_service": "service",
        "service": "service",
    }
    return mapping.get(str(step).lower(), str(step).lower())

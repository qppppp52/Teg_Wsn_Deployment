"""Unified constraint evaluation."""
from src.constraints.ap_service_constraints import check_ap_service
from src.constraints.capacity_constraints import check_capacity_constraints
from src.constraints.deployment_constraints import check_deployment_constraints
from src.constraints.energy_constraints import check_energy_constraints
from src.constraints.heatsink_constraints import heatsink_constraint_components
from src.constraints.link_constraints import check_link_constraints


def evaluate_all_constraints(solution, ctx):
    config = ctx.config.get("constraints", {})
    weights = config.get("cv_weights", {})
    cv_deploy = check_deployment_constraints(solution, ctx)
    cv_link = check_link_constraints(solution, ctx)
    cv_capacity = check_capacity_constraints(solution, ctx)
    cv_energy = check_energy_constraints(solution, ctx)
    sink_components = heatsink_constraint_components(solution, ctx)
    cv_sink = sink_components["total"]
    cv_service = check_ap_service(solution, ctx)

    total = (
        weights.get("deploy", 1.0) * cv_deploy
        + weights.get("link", 1.0) * cv_link
        + weights.get("capacity", 1.0) * cv_capacity
        + weights.get("energy", 1.0) * cv_energy
        + weights.get("sink", 1.0) * cv_sink
        + weights.get("service", 1.0) * cv_service
    )
    solution.cv_deploy = cv_deploy
    solution.cv_energy = cv_energy
    solution.cv_link = cv_link
    solution.cv_capacity = cv_capacity
    solution.cv_sink = cv_sink
    solution.cv_sink_conflict = sink_components["sink_conflict_cv"]
    solution.cv_sink_shortage = sink_components["sink_shortage_cv"]
    solution.cv_sink_invalid = (
        sink_components["invalid_cv"] + sink_components["duplicate_cv"]
    )
    solution.cv_service = cv_service
    solution.cv = float(total)
    feasible_tol = float(config.get("feasible_tol", 1.0e-8))
    solution.feasible = bool(total <= feasible_tol)
    return solution.cv

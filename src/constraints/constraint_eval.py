import numpy as np
from src.constraints.deployment_constraints import check_deployment_constraints
from src.constraints.link_constraints import check_link_constraints
from src.constraints.capacity_constraints import check_capacity_constraints
from src.constraints.ap_service_constraints import check_ap_service
from src.constraints.energy_constraints import check_energy_constraints
from src.constraints.heatsink_constraints import check_heatsink_constraints

def evaluate_all_constraints(solution, ctx):
    cfg = ctx.config.get("constraints", {})
    weights = cfg.get("cv_weights", {})
    cv_d = check_deployment_constraints(solution, ctx)
    cv_l = check_link_constraints(solution, ctx)
    cv_c = check_capacity_constraints(solution, ctx)
    cv_e = check_energy_constraints(solution, ctx)
    cv_s = check_heatsink_constraints(solution, ctx)
    cv_sv = check_ap_service(solution, ctx)
    wd = weights.get("deploy", 1.0)
    wl = weights.get("link", 1.0)
    wc = weights.get("capacity", 1.0)
    we = weights.get("energy", 1.0)
    ws = weights.get("sink", 1.0)
    wsv = weights.get("service", 1.0)
    total = wd*cv_d + wl*cv_l + wc*cv_c + we*cv_e + ws*cv_s + wsv*cv_sv
    solution.cv_energy = cv_e
    solution.cv_link = cv_l
    solution.cv_capacity = cv_c
    solution.cv_sink = cv_s
    solution.cv_service = cv_sv
    solution.cv = total
    solution.feasible = (total < 0.05)
    return total

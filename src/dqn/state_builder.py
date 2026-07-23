"""State vector builder for DQN-CR-MODE."""
from __future__ import annotations

import numpy as np

from src.constraints.constraint_report import evaluate_constraints
from src.constraints.repair_config import resolve_repair_config
from src.constraints.cv_pressure import (
    PopulationPressureSummary,
    aggregate_population_pressure,
    load_state_cv_total_ref,
)
from src.constraints.cv_schema import CV_COMPONENT_KEYS
from src.evaluation.diversity import objective_space_diversity


STATE_KEYS = (
    "CV_mean_norm", "CV_min_norm", "FR", "HV_norm", "delta_HV_norm",
    "BestCoverage", "MeanCoverageFeasible", "BestRsum_norm",
    "MeanRsumFeasible_norm", "Diversity",
    "P_deploy_mean", "P_link_mean", "P_power_mean", "P_service_mean",
    "P_sink_mean", "P_energy_mean",
    "B_deploy_rate", "B_link_rate", "B_power_rate", "B_service_rate",
    "B_sink_rate", "B_energy_rate",
    "repair_success_rate", "mean_repair_iter_norm", "generation_progress",
)


def build_state(population, archive, ctx, prev_metrics=None, gen=0, max_generations=1):
    """Build a fixed report-backed state using one shared pressure summary."""
    del archive
    prev_metrics = prev_metrics or {}
    sols = list(population.solutions or [])
    if not sols:
        return np.zeros(len(STATE_KEYS), dtype=np.float32)
    reports = []
    for solution in sols:
        report = getattr(solution, "constraint_report", None)
        if report is None:
            report = evaluate_constraints(solution, ctx)
        reports.append(report)
    feasible = [solution for solution, report in zip(sols, reports) if report.feasible]
    summary = prev_metrics.get("population_pressure")
    if not isinstance(summary, PopulationPressureSummary):
        summary = aggregate_population_pressure(sols, ctx)

    cfg_eval = ctx.config.get("evaluation", {})
    rsum_ref_max = float(cfg_eval.get("rsum_ref_max", 1.0e9))
    if not np.isfinite(rsum_ref_max) or rsum_ref_max <= 0.0:
        raise ValueError("evaluation.rsum_ref_max must be finite and positive")
    max_repair = float(resolve_repair_config(ctx.config).max_outer_repair_rounds)
    cv_total_ref = load_state_cv_total_ref(ctx.config)
    current_hv = float(prev_metrics.get("current_HV", prev_metrics.get("HV", 0.0)))
    previous_hv = float(prev_metrics.get("HV", 0.0))
    best_cov = max((float(s.coverage) for s in feasible), default=0.0)
    mean_cov = float(np.mean([s.coverage for s in feasible])) if feasible else 0.0
    rsum_values = [float(s.rsum_capacity) for s in feasible]
    best_rsum = max(rsum_values, default=0.0) / rsum_ref_max
    mean_rsum = float(np.mean(rsum_values)) / rsum_ref_max if rsum_values else 0.0
    repair_iters = np.asarray([getattr(s, "repair_iter", 0) for s in sols], dtype=float)
    repair_success = np.asarray([getattr(s, "repair_success", False) for s in sols], dtype=float)
    cv_values = np.asarray([float(report.cv_total) for report in reports], dtype=float)
    values = [
        float(np.clip(np.mean(cv_values) / cv_total_ref, 0.0, 1.0)),
        float(np.clip(np.min(cv_values) / cv_total_ref, 0.0, 1.0)),
        len(feasible) / len(sols),
        float(np.clip(current_hv, 0.0, 1.0)),
        float(np.clip(current_hv - previous_hv, -1.0, 1.0)),
        float(np.clip(best_cov, 0.0, 1.0)),
        float(np.clip(mean_cov, 0.0, 1.0)),
        float(np.clip(best_rsum, 0.0, 1.0)),
        float(np.clip(mean_rsum, 0.0, 1.0)),
        float(np.clip(objective_space_diversity(sols), 0.0, 1.0)),
        *[float(summary.mean_pressure[key]) for key in CV_COMPONENT_KEYS],
        *[float(summary.violation_rate[key]) for key in CV_COMPONENT_KEYS],
        float(np.clip(np.mean(repair_success), 0.0, 1.0)),
        float(np.clip(np.mean(repair_iters) / max(max_repair, 1.0), 0.0, 1.0)),
        float(np.clip(float(gen) / max(float(max_generations), 1.0), 0.0, 1.0)),
    ]
    state = np.asarray(values, dtype=np.float32)
    if state.shape != (len(STATE_KEYS),) or not np.all(np.isfinite(state)):
        raise ValueError("DQN state must have the configured finite fixed dimension")
    return state

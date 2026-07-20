"""State vector builder for DQN-CR-MODE."""
from __future__ import annotations

import numpy as np

from src.constraints.constraint_report import evaluate_constraints
from src.evaluation.diversity import objective_space_diversity

STATE_KEYS = [
    "CV_mean_norm", "CV_min_norm", "FR", "HV_norm", "delta_HV_norm",
    "BestCoverage", "MeanCoverageFeasible", "BestRsum_norm",
    "MeanRsumFeasible_norm", "Diversity", "P_deploy_mean", "P_link_mean",
    "P_power_mean", "P_energy_mean", "P_sensor_energy_mean",
    "P_ap_energy_mean", "P_sink_mean", "P_service_mean",
    "repair_success_rate", "mean_repair_iter_norm", "generation_progress",
]


def bounded(value):
    value = float(value)
    if not np.isfinite(value):
        raise ValueError("state input must be finite")
    return 0.0 if value <= 0.0 else value / (1.0 + value)


def build_state(population, archive, ctx, prev_metrics=None, gen=0, max_generations=1):
    """Build a fixed, report-backed and deterministically scaled state vector."""
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
    cfg_eval = ctx.config.get("evaluation", {})
    rsum_ref_max = float(cfg_eval.get("rsum_ref_max", 1.0e9))
    if not np.isfinite(rsum_ref_max) or rsum_ref_max <= 0.0:
        raise ValueError("evaluation.rsum_ref_max must be finite and positive")
    max_repair = float(ctx.config.get("constraints", {}).get("max_repair_iter", 5))
    current_hv = float(prev_metrics.get("current_HV", prev_metrics.get("HV", 0.0)))
    previous_hv = float(prev_metrics.get("HV", 0.0))
    pressure_values = {
        "deploy": [bounded(report.components.deploy) for report in reports],
        "link": [bounded(report.components.link) for report in reports],
        "power": [bounded(report.components.power) for report in reports],
        "energy": [bounded(report.components.energy) for report in reports],
        "sensor_energy": [bounded(report.energy_cv.sensor) for report in reports],
        "ap_energy": [bounded(report.energy_cv.ap) for report in reports],
        "sink": [bounded(report.components.sink) for report in reports],
        "service": [bounded(report.components.service) for report in reports],
    }
    best_cov = max((float(s.coverage) for s in feasible), default=0.0)
    mean_cov = float(np.mean([s.coverage for s in feasible])) if feasible else 0.0
    rsum_values = [float(s.rsum_capacity) for s in feasible]
    best_rsum = max(rsum_values, default=0.0) / rsum_ref_max
    mean_rsum = float(np.mean(rsum_values)) / rsum_ref_max if rsum_values else 0.0
    repair_iters = np.asarray([getattr(s, "repair_iter", 0) for s in sols], dtype=float)
    repair_success = np.asarray([getattr(s, "repair_success", False) for s in sols], dtype=float)
    cv_values = [bounded(report.cv_total) for report in reports]
    values = [
        float(np.mean(cv_values)),
        float(np.min(cv_values)),
        len(feasible) / len(sols),
        float(np.clip(current_hv, 0.0, 1.0)),
        float(np.clip(current_hv - previous_hv, -1.0, 1.0)),
        float(np.clip(best_cov, 0.0, 1.0)),
        float(np.clip(mean_cov, 0.0, 1.0)),
        float(np.clip(best_rsum, 0.0, 1.0)),
        float(np.clip(mean_rsum, 0.0, 1.0)),
        float(np.clip(objective_space_diversity(sols), 0.0, 1.0)),
        float(np.mean(pressure_values["deploy"])),
        float(np.mean(pressure_values["link"])),
        float(np.mean(pressure_values["power"])),
        float(np.mean(pressure_values["energy"])),
        float(np.mean(pressure_values["sensor_energy"])),
        float(np.mean(pressure_values["ap_energy"])),
        float(np.mean(pressure_values["sink"])),
        float(np.mean(pressure_values["service"])),
        float(np.mean(repair_success)),
        float(np.clip(np.mean(repair_iters) / max(max_repair, 1.0), 0.0, 1.0)),
        float(np.clip(float(gen) / max(float(max_generations), 1.0), 0.0, 1.0)),
    ]
    return np.asarray(values, dtype=np.float32)
"""State vector builder for DQN-CR-MODE."""
from __future__ import annotations

import numpy as np
from src.constraints.cv_pressure import normalize_cv_components
from src.evaluation.diversity import objective_space_diversity

STATE_KEYS = [
    "CV_mean_norm", "CV_min_norm", "FR", "HV_norm", "delta_HV_norm",
    "BestCoverage", "MeanCoverageFeasible", "BestRsum_norm",
    "MeanRsumFeasible_norm", "Diversity", "P_deploy_mean", "P_link_mean",
    "P_capacity_mean", "P_energy_mean", "P_sink_mean", "P_service_mean",
    "repair_success_rate", "mean_repair_iter_norm", "generation_progress",
]


def build_state(population, archive, ctx, prev_metrics=None, gen=0, max_generations=1):
    """Build a fixed-length normalized state vector from current evaluated data."""
    prev_metrics = prev_metrics or {}
    sols = population.solutions or []
    if not sols:
        return np.zeros(len(STATE_KEYS), dtype=np.float32)
    feasible = [s for s in sols if s.feasible]
    cvs = np.array([s.cv for s in sols], dtype=float)
    cfg_eval = ctx.config.get("evaluation", {})
    rsum_ref_max = float(cfg_eval.get("rsum_ref_max", 1.0e9))
    max_repair = float(ctx.config.get("constraints", {}).get("max_repair_iter", 5))
    hv_hist = prev_metrics.get("HV", 0.0)
    current_hv = float(prev_metrics.get("current_HV", hv_hist))

    pressures = [normalize_cv_components(s, ctx) for s in sols]
    pressure_mean = {k: float(np.mean([p[k] for p in pressures])) for k in pressures[0]}
    best_cov = max([s.coverage for s in feasible], default=0.0)
    mean_cov = float(np.mean([s.coverage for s in feasible])) if feasible else 0.0
    best_rsum = max([float(s.metadata.get("rsum_capacity", s.rsum_capacity)) for s in feasible], default=0.0) / (rsum_ref_max + 1e-12)
    mean_rsum = (float(np.mean([s.metadata.get("rsum_capacity", s.rsum_capacity) for s in feasible])) / (rsum_ref_max + 1e-12)) if feasible else 0.0
    repair_iters = np.array([getattr(s, "repair_iter", 0) for s in sols], dtype=float)
    repair_success = np.array([getattr(s, "repair_success", False) for s in sols], dtype=float)

    values = [
        min(float(np.mean(cvs)) / 20.0, 1.0),
        min(float(np.min(cvs)) / 20.0, 1.0),
        len(feasible) / len(sols),
        current_hv,
        np.clip(current_hv - hv_hist, -1.0, 1.0),
        best_cov,
        mean_cov,
        min(best_rsum, 1.0),
        min(mean_rsum, 1.0),
        objective_space_diversity(sols),
        pressure_mean["deploy"], pressure_mean["link"], pressure_mean["capacity"],
        pressure_mean["energy"], pressure_mean["sink"], pressure_mean["service"],
        float(np.mean(repair_success)),
        min(float(np.mean(repair_iters)) / max(max_repair, 1.0), 1.0),
        min(float(gen) / max(float(max_generations), 1.0), 1.0),
    ]
    return np.asarray(values, dtype=np.float32)

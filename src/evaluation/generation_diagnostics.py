"""Generation-level diagnostics for repair pressure and objective trade-offs."""
from __future__ import annotations

import numpy as np
from src.evaluation.diversity import objective_space_diversity
from src.evaluation.hypervolume import hypervolume_2d_max


def make_convergence_history() -> dict:
    """Return a convergence-history dict with legacy and diagnostic keys."""
    keys = [
        "feasible_count", "FR_current", "Coverage_feasible", "Rsum_feasible_mbps",
        "Coverage_all", "Rsum_all_mbps", "CV_mean", "CV_min",
        "archive_best_coverage", "archive_best_rsum_mbps", "HV",
        "FR_before_repair", "CV_before_repair_mean", "FR_after_repair",
        "CV_after_repair_mean", "cv_deploy_mean", "cv_link_mean",
        "cv_capacity_mean", "cv_energy_mean", "cv_sink_mean", "cv_service_mean",
        "mean_repair_iter", "repair_success_rate", "coverage_best",
        "coverage_mean_feasible", "rsum_capacity_best",
        "min_link_capacity_bps", "max_link_capacity_bps", "archive_size",
        "pareto_count", "diversity",
    ]
    return {key: [] for key in keys}


def record_generation(history: dict, population, archive, config: dict) -> dict:
    """Append current-generation diagnostics and return the appended metrics."""
    sols = list(population.solutions or [])
    feasible = [s for s in sols if s.feasible]
    n = max(len(sols), 1)
    fe_objs = archive.get_feasible_objectives()
    eval_cfg = config.get("evaluation", {})
    rsum_ref_max = float(eval_cfg.get("rsum_ref_max", max(float(fe_objs[:, 1].max()), 1.0) if len(fe_objs) else 1.0))
    hv = hypervolume_2d_max(
        fe_objs,
        ref=tuple(eval_cfg.get("hv_reference", [0.0, 0.0])),
        bounds=((0.0, 1.0), (float(eval_cfg.get("rsum_ref_min", 0.0)), rsum_ref_max)),
    ) if len(fe_objs) else 0.0

    metric = {
        "feasible_count": len(feasible),
        "FR_current": len(feasible) / n,
        "Coverage_feasible": _mean([s.coverage for s in feasible], nan=True),
        "Rsum_feasible_mbps": _mean([s.rsum_capacity for s in feasible], nan=True) / 1e6 if feasible else float("nan"),
        "Coverage_all": _mean([s.coverage for s in sols]),
        "Rsum_all_mbps": _mean([s.rsum_capacity for s in sols]) / 1e6,
        "CV_mean": _mean([s.cv for s in sols], default=float("inf")),
        "CV_min": min([s.cv for s in sols], default=float("inf")),
        "archive_best_coverage": float(np.max(fe_objs[:, 0])) if len(fe_objs) else float("nan"),
        "archive_best_rsum_mbps": float(np.max(fe_objs[:, 1])) / 1e6 if len(fe_objs) else float("nan"),
        "HV": hv,
        "FR_before_repair": _mean([getattr(s, "feasible_before_repair", False) for s in sols]),
        "CV_before_repair_mean": _mean([getattr(s, "cv_before_repair", s.cv) for s in sols], default=float("inf")),
        "FR_after_repair": len(feasible) / n,
        "CV_after_repair_mean": _mean([getattr(s, "cv_after_repair", s.cv) for s in sols], default=float("inf")),
        "cv_deploy_mean": _mean([getattr(s, "cv_deploy", 0.0) for s in sols]),
        "cv_link_mean": _mean([getattr(s, "cv_link", 0.0) for s in sols]),
        "cv_capacity_mean": _mean([getattr(s, "cv_capacity", 0.0) for s in sols]),
        "cv_energy_mean": _mean([getattr(s, "cv_energy", 0.0) for s in sols]),
        "cv_sink_mean": _mean([getattr(s, "cv_sink", 0.0) for s in sols]),
        "cv_service_mean": _mean([getattr(s, "cv_service", 0.0) for s in sols]),
        "mean_repair_iter": _mean([getattr(s, "repair_iter", 0) for s in sols]),
        "repair_success_rate": _mean([getattr(s, "repair_success", False) for s in sols]),
        "coverage_best": max([s.coverage for s in feasible], default=0.0),
        "coverage_mean_feasible": _mean([s.coverage for s in feasible], nan=True),
        "rsum_capacity_best": max([float(s.metadata.get("rsum_capacity", s.rsum_capacity)) for s in feasible], default=0.0),
        "mean_link_capacity_bps": _mean([s.metadata.get("mean_link_capacity_bps", 0.0) for s in sols]),
        "min_link_capacity_bps": _min_positive([s.metadata.get("min_link_capacity_bps", 0.0) for s in sols]),
        "max_link_capacity_bps": max([s.metadata.get("max_link_capacity_bps", 0.0) for s in sols], default=0.0),
        "archive_size": len(archive),
        "pareto_count": len(feasible),
        "diversity": objective_space_diversity(feasible),
    }
    for key, value in metric.items():
        history.setdefault(key, []).append(value)
    return metric


def _mean(values, default=0.0, nan=False):
    vals = list(values)
    if not vals:
        return float("nan") if nan else default
    return float(np.mean(vals))


def _min_positive(values):
    vals = [float(v) for v in values if float(v) > 0.0]
    return min(vals) if vals else 0.0

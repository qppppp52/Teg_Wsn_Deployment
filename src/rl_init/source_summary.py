"""Source-wise quality summaries for mixed DRL initialization populations."""
from __future__ import annotations

import math
import numpy as np


SOURCE_TYPES = ("drl", "heuristic", "random")


def build_source_wise_summary(rows: list[dict]) -> tuple[dict, dict]:
    source_wise = {source: _summarize_source(rows, source) for source in SOURCE_TYPES}
    differences = {
        "drl_minus_heuristic_mean_coverage": _difference(source_wise, "drl", "heuristic", "mean_coverage"),
        "drl_minus_random_mean_coverage": _difference(source_wise, "drl", "random", "mean_coverage"),
        "drl_minus_heuristic_mean_rsum_capacity": _difference(source_wise, "drl", "heuristic", "mean_rsum_capacity"),
        "drl_minus_random_mean_rsum_capacity": _difference(source_wise, "drl", "random", "mean_rsum_capacity"),
    }
    return source_wise, differences


def build_ppo_training_summary(rows: list[dict], window_size: int = 100) -> dict:
    episode_rows = [row for row in rows if int(row.get("episode", -1)) >= 0]
    updated = [row for row in episode_rows if _as_bool(row.get("updated_this_episode", False))]
    window = max(int(window_size), 1)
    first = episode_rows[:window]
    last = episode_rows[-window:] if episode_rows else []
    final_update = updated[-1] if updated else {}
    return {
        "episodes": len(episode_rows),
        "update_count": max([int(row.get("ppo_update_count", 0)) for row in episode_rows], default=0),
        "first_update_episode": int(updated[0]["episode"]) if updated else None,
        "final_policy_loss": _finite_or_none(final_update.get("policy_loss")),
        "final_value_loss": _finite_or_none(final_update.get("value_loss")),
        "final_entropy": _finite_or_none(final_update.get("entropy")),
        "final_approx_kl": _finite_or_none(final_update.get("approx_kl")),
        "first_window_mean_reward": _mean(first, "episode_reward"),
        "last_window_mean_reward": _mean(last, "episode_reward"),
        "first_window_feasible_rate": _bool_mean(first, "feasible"),
        "last_window_feasible_rate": _bool_mean(last, "feasible"),
        "window_size": window,
    }


def _summarize_source(rows: list[dict], source: str) -> dict:
    selected = [row for row in rows if row.get("source_type") == source]
    if not selected:
        return {
            "count": 0,
            "feasible_rate_before_repair": None,
            "feasible_rate_after_repair": None,
            "mean_cv_before_repair": None,
            "median_cv_before_repair": None,
            "mean_cv_after_repair": None,
            "median_cv_after_repair": None,
            "mean_coverage": None,
            "max_coverage": None,
            "mean_rsum_capacity": None,
            "max_rsum_capacity": None,
            "mean_repair_iter": None,
            "mean_quality_score": None,
        }
    return {
        "count": len(selected),
        "feasible_rate_before_repair": _bool_mean(selected, "feasible_before_repair"),
        "feasible_rate_after_repair": _bool_mean(selected, "feasible"),
        "mean_cv_before_repair": _mean(selected, "cv_before_repair"),
        "median_cv_before_repair": _median(selected, "cv_before_repair"),
        "mean_cv_after_repair": _mean(selected, "cv"),
        "median_cv_after_repair": _median(selected, "cv"),
        "mean_coverage": _mean(selected, "coverage"),
        "max_coverage": _max(selected, "coverage"),
        "mean_rsum_capacity": _mean(selected, "rsum_capacity"),
        "max_rsum_capacity": _max(selected, "rsum_capacity"),
        "mean_repair_iter": _mean(selected, "repair_iter"),
        "mean_quality_score": _mean(selected, "quality_score"),
    }


def _values(rows, key):
    return [float(row[key]) for row in rows if _finite_or_none(row.get(key)) is not None]


def _mean(rows, key):
    values = _values(rows, key)
    return float(np.mean(values)) if values else None


def _median(rows, key):
    values = _values(rows, key)
    return float(np.median(values)) if values else None


def _max(rows, key):
    values = _values(rows, key)
    return float(max(values)) if values else None


def _bool_mean(rows, key):
    return float(np.mean([_as_bool(row.get(key, False)) for row in rows])) if rows else None


def _difference(summary, left, right, key):
    left_value = summary[left].get(key)
    right_value = summary[right].get(key)
    if left_value is None or right_value is None:
        return None
    return float(left_value - right_value)


def _as_bool(value):
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _finite_or_none(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None

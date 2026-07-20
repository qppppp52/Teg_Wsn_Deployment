"""Sink-hard predicates and diagnostic shortages from the unified report."""
from __future__ import annotations

from src.constraints.constraint_report import evaluate_constraints


def heatsink_constraint_components(solution, ctx):
    report = evaluate_constraints(solution, ctx)
    hard = report.raw.sink_hard
    diagnostics = report.raw.sink_diagnostics
    return {
        "invalid_index_or_type_count": hard.invalid_index_or_type_count,
        "undeployed_owner_count": hard.undeployed_owner_count,
        "outside_allowed_neighborhood_count": hard.outside_allowed_neighborhood_count,
        "illegal_node_overlap_excess": hard.illegal_node_overlap_excess,
        "duplicate_excess": hard.duplicate_excess,
        "cross_owner_conflict_excess": hard.cross_owner_conflict_excess,
        "shortage_sensor": diagnostics.shortage_sensor,
        "shortage_ap": diagnostics.shortage_ap,
        "sink_hard": hard.total,
        "sink_diagnostics_shortage": diagnostics.shortage_sensor + diagnostics.shortage_ap,
        "total": report.components.sink,
    }


def check_heatsink_constraints(solution, ctx):
    """Return normalized primary SinkHard CV; shortages remain diagnostics."""
    return float(heatsink_constraint_components(solution, ctx)["total"])
"""Sink-hard predicates, diagnostics and deterministic cleanup."""
from __future__ import annotations

from src.constraints.constraint_report import evaluate_constraints, mark_physical_state_dirty
from src.heatsink.sink_overlap_rules import SinkOverlapRules


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


def repair_sink_hard_constraints(solution, ctx):
    """Remove declared heatsink placements that can never become physical.

    Forbidden cross-owner claims are removed from every claimant. This preserves the
    evaluator's conservative meaning: a contested grid supplies harvest to neither
    owner, rather than arbitrarily awarding it to one owner.
    """
    rules = SinkOverlapRules(ctx.config)
    K = int(ctx.num_candidates)
    cleaned = {"sensor": [[] for _ in range(K)], "ap": [[] for _ in range(K)]}
    claims: dict[int, list[tuple[str, int]]] = {}

    for kind, deployed, positions_by_owner in (
        ("sensor", solution.x, solution.z_sink_sensor),
        ("ap", solution.y, solution.z_sink_ap),
    ):
        for owner_id in range(K):
            if deployed[owner_id] != 1:
                continue
            seen: set[int] = set()
            for raw_grid in positions_by_owner[owner_id]:
                try:
                    grid_id = int(raw_grid)
                except (TypeError, ValueError, OverflowError):
                    continue
                if grid_id in seen or not _base_sink_claim_is_legal(
                    owner_id, grid_id, deployed, solution, ctx, rules
                ):
                    continue
                seen.add(grid_id)
                cleaned[kind][owner_id].append(grid_id)
                claims.setdefault(grid_id, []).append((kind, owner_id))

    if not rules.allow_sink_sink_overlap:
        contested = {
            grid_id for grid_id, owners in claims.items() if len(owners) > 1
        }
        if contested:
            for positions in cleaned["sensor"] + cleaned["ap"]:
                positions[:] = [grid_id for grid_id in positions if grid_id not in contested]

    if cleaned["sensor"] != solution.z_sink_sensor or cleaned["ap"] != solution.z_sink_ap:
        solution.z_sink_sensor = cleaned["sensor"]
        solution.z_sink_ap = cleaned["ap"]
        mark_physical_state_dirty(solution)
    return solution


def _base_sink_claim_is_legal(owner_id, grid_id, deployed, solution, ctx, rules):
    if grid_id < 0 or grid_id >= int(ctx.num_candidates):
        return False
    if not rules.allow_sink_outside_neighborhood:
        if grid_id not in {int(value) for value in ctx.neighbor_sets[owner_id]}:
            return False
    if grid_id == owner_id and not rules.count_self_grid_as_sink:
        return False
    if not rules.allow_sink_on_other_node:
        occupied = solution.x[grid_id] == 1 or solution.y[grid_id] == 1
        own_node = grid_id == owner_id and deployed[owner_id] == 1
        if occupied and not (own_node and rules.allow_self_node_sink_overlap):
            return False
    return True

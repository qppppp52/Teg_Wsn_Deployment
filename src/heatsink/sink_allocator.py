"""Deterministic reconstruction of minimum heatsink resources."""
from __future__ import annotations

from src.heatsink.sink_overlap_rules import SinkOverlapRules
from src.constraints.constraint_report import mark_physical_state_dirty


def rebuild_minimum_sink_allocation(solution, ctx):
    """Clear stale placements and allocate only the current minimum demand."""
    rules = SinkOverlapRules(ctx.config)
    K = int(ctx.num_candidates)
    previous_signature = _sink_position_signature(solution)
    for index in range(K):
        solution.z_sink_sensor[index] = []
        solution.z_sink_ap[index] = []

    nodes = []
    for sensor_id in range(K):
        if solution.x[sensor_id] == 1 and solution.n_sink_sensor[sensor_id] > 0:
            nodes.append(("sensor", sensor_id))
    for ap_id in range(K):
        if solution.y[ap_id] == 1 and solution.n_sink_ap[ap_id] > 0:
            nodes.append(("ap", ap_id))

    while True:
        candidates = []
        for kind, node_id in nodes:
            remaining = _remaining(solution, kind, node_id)
            if remaining <= 0:
                continue
            legal = _legal_grids(solution, kind, node_id, ctx, rules)
            if not legal:
                continue
            slack = len(legal) - remaining
            criticality = (
                int(solution.c[:, node_id].sum()) if kind == "ap" else 1
            )
            type_tie = 0 if kind == "ap" else 1
            candidates.append(
                (slack, -remaining, -criticality, type_tie, node_id, kind, legal)
            )
        if not candidates:
            break

        _, _, _, _, node_id, kind, legal = min(candidates)
        grid_id = min(
            legal,
            key=lambda grid: (
                _conflict_degree(grid, kind, node_id, nodes, solution, ctx, rules),
                grid,
            ),
        )
        _positions(solution, kind, node_id).append(int(grid_id))
    if _sink_position_signature(solution) != previous_signature:
        mark_physical_state_dirty(solution)
    return solution


def _positions(solution, kind, node_id):
    return (
        solution.z_sink_sensor[node_id]
        if kind == "sensor"
        else solution.z_sink_ap[node_id]
    )


def _required(solution, kind, node_id):
    return int(
        solution.n_sink_sensor[node_id]
        if kind == "sensor"
        else solution.n_sink_ap[node_id]
    )


def _remaining(solution, kind, node_id):
    return max(0, _required(solution, kind, node_id) - len(set(_positions(solution, kind, node_id))))


def _legal_grids(solution, kind, node_id, ctx, rules):
    occupied_by_owner = set(int(grid) for grid in _positions(solution, kind, node_id))
    return [
        grid_id
        for grid_id in sorted(set(int(value) for value in ctx.neighbor_sets[node_id]))
        if grid_id not in occupied_by_owner
        and rules.can_place_sink_on_grid(
            grid_id, node_id, solution, ctx, owner_kind=kind
        )
    ]


def _conflict_degree(grid_id, kind, node_id, nodes, solution, ctx, rules):
    degree = 0
    for other_kind, other_id in nodes:
        if (other_kind, other_id) == (kind, node_id):
            continue
        if _remaining(solution, other_kind, other_id) <= 0:
            continue
        if grid_id in _legal_grids(solution, other_kind, other_id, ctx, rules):
            degree += 1
    return degree


def allocate_all_sinks(solution, ctx):
    """Compatibility entry point for minimum deterministic reconstruction."""
    return rebuild_minimum_sink_allocation(solution, ctx)


def _sink_position_signature(solution):
    return (
        tuple(tuple(int(grid) for grid in positions) for positions in solution.z_sink_sensor),
        tuple(tuple(int(grid) for grid in positions) for positions in solution.z_sink_ap),
    )

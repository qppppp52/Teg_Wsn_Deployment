"""Pure reconstruction of effective heatsink ownership."""
from __future__ import annotations

from dataclasses import dataclass

from src.heatsink.sink_overlap_rules import SinkOverlapRules


SINK_OWNERSHIP_SEMANTICS_VERSION = 2


@dataclass(frozen=True)
class SinkOwnership:
    effective_sensor_positions: tuple[tuple[int, ...], ...]
    effective_ap_positions: tuple[tuple[int, ...], ...]
    duplicate_sensor_count: tuple[int, ...]
    duplicate_ap_count: tuple[int, ...]
    invalid_sensor_count: tuple[int, ...]
    invalid_ap_count: tuple[int, ...]
    forbidden_conflict_count: int

    @property
    def effective_sensor_count(self) -> tuple[int, ...]:
        return tuple(len(positions) for positions in self.effective_sensor_positions)

    @property
    def effective_ap_count(self) -> tuple[int, ...]:
        return tuple(len(positions) for positions in self.effective_ap_positions)


def build_sink_ownership(solution, ctx) -> SinkOwnership:
    """Return the unique, legal and conflict-free positions of every owner."""
    K = int(ctx.num_candidates)
    rules = SinkOverlapRules(ctx.config)
    candidate_positions: dict[tuple[str, int], set[int]] = {}
    duplicates = {"sensor": [0] * K, "ap": [0] * K}
    invalid = {"sensor": [0] * K, "ap": [0] * K}

    for kind, deployed, all_positions in (
        ("sensor", solution.x, solution.z_sink_sensor),
        ("ap", solution.y, solution.z_sink_ap),
    ):
        for owner_id in range(K):
            converted = []
            for raw_grid in all_positions[owner_id]:
                try:
                    converted.append(int(raw_grid))
                except (TypeError, ValueError, OverflowError):
                    invalid[kind][owner_id] += 1
            unique = set(converted)
            duplicates[kind][owner_id] += len(converted) - len(unique)
            valid_for_conflict = set()
            for grid_id in unique:
                if not _base_position_is_legal(
                    owner_id, grid_id, deployed, solution, ctx, rules
                ):
                    invalid[kind][owner_id] += 1
                    continue
                valid_for_conflict.add(grid_id)
            candidate_positions[(kind, owner_id)] = valid_for_conflict

    claims_by_grid: dict[int, list[tuple[str, int]]] = {}
    for owner, positions in candidate_positions.items():
        for grid_id in positions:
            claims_by_grid.setdefault(grid_id, []).append(owner)

    conflicting_grids = set()
    forbidden_conflict_count = 0
    if not rules.allow_sink_sink_overlap:
        for grid_id, owners in claims_by_grid.items():
            if len(owners) > 1:
                conflicting_grids.add(grid_id)
                forbidden_conflict_count += len(owners) - 1

    effective_sensor = []
    effective_ap = []
    for kind, target in (("sensor", effective_sensor), ("ap", effective_ap)):
        for owner_id in range(K):
            positions = candidate_positions[(kind, owner_id)] - conflicting_grids
            target.append(tuple(sorted(positions)))

    return SinkOwnership(
        effective_sensor_positions=tuple(effective_sensor),
        effective_ap_positions=tuple(effective_ap),
        duplicate_sensor_count=tuple(duplicates["sensor"]),
        duplicate_ap_count=tuple(duplicates["ap"]),
        invalid_sensor_count=tuple(invalid["sensor"]),
        invalid_ap_count=tuple(invalid["ap"]),
        forbidden_conflict_count=int(forbidden_conflict_count),
    )


def _base_position_is_legal(owner_id, grid_id, deployed, solution, ctx, rules):
    K = int(ctx.num_candidates)
    if deployed[owner_id] != 1 or grid_id < 0 or grid_id >= K:
        return False
    if not rules.allow_sink_outside_neighborhood:
        if grid_id not in {int(value) for value in ctx.neighbor_sets[owner_id]}:
            return False
    if grid_id == owner_id and not rules.count_self_grid_as_sink:
        return False
    if not rules.allow_sink_on_other_node:
        occupied_by_node = solution.x[grid_id] == 1 or solution.y[grid_id] == 1
        own_node = grid_id == owner_id and deployed[owner_id] == 1
        if occupied_by_node and not (own_node and rules.allow_self_node_sink_overlap):
            return False
    return True

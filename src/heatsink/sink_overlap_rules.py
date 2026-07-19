"""Configuration-aware heatsink placement rules."""


class SinkOverlapRules:
    def __init__(self, config):
        heatsink = config.get("heatsink", {})
        self.allow_self_node_sink_overlap = heatsink.get(
            "allow_self_node_sink_overlap", True
        )
        self.allow_node_node_overlap = heatsink.get("allow_node_node_overlap", False)
        self.allow_sink_sink_overlap = heatsink.get("allow_sink_sink_overlap", False)
        self.allow_sink_on_other_node = heatsink.get("allow_sink_on_other_node", False)
        self.allow_sink_outside_neighborhood = heatsink.get(
            "allow_sink_outside_neighborhood", False
        )
        self.count_self_grid_as_sink = heatsink.get("count_self_grid_as_sink", True)

    def can_place_sink_on_grid(
        self, grid_id, node_id, solution, ctx, *, owner_kind=None
    ):
        """Check a dynamic placement using ``(owner_kind, node_id)`` identity."""
        grid_id = int(grid_id)
        node_id = int(node_id)
        if owner_kind not in {None, "sensor", "ap"}:
            raise ValueError(f"Unknown heatsink owner kind: {owner_kind}")
        if grid_id < 0 or grid_id >= ctx.num_candidates:
            return False
        if not self.allow_sink_outside_neighborhood:
            neighbors = {int(value) for value in ctx.neighbor_sets[node_id]}
            if grid_id not in neighbors:
                return False
        if grid_id == node_id and not self.count_self_grid_as_sink:
            return False
        if not self.allow_sink_sink_overlap:
            for other_id in range(ctx.num_candidates):
                if (
                    (owner_kind, node_id) != ("sensor", other_id)
                    and grid_id in solution.z_sink_sensor[other_id]
                ):
                    return False
                if (
                    (owner_kind, node_id) != ("ap", other_id)
                    and grid_id in solution.z_sink_ap[other_id]
                ):
                    return False
        if not self.allow_sink_on_other_node:
            if solution.x[grid_id] == 1 or solution.y[grid_id] == 1:
                if grid_id != node_id or not self.allow_self_node_sink_overlap:
                    return False
        return True

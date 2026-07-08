class SinkOverlapRules:
    def __init__(self, config):
        h = config.get("heatsink", {})
        self.allow_self_node_sink_overlap = h.get("allow_self_node_sink_overlap", True)
        self.allow_node_node_overlap = h.get("allow_node_node_overlap", False)
        self.allow_sink_sink_overlap = h.get("allow_sink_sink_overlap", False)
        self.allow_sink_on_other_node = h.get("allow_sink_on_other_node", False)
        self.allow_sink_outside_neighborhood = h.get("allow_sink_outside_neighborhood", False)
        self.count_self_grid_as_sink = h.get("count_self_grid_as_sink", True)

    def can_place_sink_on_grid(self, grid_id, node_id, solution, ctx):
        if not self.allow_sink_outside_neighborhood:
            if grid_id not in ctx.neighbor_sets[node_id]:
                return False
        if not self.allow_sink_sink_overlap:
            for other in range(ctx.num_candidates):
                if other != node_id:
                    if grid_id in solution.z_sink_sensor[other]:
                        return False
                    if grid_id in solution.z_sink_ap[other]:
                        return False
        if not self.allow_sink_on_other_node:
            if solution.x[grid_id] == 1 or solution.y[grid_id] == 1:
                if grid_id != node_id or not self.allow_self_node_sink_overlap:
                    return False
        return True

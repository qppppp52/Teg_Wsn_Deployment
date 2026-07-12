"""Theoretical channel-capacity objective."""
from __future__ import annotations

import numpy as np
from src.physics.rate_model import compute_snr, compute_rate


def compute_rsum_capacity(solution, ctx):
    """Compute aggregate theoretical Shannon capacity for valid links."""
    total_capacity = 0.0
    per_link = []
    for si in range(ctx.num_candidates):
        if solution.x[si] == 0:
            continue
        for aj in range(ctx.num_candidates):
            if solution.c[si, aj] == 1:
                snr = compute_snr(
                    np.array(solution.p_tx[si, aj]),
                    np.array(ctx.channel_gain_matrix[si, aj]),
                    np.array(ctx.distance_matrix[si, aj]),
                    ctx.config,
                )
                capacity = compute_rate(snr, ctx.config)
                cap_value = float(np.asarray(capacity))
                total_capacity += cap_value
                per_link.append({
                    "sensor": int(si),
                    "ap": int(aj),
                    "capacity_bps": cap_value,
                })
    capacities = [item["capacity_bps"] for item in per_link]
    solution.metadata["rsum_capacity"] = float(total_capacity)
    solution.metadata["rsum_links"] = per_link
    solution.metadata["mean_link_capacity_bps"] = float(np.mean(capacities)) if capacities else 0.0
    solution.metadata["min_link_capacity_bps"] = float(np.min(capacities)) if capacities else 0.0
    solution.metadata["max_link_capacity_bps"] = float(np.max(capacities)) if capacities else 0.0
    return float(total_capacity)

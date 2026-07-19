"""Theoretical channel-capacity objective."""
from __future__ import annotations

import numpy as np

from src.physics.rate_model import compute_rate, compute_snr


def compute_link_capacity(sensor_id, ap_id, p_tx, ctx):
    """Compute one link's theoretical Shannon capacity in bit/s."""
    snr = compute_snr(
        np.asarray(float(p_tx)),
        np.asarray(ctx.channel_gain_matrix[sensor_id, ap_id]),
        np.asarray(ctx.distance_matrix[sensor_id, ap_id]),
        ctx.config,
    )
    return float(np.asarray(compute_rate(snr, ctx.config)))


def compute_rsum_capacity(solution, ctx):
    """Compute aggregate theoretical Shannon capacity for valid links."""
    total_capacity = 0.0
    per_link = []
    for sensor_id in range(ctx.num_candidates):
        if solution.x[sensor_id] == 0:
            continue
        for ap_id in range(ctx.num_candidates):
            if solution.c[sensor_id, ap_id] != 1:
                continue
            capacity = compute_link_capacity(
                sensor_id, ap_id, solution.p_tx[sensor_id, ap_id], ctx
            )
            total_capacity += capacity
            per_link.append(
                {
                    "sensor": int(sensor_id),
                    "ap": int(ap_id),
                    "capacity_bps": capacity,
                }
            )
    capacities = [item["capacity_bps"] for item in per_link]
    solution.metadata["rsum_capacity"] = float(total_capacity)
    solution.metadata["rsum_links"] = per_link
    solution.metadata["mean_link_capacity_bps"] = (
        float(np.mean(capacities)) if capacities else 0.0
    )
    solution.metadata["min_link_capacity_bps"] = (
        float(np.min(capacities)) if capacities else 0.0
    )
    solution.metadata["max_link_capacity_bps"] = (
        float(np.max(capacities)) if capacities else 0.0
    )
    return float(total_capacity)

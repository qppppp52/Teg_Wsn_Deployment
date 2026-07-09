"""Throughput objective with theoretical/actual throughput bookkeeping."""
from __future__ import annotations

import numpy as np
from src.physics.rate_model import compute_snr, compute_rate_components


def compute_throughput(solution, ctx):
    """Compute objective throughput and store capacity/actual metadata.

    ``throughput_capacity`` is the sum of theoretical Shannon capacities.
    ``throughput_actual`` is the cap-limited service throughput used as the
    optimization objective when ``channel.use_data_rate_cap`` is enabled.
    """
    total_capacity = 0.0
    total_actual = 0.0
    cap_limit = float(ctx.config.get("channel", {}).get("sensor_data_rate_bps", 2.0e6))
    per_link = []
    saturated = 0
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
                capacity, actual = compute_rate_components(snr, ctx.config)
                cap_value = float(np.asarray(capacity))
                actual_value = float(np.asarray(actual))
                total_capacity += cap_value
                total_actual += actual_value
                if ctx.config.get("channel", {}).get("use_data_rate_cap", False) and cap_value >= cap_limit - 1e-9:
                    saturated += 1
                per_link.append({
                    "sensor": int(si),
                    "ap": int(aj),
                    "capacity_bps": cap_value,
                    "actual_bps": actual_value,
                })
    n_links = len(per_link)
    capacities = [item["capacity_bps"] for item in per_link]
    solution.metadata["throughput_capacity"] = float(total_capacity)
    solution.metadata["throughput_actual"] = float(total_actual)
    solution.metadata["throughput_links"] = per_link
    solution.metadata["num_saturated_links"] = int(saturated)
    solution.metadata["saturated_link_ratio"] = float(saturated / n_links) if n_links else 0.0
    solution.metadata["mean_link_capacity_bps"] = float(np.mean(capacities)) if capacities else 0.0
    solution.metadata["min_link_capacity_bps"] = float(np.min(capacities)) if capacities else 0.0
    solution.metadata["max_link_capacity_bps"] = float(np.max(capacities)) if capacities else 0.0
    metric = ctx.config.get("objectives", {}).get("throughput_metric", "actual")
    if metric == "capacity":
        return float(total_capacity)
    return float(total_actual)

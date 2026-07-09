"""Convert selected candidate orders into MODE priority individuals."""
from __future__ import annotations

import numpy as np
from src.model.individual import Individual


def build_individual_from_orders(ctx, sensor_order, ap_order, priority_noise_std=0.03, rng=None):
    rng = rng or np.random.default_rng()
    im = ctx.index_mapping
    ind = Individual(im.num_sensor_candidates, im.num_ap_candidates)
    ind.rho_s = rng.uniform(0.0, 0.15, im.num_sensor_candidates)
    ind.rho_a = rng.uniform(0.0, 0.15, im.num_ap_candidates)

    sensor_order = list(sensor_order)
    ap_order = list(ap_order)
    for rank, gid in enumerate(sensor_order):
        local = im.Ls_global_to_local[int(gid)]
        if local >= 0:
            base = 1.0 - 0.7 * (rank / max(len(sensor_order), 1))
            ind.rho_s[local] = base
    for rank, gid in enumerate(ap_order):
        local = im.La_global_to_local[int(gid)]
        if local >= 0:
            base = 1.0 - 0.7 * (rank / max(len(ap_order), 1))
            ind.rho_a[local] = base

    noise = float(priority_noise_std)
    if noise > 0.0:
        ind.rho_s += rng.normal(0.0, noise, len(ind.rho_s))
        ind.rho_a += rng.normal(0.0, noise, len(ind.rho_a))
    ind.clip()
    return ind

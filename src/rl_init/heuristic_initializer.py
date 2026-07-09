"""Composite heuristic initializer for DRL-Init fallback and mixing."""
from __future__ import annotations

import numpy as np
from src.rl_init.candidate_features import build_candidate_features, FEATURE_NAMES
from src.rl_init.individual_builder import build_individual_from_orders


def create_composite_heuristic_individual(ctx, config, rng=None):
    rng = rng or np.random.default_rng()
    cfg = config.get("drl_init", config)
    features = build_candidate_features(ctx, None)
    names = {name: i for i, name in enumerate(FEATURE_NAMES)}
    coverage = features[:, names["coverage_score_norm"]]
    pgrid = features[:, names["Pgrid_norm"]]
    link = features[:, names["link_quality_to_ap_candidates_norm"]]
    risk = features[:, names["estimated_sink_conflict_risk_norm"]]
    im = ctx.index_mapping
    sensor_scores = coverage + pgrid + link - risk + rng.normal(0.0, 0.01, len(features))
    ap_scores = link + pgrid + coverage - risk + rng.normal(0.0, 0.01, len(features))
    max_s = int(cfg.get("max_selected_sensors", config.get("deployment", {}).get("max_sensors", 20)))
    max_a = int(cfg.get("max_selected_aps", config.get("deployment", {}).get("max_aps", 4)))
    sensor_ids = [int(g) for g in im.Ls_local_to_global[np.argsort(sensor_scores[im.Ls_local_to_global])[::-1][:max_s]]]
    ap_ids = []
    for gid in im.La_local_to_global[np.argsort(ap_scores[im.La_local_to_global])[::-1]]:
        gid = int(gid)
        if gid not in sensor_ids:
            ap_ids.append(gid)
        if len(ap_ids) >= max_a:
            break
    return build_individual_from_orders(ctx, sensor_ids, ap_ids, cfg.get("priority_noise_std", 0.03), rng)

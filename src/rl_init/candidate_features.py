"""Candidate feature construction for PPO initialization."""
from __future__ import annotations

import numpy as np

FEATURE_NAMES = [
    "x_norm", "y_norm", "z_norm",
    "face_0", "face_1", "face_2", "face_3", "face_4", "face_5",
    "Twall_norm", "Pgrid_norm", "coverage_score_norm", "link_quality_to_ap_candidates_norm",
    "distance_to_heat_source_norm", "distance_to_center_norm", "is_sensor_candidate",
    "is_ap_candidate", "is_selected_sensor", "is_selected_ap", "is_skipped",
    "estimated_energy_margin_norm", "estimated_snr_quality_norm", "estimated_sink_conflict_risk_norm",
]


def build_candidate_features(ctx, env=None) -> np.ndarray:
    n = int(ctx.num_candidates)
    coords_full = getattr(ctx, "candidate_points_full", None)
    coords = np.asarray(coords_full[:, :3] if coords_full is not None else ctx.candidate_coords, dtype=float)
    if coords.shape[0] != n:
        coords = np.zeros((n, 3), dtype=float)
    mins = coords.min(axis=0) if n else np.zeros(3)
    span = np.maximum(coords.max(axis=0) - mins, 1.0e-12) if n else np.ones(3)
    xyz = (coords - mins) / span

    faces = np.zeros((n, 6), dtype=float)
    if coords_full is not None and coords_full.shape[1] >= 4:
        face_ids = np.asarray(coords_full[:, 3], dtype=int)
        valid = (face_ids >= 0) & (face_ids < 6)
        faces[np.arange(n)[valid], face_ids[valid]] = 1.0

    twall = _norm(getattr(ctx, "T_r", np.zeros(n)))
    pgrid = _norm(getattr(ctx, "P_grid", np.zeros(n)))
    coverage = _coverage_score(ctx)
    link_quality = _link_quality(ctx)
    heat_dist = _distance_to_heat(ctx, coords)
    center_dist = _norm(np.linalg.norm(coords - coords.mean(axis=0, keepdims=True), axis=1)) if n else np.zeros(0)

    im = ctx.index_mapping
    is_sensor = np.asarray(im.sensor_mask, dtype=float)
    is_ap = np.asarray(im.ap_mask, dtype=float)
    selected_sensor = np.zeros(n, dtype=float)
    selected_ap = np.zeros(n, dtype=float)
    skipped = np.zeros(n, dtype=float)
    sink_risk = np.zeros(n, dtype=float)
    if env is not None:
        for gid in env.selected_sensors:
            selected_sensor[int(gid)] = 1.0
        for gid in env.selected_aps:
            selected_ap[int(gid)] = 1.0
        for gid in env.skipped_candidates:
            skipped[int(gid)] = 1.0
        selected = set(env.selected_sensors) | set(env.selected_aps)
        neighbors = getattr(ctx, "neighbor_sets", None) or []
        for gid in range(n):
            ns = neighbors[gid] if gid < len(neighbors) else []
            sink_risk[gid] = sum(1 for item in ns if int(item) in selected) / max(len(ns), 1)

    node_cfg = ctx.config.get("node", {}) if getattr(ctx, "config", None) else {}
    p_node = float(node_cfg.get("sensor_circuit_power", node_cfg.get("sensor_power_w", 1.0e-3)))
    energy_margin = _norm(np.asarray(getattr(ctx, "P_grid", np.zeros(n)), dtype=float) - p_node)
    snr_quality = _norm(link_quality)

    return np.column_stack([
        xyz, faces, twall, pgrid, coverage, link_quality, heat_dist, center_dist,
        is_sensor, is_ap, selected_sensor, selected_ap, skipped,
        energy_margin, snr_quality, sink_risk,
    ]).astype(np.float32)


def _norm(values):
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return arr
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi - lo <= 1.0e-12:
        return np.zeros_like(arr, dtype=float)
    return (arr - lo) / (hi - lo)


def _coverage_score(ctx):
    mat = getattr(ctx, "coverage_matrix", None)
    if mat is None or len(mat) == 0:
        return np.zeros(int(ctx.num_candidates), dtype=float)
    total = max(int(getattr(ctx, "num_targets", mat.shape[1])), 1)
    return np.asarray(mat, dtype=float).sum(axis=1) / total


def _link_quality(ctx):
    n = int(ctx.num_candidates)
    im = ctx.index_mapping
    if getattr(ctx, "potential_rate_matrix", None) is not None:
        mat = np.asarray(ctx.potential_rate_matrix, dtype=float)
    elif getattr(ctx, "channel_gain_matrix", None) is not None:
        mat = np.asarray(ctx.channel_gain_matrix, dtype=float)
    else:
        return np.zeros(n, dtype=float)
    ap_ids = np.asarray(im.La_local_to_global, dtype=int)
    if len(ap_ids) == 0:
        return np.zeros(n, dtype=float)
    values = np.max(mat[:, ap_ids], axis=1)
    return _norm(values)


def _distance_to_heat(ctx, coords):
    temp_cfg = ctx.config.get("temperature", {}) if getattr(ctx, "config", None) else {}
    source = np.asarray(temp_cfg.get("source_position", coords.mean(axis=0) if len(coords) else [0, 0, 0]), dtype=float)
    return _norm(np.linalg.norm(coords - source.reshape(1, 3), axis=1)) if len(coords) else np.zeros(0)

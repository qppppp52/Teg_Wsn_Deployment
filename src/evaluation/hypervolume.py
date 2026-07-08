"""Standard 2D hypervolume for maximization objectives."""
from __future__ import annotations

import numpy as np


def normalize_objectives(points, bounds):
    """Normalize objective points to [0, 1] using explicit bounds."""
    arr = np.asarray(points, dtype=float)
    if arr.size == 0:
        return arr.reshape(0, 2)
    lo = np.asarray([bounds[0][0], bounds[1][0]], dtype=float)
    hi = np.asarray([bounds[0][1], bounds[1][1]], dtype=float)
    return np.clip((arr - lo) / (hi - lo + 1e-12), 0.0, 1.0)


def _nondominated_2d_max(points):
    pts = np.asarray(points, dtype=float)
    if len(pts) == 0:
        return pts.reshape(0, 2)
    order = np.lexsort((-pts[:, 1], pts[:, 0]))
    sorted_pts = pts[order]
    keep = []
    best_y = -np.inf
    for x, y in sorted_pts[::-1]:
        if y > best_y + 1e-12:
            keep.append([x, y])
            best_y = y
    return np.asarray(keep[::-1], dtype=float)


def hypervolume_2d_max(points, ref=(0.0, 0.0), bounds=None):
    """Compute 2D dominated hypervolume for maximization objectives.

    ``points`` columns are [Coverage, Rsum]. If ``bounds`` is provided, points
    are normalized before HV so the result is bounded by 0..1 for ref=(0, 0).
    """
    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        return 0.0
    pts = pts.reshape(-1, 2)
    if bounds is not None:
        pts = normalize_objectives(pts, bounds)
    ref = np.asarray(ref, dtype=float)
    pts = pts[(pts[:, 0] > ref[0]) & (pts[:, 1] > ref[1])]
    if len(pts) == 0:
        return 0.0
    pts = _nondominated_2d_max(pts)
    pts = pts[np.argsort(pts[:, 0])]
    hv = 0.0
    prev_x = ref[0]
    for x, y in pts:
        hv += max(0.0, x - prev_x) * max(0.0, y - ref[1])
        prev_x = max(prev_x, x)
    return float(np.clip(hv, 0.0, 1.0 if bounds is not None else np.inf))


def compute_hv(objectives, ref_point=(0.0, 0.0), bounds=None):
    """Backward-compatible wrapper."""
    return hypervolume_2d_max(objectives, ref=ref_point, bounds=bounds)

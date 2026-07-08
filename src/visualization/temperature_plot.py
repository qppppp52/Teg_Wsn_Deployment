"""Temperature and TEG candidate export/plot helpers."""
from __future__ import annotations

import csv
import os
import numpy as np

FACE_NAMES = ["Top", "Bottom", "Front", "Back", "Left", "Right"]


def export_candidate_temperature(ctx, csv_path):
    """Export candidate wall temperature and P_grid data to CSV."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "x", "y", "z", "face", "T_wall_K", "T_wall_C", "P_grid"])
        for idx, row in enumerate(ctx.candidate_points_full):
            face_id = int(row[3])
            writer.writerow([idx, row[0], row[1], row[2], FACE_NAMES[face_id], ctx.T_r[idx], ctx.T_r[idx] - 273.15, ctx.P_grid[idx]])


def plot_temperature_faces(ctx, out_path):
    """Plot six face temperature scatter panels."""
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pts = ctx.candidate_points_full
    T_c = ctx.T_r - 273.15
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for face_id, ax in enumerate(axes.ravel()):
        mask = pts[:, 3].astype(int) == face_id
        sc = ax.scatter(pts[mask, 4], pts[mask, 5], c=T_c[mask], cmap="inferno", s=35, edgecolors="none")
        ax.set_title(FACE_NAMES[face_id])
        ax.set_xlabel("u (m)")
        ax.set_ylabel("v (m)")
        ax.set_aspect("equal", adjustable="box")
    fig.colorbar(sc, ax=axes.ravel().tolist(), label="Wall temperature (C)")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_pgrid_distribution(ctx, out_path):
    """Plot candidate TEG power distribution."""
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(ctx.P_grid, bins=30, color="#2A9D8F", edgecolor="#174C45")
    ax.set_xlabel("P_grid (W)")
    ax.set_ylabel("Candidate count")
    ax.set_title("TEG Candidate Harvesting Power Distribution")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

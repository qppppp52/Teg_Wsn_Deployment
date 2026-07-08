"""Synthetic center-air heat-source temperature field.

This model is intended for small algorithm-validation scenarios. It creates a
stable, non-uniform wall temperature distribution from an internal center heat
source and simplified natural-convection terms; it is not a replacement for a
high-fidelity ANSYS thermal field.
"""
from __future__ import annotations

import numpy as np


FACE_NAMES = ["Top", "Bottom", "Front", "Back", "Left", "Right"]


def _unit(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-12:
        return np.zeros_like(vec, dtype=float)
    return vec.astype(float) / norm


def compute_center_air_convection_temperature(
    point: np.ndarray,
    face_name: str,
    config: dict,
) -> float:
    """Compute one wall point temperature in Kelvin."""
    tcfg = config.get("temperature", {})
    scfg = config.get("space", {})

    ambient = float(tcfg.get("ambient", 295.15))
    source_temperature = float(tcfg.get("source_temperature", 343.15))
    source_position = np.asarray(
        tcfg.get("source_position", [1.5, 1.5, 1.5]), dtype=float
    )
    decay_length = max(float(tcfg.get("decay_length", 1.15)), 1e-9)
    buoyancy_strength = float(tcfg.get("buoyancy_strength", 0.35))
    face_multipliers = tcfg.get("face_heat_multipliers", {})
    plume_direction = _unit(np.asarray(tcfg.get("plume_direction", [0.25, 0.10, 1.0]), dtype=float))
    plume_strength = float(tcfg.get("plume_strength", 0.18))

    min_wall_temperature = float(tcfg.get("min_wall_temperature", ambient))
    max_wall_temperature = min(
        float(tcfg.get("max_wall_temperature", source_temperature)),
        source_temperature,
    )

    p = np.asarray(point, dtype=float)
    v = p - source_position
    distance = float(np.linalg.norm(v))
    delta_t = max(source_temperature - ambient, 0.0)
    base = np.exp(-distance / decay_length)

    lz = max(float(scfg.get("Lz", 3.0)), 1e-9)
    z_norm = float(np.clip(p[2] / lz, 0.0, 1.0))
    buoyancy = max(0.05, 1.0 + buoyancy_strength * (z_norm - 0.5))

    face_factor = float(face_multipliers.get(face_name, 1.0))
    direction = _unit(v)
    plume = 1.0 + plume_strength * max(0.0, float(np.dot(direction, plume_direction)))

    temperature = ambient + delta_t * base * buoyancy * face_factor * plume
    return float(np.clip(temperature, min_wall_temperature, max_wall_temperature))


def compute_center_air_convection_temperature_field(
    candidate_points: np.ndarray,
    space,
    config: dict,
) -> np.ndarray:
    """Compute synthetic wall temperatures for all candidate points."""
    temperatures = np.zeros(len(candidate_points), dtype=float)
    for idx, row in enumerate(candidate_points):
        face_id = int(row[3])
        face_name = FACE_NAMES[face_id] if 0 <= face_id < len(FACE_NAMES) else "Unknown"
        temperatures[idx] = compute_center_air_convection_temperature(
            row[:3], face_name, config
        )

    tcfg = config.get("temperature", {})
    min_std = float(tcfg.get("min_surface_temp_std", 0.0))
    min_range = float(tcfg.get("min_surface_temp_range", 0.0))
    std = float(np.std(temperatures))
    temp_range = float(np.max(temperatures) - np.min(temperatures))
    if std < min_std or temp_range < min_range:
        raise ValueError(
            "center_air_convection_synthetic temperature field is too uniform: "
            f"std={std:.3f}, range={temp_range:.3f}; required std>={min_std}, "
            f"range>={min_range}"
        )
    return temperatures

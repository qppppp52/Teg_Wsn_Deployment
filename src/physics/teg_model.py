"""TEG energy-harvesting model.

The thermal-resistance branch follows the modeling PDF:
    R_sink_grid = 1 / (h_sink * A_grid)
    delta_T = (T_wall - T_amb) * R_TEG / (R_TEG + R_sink_grid)
    P_grid = eta * sigma^2 * delta_T^2 / (4 * R_int)
"""
from __future__ import annotations

import numpy as np


def compute_delta_T_simplified(T_wall: np.ndarray, Tc: float) -> np.ndarray:
    """Simplified temperature difference: delta_T = T_wall - Tc."""
    return np.maximum(T_wall - Tc, 0.0)


def compute_delta_T_thermal_resistance(T_wall: np.ndarray, config: dict) -> np.ndarray:
    """Compute unit-grid TEG temperature difference from the PDF model."""
    tr = config.get("teg", {}).get("thermal_resistance", {})
    T_amb = float(tr.get("T_amb", 298.15))
    R_TEG = float(tr.get("R_TEG", 0.05))
    h_sink = float(tr.get("h_sink", 50.0))

    grid_spacing = float(config.get("discretization", {}).get("grid_spacing", 0.8))
    A_grid = grid_spacing * grid_spacing
    R_sink_grid = 1.0 / (h_sink * A_grid)

    delta_T = (T_wall - T_amb) * R_TEG / (R_TEG + R_sink_grid)
    return np.maximum(delta_T, 0.0)


def compute_grid_power_thermal_resistance(delta_T: np.ndarray, config: dict) -> np.ndarray:
    """Compute unit-grid harvestable power from Seebeck effect."""
    tr = config.get("teg", {}).get("thermal_resistance", {})
    sigma = float(tr.get("sigma", 0.1))
    R_int = float(tr.get("R_int", 1.0))
    eta = float(tr.get("eta", 0.8))
    return eta * (sigma ** 2 * delta_T ** 2) / (4 * R_int)


def compute_grid_power_simplified(delta_T: np.ndarray, k_teg: float) -> np.ndarray:
    """Simplified harvest model: P_grid = k_teg * delta_T^2."""
    return k_teg * delta_T ** 2


def compute_delta_T(T_wall: np.ndarray, config: dict) -> np.ndarray:
    """Dispatch temperature-difference calculation by configured TEG model."""
    teg_cfg = config.get("teg", {})
    model = teg_cfg.get("model_type", "simplified")
    if model == "simplified":
        Tc = teg_cfg.get("simplified", {}).get("Tc", 298.15)
        return compute_delta_T_simplified(T_wall, Tc)
    if model == "thermal_resistance":
        return compute_delta_T_thermal_resistance(T_wall, config)
    raise ValueError(f"Unknown TEG model: {model}")


def compute_grid_power(delta_T: np.ndarray, config: dict) -> np.ndarray:
    """Dispatch unit-grid power calculation by configured TEG model."""
    teg_cfg = config.get("teg", {})
    model = teg_cfg.get("model_type", "simplified")
    if model == "simplified":
        k = teg_cfg.get("simplified", {}).get("k_teg", 0.0005)
        return compute_grid_power_simplified(delta_T, k)
    if model == "thermal_resistance":
        return compute_grid_power_thermal_resistance(delta_T, config)
    raise ValueError(f"Unknown TEG model: {model}")

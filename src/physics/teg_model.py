"""TEG 热电采能模型 — 支持 simplified 和 thermal_resistance 双模式"""
import numpy as np


def compute_delta_T_simplified(T_wall: np.ndarray, Tc: float) -> np.ndarray:
    """简化温差: ΔT = T_wall - Tc"""
    return T_wall - Tc


def compute_delta_T_thermal_resistance(T_wall: np.ndarray, config: dict) -> np.ndarray:
    """
    建模文件公式:
      A_grid = grid_spacing^2          (单位网格面积,如0.8m间隔则0.64m^2)
      R_sink_grid = 1 / (h_sink * A_grid)
      ΔT_r = (T_wall - T_amb) * R_sink_grid / (R_TEG + R_sink_grid)
    """
    tr = config.get("teg", {}).get("thermal_resistance", {})
    T_amb = float(tr.get("T_amb", 298.15))
    R_TEG = float(tr.get("R_TEG", 0.05))
    h_sink = float(tr.get("h_sink", 50.0))

    # A_grid 从离散化参数中读取: grid_spacing^2
    gs = float(config.get("discretization", {}).get("grid_spacing", 0.8))
    A_grid = gs * gs

    R_sink_grid = 1.0 / (h_sink * A_grid)
    delta_T = (T_wall - T_amb) * R_sink_grid / (R_TEG + R_sink_grid)
    return np.maximum(delta_T, 0.0)


def compute_grid_power_thermal_resistance(delta_T: np.ndarray, config: dict) -> np.ndarray:
    """
    建模文件公式:
      P_grid_r = η × σ^2 × ΔT_r^2 / (4 × R_int)
    """
    tr = config.get("teg", {}).get("thermal_resistance", {})
    sigma = float(tr.get("sigma", 0.1))
    R_int = float(tr.get("R_int", 1.0))
    eta = float(tr.get("eta", 0.8))
    return eta * (sigma**2 * delta_T**2) / (4 * R_int)


def compute_grid_power_simplified(delta_T: np.ndarray, k_teg: float) -> np.ndarray:
    """简化模型: Pgrid = k_teg * ΔT^2"""
    return k_teg * delta_T ** 2


def compute_delta_T(T_wall: np.ndarray, config: dict) -> np.ndarray:
    """根据模型类型计算温差"""
    teg_cfg = config.get("teg", {})
    model = teg_cfg.get("model_type", "simplified")
    if model == "simplified":
        Tc = teg_cfg.get("simplified", {}).get("Tc", 298.15)
        return compute_delta_T_simplified(T_wall, Tc)
    elif model == "thermal_resistance":
        return compute_delta_T_thermal_resistance(T_wall, config)
    else:
        raise ValueError(f"Unknown TEG model: {model}")


def compute_grid_power(delta_T: np.ndarray, config: dict) -> np.ndarray:
    """统一采能计算入口"""
    teg_cfg = config.get("teg", {})
    model = teg_cfg.get("model_type", "simplified")
    if model == "simplified":
        k = teg_cfg.get("simplified", {}).get("k_teg", 0.0005)
        return compute_grid_power_simplified(delta_T, k)
    elif model == "thermal_resistance":
        return compute_grid_power_thermal_resistance(delta_T, config)
    else:
        raise ValueError(f"Unknown TEG model: {model}")

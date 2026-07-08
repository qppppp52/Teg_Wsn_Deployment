"""温度场统一接口"""
import numpy as np
from .analytic_temperature import generate_analytic_temperature


def compute_temperature(scenario, config: dict) -> np.ndarray:
    """
    返回每个候选点的温度 T_r。优先使用ANSYS，否则用解析温度场。
    """
    ansys_cfg = config.get("ansys_temperature", {})
    if ansys_cfg.get("enabled", False):
        # TODO: 后续接入 ANSYS 映射
        raise NotImplementedError("ANSYS temperature mapping not yet implemented")
    else:
        return generate_analytic_temperature(
            scenario.candidate_points,
            scenario.space.Lx, scenario.space.Ly, scenario.space.Lz,
            seed=config.get("seed", 42),
            config=config
        )

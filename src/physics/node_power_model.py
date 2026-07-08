"""节点动态功耗模型"""
import numpy as np


def compute_sensor_consumption(sensor_idx: int, p_tx: float, config: dict) -> float:
    """传感器功耗: P_sens + P_proc + p_tx"""
    scfg = config.get("sensor", {})
    return scfg.get("P_sens", 0.01) + scfg.get("P_proc", 0.005) + p_tx


def compute_ap_consumption(num_connections: int, config: dict) -> float:
    """AP功耗: P_idle + P_proc + P_rx * num_connections"""
    acfg = config.get("ap", {})
    return acfg.get("P_idle", 0.05) + acfg.get("P_proc", 0.02) + acfg.get("P_rx", 0.003) * num_connections


def compute_ap_min_service_consumption(config: dict) -> float:
    """AP最低非空服务功耗: P_idle + P_proc + P_rx * 1"""
    acfg = config.get("ap", {})
    return acfg.get("P_idle", 0.05) + acfg.get("P_proc", 0.02) + acfg.get("P_rx", 0.003) * 1

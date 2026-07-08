"""Shannon吞吐量计算。SNR = p_tx * |h|^2 / (N₀B * d^α)"""
import numpy as np


def compute_snr(p_tx: np.ndarray, channel_gain: np.ndarray,
                distance: np.ndarray, config: dict) -> np.ndarray:
    """
    计算链路 SNR。
    SNR_ij = p_tx_ij * |h_ij|^2 / (N₀ * B * d_ij^α)
    channel_gain = |h|^2   (仅衰落，不含路径损耗)
    distance = d_ij
    """
    ch = config.get("channel", {})
    N0 = float(ch.get("noise_power_density", 1e-12))
    B = float(ch.get("bandwidth", 1e6))
    N_total = N0 * B
    alpha = float(ch.get("path_loss_alpha", 2.0))
    d_alpha = np.maximum(distance, 0.01) ** alpha
    snr = p_tx * channel_gain / (N_total * d_alpha)
    return np.maximum(snr, 0.0)


def compute_rate(snr: np.ndarray, config: dict) -> np.ndarray:
    """R_ij = B * log2(1 + SNR)"""
    B = float(config.get("channel", {}).get("bandwidth", 1e6))
    return B * np.log2(1.0 + snr)

"""Rate and SNR utilities for theoretical channel capacity."""
from __future__ import annotations

import numpy as np


def compute_snr(p_tx: np.ndarray, channel_gain: np.ndarray,
                distance: np.ndarray, config: dict) -> np.ndarray:
    """Compute link SNR with distance-based path loss."""
    ch = config.get("channel", {})
    n0 = float(ch.get("noise_power_density", 1e-12))
    bandwidth = float(ch.get("bandwidth", 1e6))
    noise_total = n0 * bandwidth
    alpha = float(ch.get("path_loss_alpha", 2.0))
    d_alpha = np.maximum(distance, 0.01) ** alpha
    snr = p_tx * channel_gain / (noise_total * d_alpha)
    return np.maximum(snr, 0.0)


def compute_rate(snr: np.ndarray, config: dict) -> np.ndarray:
    """Return theoretical Shannon channel capacity."""
    bandwidth = float(config.get("channel", {}).get("bandwidth", 1e6))
    return bandwidth * np.log2(1.0 + snr)
"""通信信道模型：路径损耗 d^α + K-mu 衰落"""
import numpy as np


def compute_distance_matrix(points: np.ndarray) -> np.ndarray:
    """候选点间距离矩阵 (K, K)。points: (K, 3)"""
    diff = points[:, None, :] - points[None, :, :]
    return np.sqrt(np.sum(diff**2, axis=2))


def compute_channel_gain(config: dict, K: int, seed: int = 42) -> np.ndarray:
    """
    信道增益 |h̃|^2 —— K-mu 小尺度衰落，E[|h̃|^2] = 1。
    路径损耗 d^(-alpha) 在 SNR 和 ptx_min 公式中单独计算。

    K-mu 衰落模型：
    - K (k)：莱斯因子，主径功率与散射功率之比
    - mu (μ)：簇数参数
      * μ=1 → Rayleigh
      * μ→∞ → 确定性（无衰落）
      * μ=2, K>0 → 经典 Rician

    实现：级联 Nakagami-m 近似。
    |h| = sqrt( Σ_{j=1}^{2μ} (x_j^2) / (2μ) )
    其中 x_j ~ N(sqrt(K/(K+μ)), sqrt(1/(2μ*(K+μ))))
    这使得 E[|h|^2] = 1。
    """
    fading = config.get("channel", {}).get("fading", {})
    k_val = float(fading.get("k", 3.0))
    mu_val = float(fading.get("mu", 2.0))

    rng = np.random.default_rng(seed)
    n_clusters = int(2 * mu_val)
    sigma = np.sqrt(1.0 / (2 * mu_val * (k_val + mu_val)))
    mean_val = np.sqrt(k_val / (k_val + mu_val))

    h_mag_sq = np.zeros((K, K))
    for _ in range(n_clusters):
        x = mean_val + rng.normal(0, sigma, (K, K))
        h_mag_sq += x**2
    h_mag_sq /= (2 * mu_val)

    # 归一化使得 E[|h|^2] = 1
    if n_clusters > 0:
        actual_mean = np.mean(h_mag_sq)
        if actual_mean > 1e-12:
            h_mag_sq /= actual_mean

    return h_mag_sq


def compute_ptx_min(channel_gain: np.ndarray, distance: np.ndarray, config: dict) -> np.ndarray:
    """
    最小发射功率矩阵。
    ptx_min_ij = gamma_th * N0 * B * d_ij^alpha / |h_ij|^2
    """
    ch = config.get("channel", {})
    gamma_th = float(ch.get("snr_threshold", 10.0))
    N0 = float(ch.get("noise_power_density", 1e-12))
    B = float(ch.get("bandwidth", 1e6))
    N_total = N0 * B
    alpha = float(ch.get("path_loss_alpha", 2.0))

    d_alpha = np.maximum(distance, 0.01) ** alpha
    h_safe = np.maximum(channel_gain, 1e-15)
    ptx_min = gamma_th * N_total * d_alpha / h_safe
    return ptx_min


def compute_link_feasibility(ptx_min: np.ndarray, config: dict) -> np.ndarray:
    """链路可达矩阵 A_ij。1=可行，0=不可行"""
    ptx_max = float(config.get("channel", {}).get("p_tx_max", 0.5))
    return (ptx_min <= ptx_max).astype(int)

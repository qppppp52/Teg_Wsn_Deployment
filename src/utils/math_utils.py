"""数学工具：距离计算、归一化、安全除法等"""
import numpy as np


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """计算点集间的欧氏距离矩阵。a: (N,D), b: (M,D) -> (N,M)"""
    a2 = np.sum(a**2, axis=1, keepdims=True)
    b2 = np.sum(b**2, axis=1, keepdims=True).T
    ab = a @ b.T
    return np.sqrt(np.maximum(a2 + b2 - 2*ab, 0.0))


def safe_divide(num: np.ndarray, den: np.ndarray, fill=0.0) -> np.ndarray:
    """安全除法，den=0时填充fill"""
    result = np.full_like(num, fill, dtype=float)
    mask = den != 0
    result[mask] = num[mask] / den[mask]
    return result


def normalize_minmax(x: np.ndarray, min_val=None, max_val=None):
    """Min-Max 归一化到 [0,1]"""
    if min_val is None:
        min_val = np.min(x)
    if max_val is None:
        max_val = np.max(x)
    if max_val - min_val < 1e-12:
        return np.zeros_like(x)
    return (x - min_val) / (max_val - min_val)


def clip_array(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip(x, lo, hi)

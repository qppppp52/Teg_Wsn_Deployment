"""目标点集合 U 生成"""
import numpy as np


def generate_random_targets(num_targets: int, Lx: float, Ly: float, Lz: float,
                            margin: float = 0.1) -> np.ndarray:
    """在密闭空间内部随机生成目标点。返回 (M, 3)"""
    np.random.seed(None)  # 使用当前全局状态
    targets = np.random.uniform(
        low=[margin, margin, margin],
        high=[Lx - margin, Ly - margin, Lz - margin],
        size=(num_targets, 3)
    )
    return targets

"""散热片邻域生成 — 按建模文件：切比雪夫距离 max(|Δu|,|Δv|) <= q*gs"""
import numpy as np


def compute_neighborhoods(candidate_points, neighbor_order, grid_spacing=1.0):
    """
    建模文件公式：
      N_q(r) = {k | max(|u_k - u_r|, |v_k - v_r|) <= q*gs, same face as r}
    其中 (u,v) 为面内坐标，gs 为网格间距。
    返回: (neighbor_sets, nmax)
    """
    K = len(candidate_points)
    u = np.array(candidate_points[:, 4], dtype=float)
    v = np.array(candidate_points[:, 5], dtype=float)
    faces = np.array(candidate_points[:, 3], dtype=int)
    threshold = neighbor_order * grid_spacing

    neighbor_sets = []
    nmax = np.zeros(K, dtype=int)

    for r in range(K):
        same_face = (faces == faces[r])
        du = np.abs(u - u[r])
        dv = np.abs(v - v[r])
        in_nb = same_face & (np.maximum(du, dv) <= threshold + 1e-9)
        neighbors = np.where(in_nb)[0]
        neighbor_sets.append(neighbors)
        nmax[r] = len(neighbors)

    return neighbor_sets, nmax

"""内表面离散化，生成候选部署点 L"""
import numpy as np
from .enclosed_space import EnclosedSpace
from .face_coordinate import identify_face, to_face_uv


def discretize_surface(space: EnclosedSpace, grid_spacing: float) -> np.ndarray:
    """
    离散化六个内表面，生成候选点。
    返回: (K, 7) 数组 [x, y, z, face_id_int, u, v, global_id]
    """
    points = []
    Lx, Ly, Lz = space.Lx, space.Ly, space.Lz

    for face in space.faces:
        (u_min, u_max), (v_min, v_max) = space.get_face_bounds(face)
        # 建模文件: 以各网格中心点的三维空间坐标表示候选部署位置
        start = grid_spacing / 2.0
        u_vals = np.arange(start, u_max, grid_spacing)
        v_vals = np.arange(start, v_max, grid_spacing)

        uu, vv = np.meshgrid(u_vals, v_vals)
        uu, vv = uu.ravel(), vv.ravel()

        for ui, vi in zip(uu, vv):
            if face == "Top":
                x, y, z = ui, vi, Lz
            elif face == "Bottom":
                x, y, z = ui, vi, 0.0
            elif face == "Front":
                x, y, z = ui, 0.0, vi
            elif face == "Back":
                x, y, z = ui, Ly, vi
            elif face == "Left":
                x, y, z = 0.0, ui, vi
            else:  # Right
                x, y, z = Lx, ui, vi
            points.append([x, y, z, face, ui, vi])

    points = np.array(points, dtype=object)
    K = len(points)
    result = np.zeros((K, 7))
    result[:, 0:3] = np.array(points[:, :3].tolist(), dtype=float)
    # face_id编码为int: Top=0, Bottom=1, Front=2, Back=3, Left=4, Right=5
    face_map = {f: i for i, f in enumerate(space.faces)}
    result[:, 3] = [face_map[p[3]] for p in points]
    result[:, 4:6] = np.array(points[:, 4:6].tolist(), dtype=float)
    result[:, 6] = np.arange(K)  # global_candidate_id
    return result

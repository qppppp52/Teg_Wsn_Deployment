"""六面局部坐标定义：3D <-> 2D面内坐标"""
import numpy as np

# 每个面的平面轴和固定值
FACE_PLANE = {
    "Top":    ("z", "Lz"),
    "Bottom": ("z", 0),
    "Front":  ("y", 0),
    "Back":   ("y", "Ly"),
    "Left":   ("x", 0),
    "Right":  ("x", "Lx"),
}

# 每个面的面内坐标映射 (u_axis, v_axis)
FACE_UV = {
    "Top":    ("x", "y"),
    "Bottom": ("x", "y"),
    "Front":  ("x", "z"),
    "Back":   ("x", "z"),
    "Left":   ("y", "z"),
    "Right":  ("y", "z"),
}


def identify_face(point: np.ndarray, Lx: float, Ly: float, Lz: float, tol=1e-9) -> str:
    """判断3D点属于哪个面"""
    x, y, z = point
    if abs(z - Lz) <= tol: return "Top"
    if abs(z) <= tol: return "Bottom"
    if abs(y) <= tol: return "Front"
    if abs(y - Ly) <= tol: return "Back"
    if abs(x) <= tol: return "Left"
    if abs(x - Lx) <= tol: return "Right"
    # 默认最近面
    dists = {abs(z-Lz): "Top", abs(z): "Bottom", abs(y): "Front",
             abs(y-Ly): "Back", abs(x): "Left", abs(x-Lx): "Right"}
    return dists[min(dists)]


def to_face_uv(point: np.ndarray, face: str) -> tuple:
    """3D点 -> (u,v) 面内坐标"""
    x, y, z = point
    u_axis, v_axis = FACE_UV[face]
    u = x if u_axis == "x" else (y if u_axis == "y" else z)
    v = z if v_axis == "z" else (y if v_axis == "y" else x)
    return u, v


def face_coords_to_global(face: str, u: float, v: float, Lx: float, Ly: float, Lz: float) -> np.ndarray:
    """面内(u,v) -> 3D坐标"""
    if face == "Top": return np.array([u, v, Lz])
    if face == "Bottom": return np.array([u, v, 0.0])
    if face == "Front": return np.array([u, 0.0, v])
    if face == "Back": return np.array([u, Ly, v])
    if face == "Left": return np.array([0.0, u, v])
    if face == "Right": return np.array([Lx, u, v])

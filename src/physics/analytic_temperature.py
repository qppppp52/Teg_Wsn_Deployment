"""解析温度场：局部表面热源有限影响半径 + 最强热源主导模型"""
import numpy as np


def sample_heat_source_ids(candidate_points: np.ndarray,
                           seed: int = 42,
                           config: dict = None) -> list[int]:
    """Sample heat-source candidate ids using the same RNG logic as the temperature field."""
    tcfg = config.get("temperature", {}) if config else {}
    n_min = int(tcfg.get("n_sources_min", 2))
    n_max = int(tcfg.get("n_sources_max", 3))

    rng = np.random.default_rng(seed)
    faces = candidate_points[:, 3].astype(int)
    face_name_to_id = {
        "Top": 0,
        "Bottom": 1,
        "Front": 2,
        "Back": 3,
        "Left": 4,
        "Right": 5,
    }
    n_src = rng.integers(n_min, n_max + 1)

    fixed_faces_cfg = tcfg.get("fixed_source_faces", [])
    fixed_face_ids = []
    for face_name in fixed_faces_cfg:
        if face_name in face_name_to_id:
            fixed_face_ids.append(face_name_to_id[face_name])

    if fixed_face_ids:
        chosen_faces = np.array(fixed_face_ids[:n_src], dtype=int)
    else:
        available = list(range(6))
        chosen_faces = rng.choice(available, size=min(n_src, 6), replace=False)

    source_ids = []
    for face_id in chosen_faces:
        face_candidates = np.where(faces == face_id)[0]
        source_ids.append(int(rng.choice(face_candidates)))
    return source_ids


def generate_analytic_temperature(candidate_points: np.ndarray,
                                  Lx: float, Ly: float, Lz: float,
                                  seed: int = 42,
                                  config: dict = None) -> np.ndarray:
    """
    局部表面热源有限影响半径 + 最强热源主导模型。

    热源数量由配置中的 n_sources_min / n_sources_max 控制。
    每个热源仅影响距离不超过 R_heat 的候选点，超出 R_heat 的
    候选点保持环境温度 Tamb。
    多个热源贡献取 max（不作求和叠加），防止人为抬高温度。
    使用表面切比雪夫距离近似热传导路径。

    T_k = Tamb + max_i[ delta_i * exp(-d_surf(i,k) / lambda_i) * I(d_surf(i,k) <= R_heat) ]

    candidate_points: (K, 7) [x, y, z, face_id, u, v, global_id]
    """
    tcfg = config.get("temperature", {}) if config else {}
    Tamb = float(tcfg.get("ambient", 295.15))
    T_max = float(tcfg.get("T_max", 328.15))
    T_source = float(tcfg.get("source_temperature", 323.15))
    delta = T_source - Tamb
    decay = float(tcfg.get("decay", 0.8))
    R_heat = float(tcfg.get("R_heat", 1.8))

    rng = np.random.default_rng(seed)
    K = len(candidate_points)
    faces = candidate_points[:, 3].astype(int)
    u = candidate_points[:, 4].astype(float)
    v = candidate_points[:, 5].astype(float)
    coords_3d = candidate_points[:, :3].astype(float)

    # ---- 1) 热源选取：不同面 ----------
    source_ids = sample_heat_source_ids(candidate_points, seed=seed, config=config)

    # ---- 2) 温度计算：局部影响 + 最强主导 ----------
    temperature = np.full(K, Tamb, dtype=float)

    for src_id in source_ids:
        src_face = faces[src_id]
        src_u = u[src_id]
        src_v = v[src_id]
        src_3d = coords_3d[src_id]

        # 表面距离近似：
        #  同面：Chebyshev 距离 max(|du|,|dv|)
        #  异面：从源面边缘到目标面的 3D 最短距离作为额外惩罚
        same_face = (faces == src_face)
        du = np.abs(u - src_u)
        dv = np.abs(v - src_v)
        d_surface = np.full(K, np.inf)

        # 同面
        d_surface[same_face] = np.maximum(du[same_face], dv[same_face])

        # 异面：近似为 3D 距离加上面间惩罚
        diff_face = ~same_face
        if diff_face.any():
            # 源面四个角坐标用于计算到异面的最短距离
            d3d = np.linalg.norm(coords_3d[diff_face] - src_3d, axis=1)
            # 最小距离不应小于网格间距
            d3d_clipped = np.maximum(d3d, 1.5)
            d_surface[diff_face] = d3d_clipped

        # 影响半径截断
        in_radius = (d_surface <= R_heat + 1e-9)
        contribution = np.where(
            in_radius,
            delta * np.exp(-d_surface / decay),
            0.0
        )
        # 最强主导：取 max
        temperature = np.maximum(temperature, Tamb + contribution)

    # 壁面上下限
    temperature = np.clip(temperature, Tamb, T_max)
    temperature += rng.normal(0, 0.3, size=K)
    temperature = np.clip(temperature, Tamb, T_max)
    return temperature


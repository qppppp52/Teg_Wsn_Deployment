"""实验结果读写"""
import os
import pickle
import numpy as np


def save_pareto(solutions, path: str):
    """保存 feasible Pareto 解集为 npz，包含目标值和完整部署信息"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    feasible = [s for s in solutions if s.feasible]
    objectives = np.array([[s.coverage, s.rsum_capacity] for s in feasible])

    # 保存选中的部署信息
    K = len(feasible[0].x) if feasible else 0
    if K > 0:
        N = len(feasible)
        x_all = np.array([s.x for s in feasible], dtype=np.int8)
        y_all = np.array([s.y for s in feasible], dtype=np.int8)
        c_all = np.array([s.c for s in feasible], dtype=np.int8)
        ptx_all = np.array([s.p_tx for s in feasible], dtype=np.float32)
        ns_s = np.array([s.n_sink_sensor for s in feasible], dtype=np.int16)
        na_s = np.array([s.n_sink_ap for s in feasible], dtype=np.int16)
        zs_mat = np.zeros((N, K), dtype=np.int16)
        za_mat = np.zeros((N, K), dtype=np.int16)
        cv_all = np.array([s.cv for s in feasible], dtype=np.float32)
        # 展平散热片占位信息
        for i, s in enumerate(feasible):
            for si in range(K):
                if s.x[si] == 1:
                    for g in s.z_sink_sensor[si]:
                        if 0 <= g < K:
                            zs_mat[i, g] = 1
            for ai in range(K):
                if s.y[ai] == 1:
                    for g in s.z_sink_ap[ai]:
                        if 0 <= g < K:
                            za_mat[i, g] = 1

        np.savez_compressed(path,
            objectives=objectives,
            x=x_all, y=y_all,
            c=c_all, p_tx=ptx_all,
            n_sink_sensor=ns_s, n_sink_ap=na_s,
            z_sink_sensor=zs_mat, z_sink_ap=za_mat,
            cv=cv_all,
            feasible_mask=np.ones(len(feasible), dtype=bool),
            solution_count=N,
        )
    else:
        np.savez_compressed(path, objectives=objectives)


def load_pareto(path: str) -> np.ndarray:
    data = np.load(path)
    return data["objectives"]


def save_solutions_raw(solutions, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(solutions, f)


def save_log(text: str, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)

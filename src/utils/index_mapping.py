"""候选点索引映射模块：管理 global_candidate_id、Ls/La local index"""
import numpy as np


class IndexMapping:
    """管理候选点原始索引与筛选后局部索引的映射"""
    def __init__(self, num_candidates: int, sensor_mask: np.ndarray, ap_mask: np.ndarray):
        self.num_original = num_candidates
        self.sensor_mask = sensor_mask.copy()  # bool (K,)
        self.ap_mask = ap_mask.copy()          # bool (K,)

        # Ls 局部索引映射
        self.Ls_global_to_local = -np.ones(num_candidates, dtype=int)
        self.Ls_local_to_global = np.where(sensor_mask)[0]
        self.Ls_global_to_local[self.Ls_local_to_global] = np.arange(len(self.Ls_local_to_global))

        # La 局部索引映射
        self.La_global_to_local = -np.ones(num_candidates, dtype=int)
        self.La_local_to_global = np.where(ap_mask)[0]
        self.La_global_to_local[self.La_local_to_global] = np.arange(len(self.La_local_to_global))

    @property
    def num_sensor_candidates(self): return len(self.Ls_local_to_global)

    @property
    def num_ap_candidates(self): return len(self.La_local_to_global)

    def is_same_physical_point(self, ls_local: int, la_local: int) -> bool:
        """判断Ls局部索引和La局部索引是否指向同一物理候选点"""
        gid_s = self.Ls_local_to_global[ls_local]
        gid_a = self.La_local_to_global[la_local]
        return gid_s == gid_a

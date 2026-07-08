"""场景构建总控"""
import numpy as np
from .enclosed_space import EnclosedSpace
from .surface_discretizer import discretize_surface
from .target_generator import generate_random_targets


class Scenario:
    """统一场景对象"""
    def __init__(self, config: dict):
        self.config = config
        sc = config.get("space", {})
        self.space = EnclosedSpace(sc.get("Lx", 5.0), sc.get("Ly", 4.0), sc.get("Lz", 3.0))
        self.candidate_points = None  # (K,7): x,y,z,face_id,u,v,global_id
        self.target_points = None     # (M,3)
        self.num_candidates = 0
        self.num_targets = 0

    def build(self):
        dcfg = self.config.get("discretization", {})
        spacing = dcfg.get("grid_spacing", 0.5)
        self.candidate_points = discretize_surface(self.space, spacing)
        self.num_candidates = len(self.candidate_points)

        tcfg = self.config.get("targets", {})
        num = tcfg.get("num_targets", 100)
        self.target_points = generate_random_targets(num, self.space.Lx, self.space.Ly, self.space.Lz)
        self.num_targets = len(self.target_points)
        return self

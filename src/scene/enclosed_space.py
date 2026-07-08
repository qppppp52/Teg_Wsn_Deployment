"""密闭空间几何定义"""
import numpy as np


class EnclosedSpace:
    """正六面体/长方体密闭空间"""
    def __init__(self, Lx: float, Ly: float, Lz: float):
        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz
        self.faces = ["Top", "Bottom", "Front", "Back", "Left", "Right"]

    def get_face_bounds(self, face: str) -> tuple:
        """返回面内两个坐标轴的(min,max)"""
        if face in ("Top", "Bottom"):
            return (0, self.Lx), (0, self.Ly)
        elif face in ("Front", "Back"):
            return (0, self.Lx), (0, self.Lz)
        else:  # Left, Right
            return (0, self.Ly), (0, self.Lz)

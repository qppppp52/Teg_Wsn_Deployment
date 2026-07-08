import numpy as np
from src.utils.math_utils import euclidean_distance

def compute_coverage_matrix(candidate_coords, target_coords, Rs):
    dist = euclidean_distance(candidate_coords, target_coords)
    return (dist <= Rs).astype(np.int8)

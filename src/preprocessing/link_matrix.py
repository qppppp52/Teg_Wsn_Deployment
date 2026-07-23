import numpy as np
from src.physics.channel_model import (
    compute_distance_matrix, compute_channel_gain,
    compute_ptx_min, compute_link_feasibility
)
from src.physics.rate_model import compute_snr, compute_rate

def compute_all_link_matrices(candidate_coords, config, seed):
    distance = compute_distance_matrix(candidate_coords)
    channel_gain = compute_channel_gain(config, candidate_coords.shape[0], seed)
    ptx_min = compute_ptx_min(channel_gain, distance, config)
    link_feasible = compute_link_feasibility(ptx_min, config)
    snr_est = compute_snr(ptx_min, channel_gain, distance, config)
    potential_rate = compute_rate(snr_est, config)
    return {
        "distance": distance,
        "channel_gain": channel_gain,
        "ptx_min": ptx_min,
        "link_feasible": link_feasible,
        "potential_rate": potential_rate,
    }

import numpy as np

def compute_diversity(solutions):
    if len(solutions)<2: return 0.0
    objs = np.array([[s.coverage, s.throughput] for s in solutions])
    n = len(objs)
    dists = []
    for i in range(n):
        others = np.delete(objs, i, axis=0)
        dists.append(np.min(np.linalg.norm(others-objs[i], axis=1)))
    d_mean = np.mean(dists)
    if d_mean<1e-12: return 0.0
    return float(np.std(dists)/d_mean)

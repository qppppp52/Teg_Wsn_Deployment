import numpy as np

def compute_hv(objectives, ref_point):
    if len(objectives)==0: return 0.0
    idx = np.argsort(objectives[:,0])[::-1]
    objs = objectives[idx]
    hv, prev_y = 0.0, ref_point[1]
    for i in range(len(objs)):
        hv += max(0.0, objs[i,0]-ref_point[0]) * max(0.0, prev_y-ref_point[1])
        prev_y = min(prev_y, objs[i,1])
    return hv

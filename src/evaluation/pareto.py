import numpy as np

def is_dominated_by(obj_a, obj_b):
    return (obj_b[0]>=obj_a[0] and obj_b[1]>=obj_a[1] and
            (obj_b[0]>obj_a[0] or obj_b[1]>obj_a[1]))

def extract_non_dominated(objectives):
    n = len(objectives)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i!=j and is_dominated_by(objectives[i], objectives[j]):
                dominated[i] = True
                break
    return np.where(~dominated)[0]

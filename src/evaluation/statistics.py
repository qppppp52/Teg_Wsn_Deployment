import numpy as np

def compute_statistics(runs_data):
    arr = np.array(runs_data)
    return {"mean":float(np.mean(arr)),"std":float(np.std(arr)),
            "best":float(np.max(arr)),"worst":float(np.min(arr))}

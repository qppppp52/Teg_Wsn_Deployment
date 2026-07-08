import numpy as np

def decode_deployment(individual, ctx):
    im = ctx.index_mapping
    K = ctx.num_candidates
    x = np.zeros(K, dtype=np.int8)
    y = np.zeros(K, dtype=np.int8)
    scfg = ctx.config.get("deployment", {})
    max_sensors = scfg.get("max_sensors", 20)
    max_aps = scfg.get("max_aps", 4)
    order_s = np.argsort(individual.rho_s)[::-1]
    for local_idx in order_s[:max_sensors]:
        global_id = im.Ls_local_to_global[local_idx]
        if y[global_id] == 0:
            x[global_id] = 1
    order_a = np.argsort(individual.rho_a)[::-1]
    for local_idx in order_a[:max_aps]:
        global_id = im.La_local_to_global[local_idx]
        if x[global_id] == 0:
            y[global_id] = 1
    return x, y

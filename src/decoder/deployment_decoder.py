import numpy as np


DEPLOYMENT_DECODER_SEMANTICS_VERSION = 2


def decode_deployment(individual, ctx):
    im = ctx.index_mapping
    K = ctx.num_candidates
    x = np.zeros(K, dtype=np.int8)
    y = np.zeros(K, dtype=np.int8)
    scfg = ctx.config.get("deployment", {})
    max_sensors = int(scfg.get("max_sensors", 20))
    max_aps = int(scfg.get("max_aps", 4))
    count_mode = scfg.get("count_mode", "topk_up_to_max")
    if count_mode != "topk_up_to_max":
        raise ValueError("deployment.count_mode must be 'topk_up_to_max'")
    order_s = np.argsort(individual.rho_s)[::-1]
    for local_idx in order_s:
        if int(x.sum()) >= max_sensors:
            break
        global_id = im.Ls_local_to_global[local_idx]
        if y[global_id] == 0:
            x[global_id] = 1
    order_a = np.argsort(individual.rho_a)[::-1]
    for local_idx in order_a:
        if int(y.sum()) >= max_aps:
            break
        global_id = im.La_local_to_global[local_idx]
        if x[global_id] == 0:
            y[global_id] = 1
    return x, y

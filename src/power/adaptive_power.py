def adaptive_power(sensor_idx, ap_idx, ptx_min, ptx_up, strategy="conservative", eta=0.3):
    if strategy == "conservative":
        eta_used = 0.0
    elif strategy == "rsum_capacity_priority":
        eta_used = 1.0
    elif strategy == "energy_balanced":
        eta_used = 0.5
    else:
        eta_used = float(eta)
    return ptx_min + eta_used * max(0.0, ptx_up - ptx_min)

"""Action mask construction for the DRL initialization environment."""
from __future__ import annotations

import numpy as np
from src.rl_init.action_space import ROLE_SENSOR, ROLE_AP, ROLE_SKIP, ROLE_STOP


def build_joint_action_mask(env) -> np.ndarray:
    n = int(getattr(env.ctx, "num_candidates", 0))
    mask = np.zeros((n, 4), dtype=bool)
    if n <= 0:
        return mask

    im = env.ctx.index_mapping
    selected_sensors = set(env.selected_sensors)
    selected_aps = set(env.selected_aps)
    skipped = set(env.skipped_candidates)
    unavailable = selected_sensors | selected_aps | skipped

    max_sensors = int(env.cfg.get("max_selected_sensors", env._deployment_limit("max_sensors", 20)))
    max_aps = int(env.cfg.get("max_selected_aps", env._deployment_limit("max_aps", 4)))
    min_sensors = int(env.cfg.get("min_selected_sensors", env._deployment_limit("min_sensors", 1)))
    min_aps = int(env.cfg.get("min_selected_aps", env._deployment_limit("min_aps", 1)))

    can_add_sensor = len(selected_sensors) < max_sensors
    can_add_ap = len(selected_aps) < max_aps
    for gid in range(n):
        if gid in unavailable:
            continue
        if can_add_sensor and im.Ls_global_to_local[gid] >= 0 and gid not in selected_aps:
            mask[gid, ROLE_SENSOR] = True
        if can_add_ap and im.La_global_to_local[gid] >= 0 and gid not in selected_sensors:
            mask[gid, ROLE_AP] = True
        mask[gid, ROLE_SKIP] = True

    stop_allowed = bool(env.cfg.get("allow_stop_action", True))
    if stop_allowed and len(selected_sensors) >= min_sensors and len(selected_aps) >= min_aps:
        mask[0, ROLE_STOP] = True

    if not np.any(mask):
        if stop_allowed:
            mask[0, ROLE_STOP] = True
        else:
            candidates = [gid for gid in range(n) if gid not in unavailable]
            if candidates:
                mask[candidates[0], ROLE_SKIP] = True
    return mask

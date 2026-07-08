"""Constraint-violation pressure normalization."""


def normalize_cv_components(solution, ctx) -> dict:
    """Return 0..1 pressure values for each CV component and total CV."""
    cfg = ctx.config.get("constraints", {}) if ctx is not None else {}
    refs = cfg.get("cv_pressure_refs", {})
    defaults = {
        "deploy": 1.0,
        "link": 10.0,
        "capacity": 10.0,
        "energy": 1.0,
        "sink": 10.0,
        "service": 5.0,
        "total": 20.0,
    }

    raw = {
        "deploy": getattr(solution, "cv_deploy", 0.0),
        "link": getattr(solution, "cv_link", 0.0),
        "capacity": getattr(solution, "cv_capacity", 0.0),
        "energy": getattr(solution, "cv_energy", 0.0),
        "sink": getattr(solution, "cv_sink", 0.0),
        "service": getattr(solution, "cv_service", 0.0),
        "total": getattr(solution, "cv", 0.0),
    }
    pressures = {}
    for key, value in raw.items():
        ref = float(refs.get(key, defaults[key]))
        pressures[key] = float(min(max(value, 0.0) / (ref + 1e-12), 1.0))
    return pressures

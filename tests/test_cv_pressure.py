from src.constraints.cv_pressure import normalize_cv_components
from src.model.solution import Solution


class Ctx:
    config = {"constraints": {"cv_pressure_refs": {"deploy": 1, "link": 2, "power": 4, "energy": 8, "sink": 16, "service": 32, "total": 64}}}


def test_cv_pressure_components_are_clipped_to_unit_interval():
    s = Solution(2)
    s.cv_deploy = 0.5
    s.cv_link = 4.0
    s.cv_power = 2.0
    s.cv_energy = 8.0
    s.cv_sink = 99.0
    s.cv_service = 0.0
    s.cv = 32.0
    pressures = normalize_cv_components(s, Ctx())
    assert set(pressures) == {"deploy", "link", "power", "energy", "sink", "service", "total"}
    assert all(0.0 <= v <= 1.0 for v in pressures.values())
    assert pressures["sink"] == 1.0

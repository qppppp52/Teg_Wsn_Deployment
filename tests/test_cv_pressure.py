import pytest

from src.constraints.cv_pressure import (
    build_cv_pressure_snapshot,
    normalize_cv_components,
)
from src.model.solution import Solution
from tests.dqn_test_config import make_dqn_config


class Ctx:
    config = make_dqn_config()


def test_cv_pressure_components_are_clipped_to_unit_interval():
    solution = Solution(2)
    solution.cv_deploy = 0.5
    solution.cv_link = 4.0
    solution.cv_power = 2.0
    solution.cv_service = 8.0
    solution.cv_sink = 99.0
    solution.cv_energy = 0.0
    pressures = normalize_cv_components(solution, Ctx())
    assert set(pressures) == {"deploy", "link", "power", "service", "sink", "energy"}
    assert all(0.0 <= value <= 1.0 for value in pressures.values())
    assert pressures["sink"] == 1.0


def test_cv_pressure_rejects_missing_or_invalid_reference():
    solution = Solution(1)
    context = Ctx()
    context.config = make_dqn_config()
    del context.config["dqn"]["pressure_normalization"]["refs"]["power"]
    with pytest.raises(ValueError, match="must match CV schema"):
        build_cv_pressure_snapshot(solution, context)

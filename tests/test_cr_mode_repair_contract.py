from types import SimpleNamespace

import numpy as np
import pytest

from src.constraints.repair_config import resolve_repair_config
from src.constraints.report_freshness import require_fresh_constraint_report
from src.decoder.deployment_decoder import decode_deployment
from src.model.solution import Solution


def test_repair_config_separates_outer_and_energy_limits():
    cfg = {
        "constraints": {
            "max_outer_repair_rounds": 7,
            "max_energy_stabilization_iters": 3,
            "repair_patience": 2,
        }
    }
    resolved = resolve_repair_config(cfg)
    assert resolved.max_outer_repair_rounds == 7
    assert resolved.max_energy_stabilization_iters == 3


def test_legacy_repair_limit_is_rejected_when_nonpositive():
    with pytest.raises(ValueError, match="max_repair_iter"):
        resolve_repair_config({"constraints": {"max_repair_iter": 0}})


def test_deployment_decoder_scans_past_role_conflicts():
    individual = SimpleNamespace(
        rho_s=np.asarray([0.9, 0.8, 0.7]),
        rho_a=np.asarray([0.95, 0.85, 0.75]),
    )
    ctx = SimpleNamespace(
        num_candidates=3,
        index_mapping=SimpleNamespace(
            Ls_local_to_global=np.asarray([0, 1, 2]),
            La_local_to_global=np.asarray([0, 1, 2]),
        ),
        config={"deployment": {"max_sensors": 2, "max_aps": 2, "count_mode": "topk_up_to_max"}},
    )
    x, y = decode_deployment(individual, ctx)
    np.testing.assert_array_equal(x, [1, 1, 0])
    np.testing.assert_array_equal(y, [0, 0, 1])


def test_fresh_report_guard_rejects_stale_revision():
    solution = Solution(1)
    solution.state_revision = 2
    solution.constraint_report = SimpleNamespace(state_revision=1, feasible=True)
    with pytest.raises(ValueError, match="stale constraint report"):
        require_fresh_constraint_report(solution, "test")
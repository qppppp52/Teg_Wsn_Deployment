from types import SimpleNamespace

import numpy as np

import src.constraints.energy_constraints as energy_constraints
from src.model.solution import Solution


def _report(cv_total, sensor_deficit):
    return SimpleNamespace(
        feasible=False,
        cv_total=float(cv_total),
        max_component_cv=float(cv_total),
        violated_components=("energy",),
        physics=SimpleNamespace(
            energy=SimpleNamespace(
                sensor_deficit_violation=np.asarray(sensor_deficit, dtype=float)
            )
        ),
    )


def test_sensor_energy_fallback_uses_lower_pmin_ap_before_removal(monkeypatch):
    solution = Solution(3)
    solution.x[0] = 1
    solution.y[1] = 1
    solution.y[2] = 1
    solution.c[0, 1] = 1
    solution.p_tx[0, 1] = 0.20
    ctx = SimpleNamespace(
        num_candidates=3,
        ptx_min_matrix=np.asarray(
            [[0.0, 0.20, 0.05], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        ),
        link_feasible_matrix=np.ones((3, 3), dtype=np.int8),
        config={
            "channel": {"p_tx_max": 0.5},
            "constraints": {"max_sensor_energy_reassignments": 1},
        },
    )
    before = _report(2.0, [0.30, 0.0, 0.0])
    after = _report(1.0, [0.0, 0.0, 0.0])

    monkeypatch.setattr(
        energy_constraints,
        "ensure_constraint_spec",
        lambda _ctx: SimpleNamespace(cv_compare_tol=1.0e-8, energy_compare_tol=1.0e-8),
    )
    monkeypatch.setattr(
        energy_constraints,
        "stabilize_energy_sink_state",
        lambda candidate, _ctx: None,
    )
    monkeypatch.setattr(
        energy_constraints,
        "evaluate_constraints",
        lambda candidate, _ctx: after if candidate.c[0, 2] == 1 else before,
    )

    accepted, report = energy_constraints._reassign_sensor_energy_deficits(
        solution, ctx, before
    )

    assert accepted == 1
    assert report is after
    assert solution.c[0, 2] == 1
    assert solution.c[0, 1] == 0
    assert solution.p_tx[0, 2] == 0.05


def test_sensor_specific_repair_precedes_ap_rebalance(monkeypatch):
    solution = Solution(1)
    report = _report(0.0, [0.0])
    report.physics.energy.ap_deficit_violation = np.asarray([0.0])
    events = []

    monkeypatch.setattr(
        energy_constraints,
        "stabilize_energy_sink_state",
        lambda *_: events.append("stabilize"),
    )
    monkeypatch.setattr(
        energy_constraints,
        "evaluate_constraints",
        lambda *_: events.append("evaluate") or report,
    )
    monkeypatch.setattr(
        energy_constraints,
        "_reassign_sensor_energy_deficits",
        lambda _solution, _ctx, current: (events.append("sensor") or (0, current)),
    )
    monkeypatch.setattr(
        energy_constraints,
        "_rebalance_ap_energy",
        lambda *_: events.append("ap_rebalance"),
    )

    energy_constraints.repair_energy_constraints(
        solution, SimpleNamespace(config={})
    )

    assert events == [
        "stabilize",
        "evaluate",
        "sensor",
        "stabilize",
        "evaluate",
        "ap_rebalance",
        "stabilize",
        "evaluate",
    ]

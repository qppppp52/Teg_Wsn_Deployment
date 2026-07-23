from types import SimpleNamespace

import numpy as np

import src.constraints.repair_operator as repair_operator
import src.evaluator.repair_loop as repair_loop_module
from src.constraints.ap_service_constraints import repair_empty_aps
from src.constraints.constraint_report import evaluate_constraints
from src.constraints.heatsink_constraints import repair_sink_hard_constraints
from src.model.solution import Solution
from src.optimizers.generation_executor import environmental_select
from tests.test_actual_heatsink_harvest import _ctx


def test_baseline_repair_order_is_fixed_and_separate_from_strategy(monkeypatch):
    calls = []
    solution = SimpleNamespace(metadata={})
    for name in (
        "repair_deployment",
        "repair_link_constraints",
        "repair_sink_hard_constraints",
        "repair_power_legality",
        "repair_energy_constraints",
        "repair_empty_aps",
    ):
        monkeypatch.setattr(
            repair_operator,
            name,
            lambda solution, ctx, name=name: calls.append(name),
        )

    repair_operator.repair_solution(solution, object())

    assert calls == [
        "repair_deployment",
        "repair_link_constraints",
        "repair_sink_hard_constraints",
        "repair_power_legality",
        "repair_energy_constraints",
        "repair_empty_aps",
    ]
    assert solution.metadata["baseline_repair_order"] == list(
        repair_operator.BASELINE_REPAIR_ORDER
    )


def test_sink_hard_cleanup_removes_invalid_duplicate_and_contested_claims():
    ctx = _ctx()
    solution = Solution(ctx.num_candidates)
    solution.x[0] = 1
    solution.y[1] = 1
    solution.z_sink_sensor[0] = [0, 2, 2, 99]
    solution.z_sink_ap[1] = [2]

    repair_sink_hard_constraints(solution, ctx)

    assert solution.z_sink_sensor[0] == [0]
    assert solution.z_sink_ap[1] == []


def test_service_constraint_and_repair_ignore_declared_but_invalid_link():
    ctx = _ctx()
    ctx.link_feasible_matrix[0, 1] = 0
    solution = Solution(ctx.num_candidates)
    solution.x[0] = 1
    solution.y[1] = 1
    solution.c[0, 1] = 1

    report = evaluate_constraints(solution, ctx)
    assert report.components.service == 1.0

    repair_empty_aps(solution, ctx)
    assert solution.y[1] == 0
    assert not np.any(solution.c[:, 1])


def test_infeasible_constraint_ties_keep_stable_order_without_objective_ranking():
    solutions = []
    for coverage, rsum in ((0.1, 1.0), (0.5, 50.0), (0.9, 100.0)):
        solution = Solution(1)
        solution.feasible = False
        solution.cv = 1.0
        solution.coverage = coverage
        solution.rsum_capacity = rsum
        solution.constraint_report = SimpleNamespace(
            feasible=False,
            cv_total=1.0,
            max_component_cv=1.0,
            violated_components=("energy",),
            state_revision=0,
        )
        solutions.append(solution)

    individuals = [object(), object(), object()]
    selected_individuals, selected_solutions = environmental_select(
        individuals, solutions, 2
    )

    assert selected_individuals == individuals[:2]
    assert selected_solutions == solutions[:2]


def test_repair_loop_retains_best_constraint_state_and_defers_objectives(monkeypatch):
    solution = Solution(1)
    objective_calls = []
    stage = {"value": 0}

    def fake_evaluate_all(candidate, _ctx):
        score = {0: 5.0, 1: 1.0, 2: 3.0, 3: 4.0}[int(candidate.p_tx[0, 0])]
        candidate.cv = score
        candidate.feasible = False
        candidate.constraint_report = SimpleNamespace(
            feasible=False,
            cv_total=score,
            max_component_cv=score,
            violated_components=("energy",),
        )
        return score

    def fake_repair(candidate, _ctx):
        stage["value"] += 1
        candidate.p_tx[0, 0] = stage["value"]
        return candidate

    monkeypatch.setattr(repair_loop_module, "decode_solution", lambda *_: solution)
    monkeypatch.setattr(repair_loop_module, "evaluate_all_constraints", fake_evaluate_all)
    monkeypatch.setattr(repair_loop_module, "repair_solution", fake_repair)
    monkeypatch.setattr(repair_loop_module, "_deployment_changed", lambda *_: False)
    monkeypatch.setattr(
        repair_loop_module,
        "evaluate_objectives",
        lambda candidate, _ctx: objective_calls.append(candidate.cv),
    )
    ctx = SimpleNamespace(
        config={"constraints": {"repair_patience": 3, "cv_improvement_tol": 0.0}}
    )

    best, repaired = repair_loop_module.repair_loop(object(), ctx, max_iter=3)

    assert repaired is None
    assert best.cv == 1.0
    assert objective_calls == [1.0]

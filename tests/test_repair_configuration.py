from types import SimpleNamespace

import pytest

import src.evaluator.individual_evaluator as evaluator


def test_evaluate_individual_reads_configured_repair_limit(monkeypatch):
    captured = {}

    def fake_repair_loop(individual, ctx, max_iter, repair_strategy):
        captured["max_iter"] = max_iter
        return "solution", None

    monkeypatch.setattr(evaluator, "repair_loop", fake_repair_loop)
    ctx = SimpleNamespace(config={"constraints": {"max_repair_iter": 10}})

    assert evaluator.evaluate_individual(object(), ctx) == ("solution", None)
    assert captured["max_iter"] == 10


def test_explicit_repair_limit_overrides_context_config(monkeypatch):
    captured = {}

    def fake_repair_loop(individual, ctx, max_iter, repair_strategy):
        captured["max_iter"] = max_iter
        return "solution", None

    monkeypatch.setattr(evaluator, "repair_loop", fake_repair_loop)
    ctx = SimpleNamespace(config={"constraints": {"max_repair_iter": 10}})

    evaluator.evaluate_individual(object(), ctx, max_repair_iter=3)
    assert captured["max_iter"] == 3


def test_negative_configured_repair_limit_is_rejected():
    ctx = SimpleNamespace(config={"constraints": {"max_repair_iter": -1}})

    with pytest.raises(ValueError, match="max_repair_iter"):
        evaluator.configured_max_repair_iter(ctx)
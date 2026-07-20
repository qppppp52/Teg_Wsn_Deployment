"""One constrained-dominance contract shared by archive and optimizers."""
from __future__ import annotations

from src.constraints.constraint_report import compare_constraint_reports


def _report(solution):
    return getattr(solution, "constraint_report", None)


def compare_constraint_state(a, b, cv_compare_tol=1.0e-8) -> int:
    """Return 1 when a has the better constraint state, -1 for b, else 0."""
    report_a = _report(a)
    report_b = _report(b)
    feasible_a = bool(report_a.feasible) if report_a is not None else bool(getattr(a, "feasible", False))
    feasible_b = bool(report_b.feasible) if report_b is not None else bool(getattr(b, "feasible", False))
    if feasible_a != feasible_b:
        return 1 if feasible_a else -1
    if feasible_a:
        return 0
    if report_a is not None and report_b is not None:
        return compare_constraint_reports(report_a, report_b, cv_compare_tol)
    cv_a = float(getattr(a, "cv", 0.0))
    cv_b = float(getattr(b, "cv", 0.0))
    if cv_a < cv_b - cv_compare_tol:
        return 1
    if cv_b < cv_a - cv_compare_tol:
        return -1
    return 0


def constrained_dominates(a, b) -> bool:
    """Deb feasibility first, then report CV, then the two research objectives."""
    state = compare_constraint_state(a, b)
    if state != 0:
        return state > 0
    report_a = _report(a)
    feasible_a = bool(report_a.feasible) if report_a is not None else bool(getattr(a, "feasible", False))
    if not feasible_a:
        return False
    no_worse = float(a.coverage) >= float(b.coverage) and float(a.rsum_capacity) >= float(b.rsum_capacity)
    strictly_better = float(a.coverage) > float(b.coverage) or float(a.rsum_capacity) > float(b.rsum_capacity)
    return bool(no_worse and strictly_better)

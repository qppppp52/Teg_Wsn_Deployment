"""Comparator policy reserved for repair candidates and infeasible incumbents."""
from __future__ import annotations

from src.constraints.constraint_report import compare_constraint_reports


def is_repair_candidate_better(candidate_report, incumbent_report, tolerance=None) -> bool:
    """Return whether a repair candidate strictly improves the incumbent report."""
    kwargs = {} if tolerance is None else {"cv_compare_tol": float(tolerance)}
    return compare_constraint_reports(candidate_report, incumbent_report, **kwargs) > 0
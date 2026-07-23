"""State/revision checks at boundaries that consume constraint reports."""
from __future__ import annotations

from src.constraints.constraint_report import sync_state_revision


def require_fresh_constraint_report(solution, purpose: str = "operation"):
    """Return a report only when it still describes the authoritative solution."""
    current_revision = sync_state_revision(solution)
    report = getattr(solution, "constraint_report", None)
    if report is None:
        raise ValueError(
            f"{purpose} requires a freshly evaluated constraint report; "
            "evaluate the solution after its last physical edit"
        )
    if int(getattr(report, "state_revision", -1)) != current_revision:
        raise ValueError(
            f"{purpose} received a stale constraint report: "
            f"report revision {getattr(report, 'state_revision', None)} != "
            f"solution revision {current_revision}"
        )
    return report
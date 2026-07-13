"""Evaluation accounting for DRL initialization and online CR-MODE search."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class DRLInitEvaluationCounter:
    ppo_terminal_evaluations: int = 0
    drl_policy_rollout_terminal_evaluations: int = 0
    drl_candidate_post_evaluations: int = 0
    heuristic_candidate_evaluations: int = 0
    random_candidate_evaluations: int = 0
    gen0_re_evaluations: int = 0
    cr_mode_trial_evaluations: int = 0

    def increment(self, field: str, amount: int = 1) -> None:
        if not hasattr(self, field):
            raise KeyError(f"Unknown evaluation counter: {field}")
        setattr(self, field, int(getattr(self, field)) + int(amount))

    @property
    def online_evaluation_count(self) -> int:
        return self.gen0_re_evaluations + self.cr_mode_trial_evaluations

    @property
    def total_end_to_end_evaluations(self) -> int:
        return sum(asdict(self).values())

    def to_dict(self) -> dict:
        result = asdict(self)
        result["online_evaluation_count"] = self.online_evaluation_count
        result["total_end_to_end_evaluations"] = self.total_end_to_end_evaluations
        return result

"""Joint candidate-role action helpers for DRL initialization."""
from __future__ import annotations

ROLE_SENSOR = 0
ROLE_AP = 1
ROLE_SKIP = 2
ROLE_STOP = 3
ROLE_NAMES = ["sensor", "ap", "skip", "stop"]
NUM_ROLES = 4


def make_action(candidate_id: int, role: int) -> dict:
    return {"candidate_id": int(candidate_id), "role": int(role)}


def flatten_action(candidate_id: int, role: int) -> int:
    return int(candidate_id) * NUM_ROLES + int(role)


def unflatten_action(action_id: int) -> dict:
    action_id = int(action_id)
    return make_action(action_id // NUM_ROLES, action_id % NUM_ROLES)

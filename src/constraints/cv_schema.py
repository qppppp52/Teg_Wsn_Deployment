"""Stable top-level constraint-violation schema shared by DQN diagnostics."""
from __future__ import annotations

from typing import Final


# This is intentionally separate from the numeric constraint-report contract
# version. The latter governs feasibility evaluation; these names govern the
# semantic order used by learned-policy features and diagnostics.
CV_COMPONENT_KEYS: Final[tuple[str, ...]] = (
    "deploy",
    "link",
    "power",
    "service",
    "sink",
    "energy",
)

CV_COMPONENT_SCHEMA_VERSION: Final[str] = "cv6_top_level_v1"
PRESSURE_SCHEMA_VERSION: Final[str] = "dqn_cv6_fixed_reference_v1"
STATE_SCHEMA_VERSION: Final[str] = "dqn_state_cv6_pressure_v2"
REWARD_SCHEMA_VERSION: Final[str] = "dqn_reward_cv6_pressure_v3"
ACTION_MASK_SCHEMA_VERSION: Final[str] = "dqn_action_mask_cv6_pressure_v2"

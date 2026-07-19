"""Stable physical configuration subset used by learned-policy checkpoints."""
from __future__ import annotations

import copy


PHYSICAL_CONFIG_SECTIONS = (
    "space",
    "discretization",
    "deployment",
    "targets",
    "temperature",
    "teg",
    "sensor",
    "ap",
    "channel",
    "heatsink",
    "constraints",
    "throughput_enhancement",
    "objectives",
    "evaluation",
    "mode",
)


def physical_evaluation_contract(config: dict) -> dict:
    """Return config fields that can change repair, state, reward or objectives."""
    return {
        section: copy.deepcopy(config.get(section, {}))
        for section in PHYSICAL_CONFIG_SECTIONS
    }

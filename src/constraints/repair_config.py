"""Shared repair-loop configuration and legacy compatibility handling."""
from __future__ import annotations

from dataclasses import dataclass
import warnings


REPAIR_CONFIG_SEMANTICS_VERSION = 2
REPAIR_SEMANTICS_VERSION = 3
_legacy_warning_emitted = False


@dataclass(frozen=True)
class RepairConfig:
    max_outer_repair_rounds: int
    max_energy_stabilization_iters: int
    repair_patience: int
    max_sensor_energy_reassignments: int
    max_ap_rebalance_moves: int
    enable_repair_cycle_detection: bool
    enable_energy_node_deletion_fallback: bool


def resolve_repair_config(config: dict | None) -> RepairConfig:
    """Resolve the repair contract, accepting ``max_repair_iter`` temporarily."""
    global _legacy_warning_emitted
    constraints = (config or {}).get("constraints", {}) or {}
    legacy_limit = constraints.get("max_repair_iter")
    uses_legacy = legacy_limit is not None and (
        "max_outer_repair_rounds" not in constraints
        or "max_energy_stabilization_iters" not in constraints
    )
    if uses_legacy and not _legacy_warning_emitted:
        warnings.warn(
            "constraints.max_repair_iter is deprecated; set both "
            "max_outer_repair_rounds and max_energy_stabilization_iters.",
            DeprecationWarning,
            stacklevel=2,
        )
        _legacy_warning_emitted = True

    if legacy_limit is not None and _as_int(legacy_limit) < 1:
        raise ValueError("constraints.max_repair_iter must be at least 1")
    fallback = 5 if legacy_limit is None else legacy_limit
    return RepairConfig(
        max_outer_repair_rounds=_positive_int(
            constraints.get("max_outer_repair_rounds", fallback),
            "constraints.max_outer_repair_rounds",
        ),
        max_energy_stabilization_iters=_positive_int(
            constraints.get("max_energy_stabilization_iters", fallback),
            "constraints.max_energy_stabilization_iters",
        ),
        repair_patience=_positive_int(
            constraints.get("repair_patience", 2), "constraints.repair_patience"
        ),
        max_sensor_energy_reassignments=_non_negative_int(
            constraints.get("max_sensor_energy_reassignments", 1),
            "constraints.max_sensor_energy_reassignments",
        ),
        max_ap_rebalance_moves=_non_negative_int(
            constraints.get("max_ap_rebalance_moves", 0),
            "constraints.max_ap_rebalance_moves",
        ),
        enable_repair_cycle_detection=bool(
            constraints.get("enable_repair_cycle_detection", True)
        ),
        enable_energy_node_deletion_fallback=bool(
            constraints.get("enable_energy_node_deletion_fallback", False)
        ),
    )


def _positive_int(value, name: str) -> int:
    value = _non_negative_int(value, name)
    if value < 1:
        raise ValueError(f"{name} must be at least 1")
    return value


def _as_int(value) -> int:
    if isinstance(value, bool):
        raise ValueError("repair limits must be integers")
    try:
        integer = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("repair limits must be integers") from exc
    if integer != value:
        raise ValueError("repair limits must be integers")
    return integer


def _non_negative_int(value, name: str) -> int:
    try:
        integer = _as_int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if integer < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return integer
"""Fixed-reference CV pressure snapshots used only by DQN-CR-MODE.

Raw constraint reports and ``CVtotal`` remain the feasibility and ranking
authority. This module only gives the DQN a common, calibrated scale for the
six top-level components.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping

import numpy as np

from src.constraints.constraint_report import ConstraintReport
from src.constraints.cv_schema import CV_COMPONENT_KEYS, PRESSURE_SCHEMA_VERSION


@dataclass(frozen=True)
class PressureNormalization:
    """Validated fixed references for DQN pressure observations."""

    refs: dict[str, float]
    cv_zero_tol: float
    reference_source: str
    schema_version: str = PRESSURE_SCHEMA_VERSION


@dataclass(frozen=True)
class CVPressureSnapshot:
    raw_cv: dict[str, float]
    pressure: dict[str, float]
    violated: dict[str, bool]
    refs: dict[str, float]
    dominant_component: str | None
    max_pressure: float
    saturated_components: tuple[str, ...]
    schema_version: str = PRESSURE_SCHEMA_VERSION


@dataclass(frozen=True)
class PopulationPressureSummary:
    """One deterministic pressure summary for a complete population."""

    mean_raw_cv: dict[str, float]
    mean_pressure: dict[str, float]
    max_pressure: dict[str, float]
    violation_rate: dict[str, float]
    saturation_rate: dict[str, float]
    dominant_component: str | None
    dominant_pressure: float
    second_pressure: float
    dominant_margin: float
    sample_count: int
    schema_version: str = PRESSURE_SCHEMA_VERSION


def load_pressure_normalization(config: Mapping) -> PressureNormalization:
    """Load the explicit DQN-only pressure contract and reject ambiguous input."""
    dqn = _mapping(config, "dqn")
    raw = _mapping(dqn, "pressure_normalization")
    allowed = {"method", "schema_version", "cv_zero_tol", "refs", "reference_source"}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(
            "Unknown keys in dqn.pressure_normalization: " + ", ".join(unknown)
        )
    if raw.get("method") != "fixed_reference":
        raise ValueError("dqn.pressure_normalization.method must be 'fixed_reference'")
    schema_version = str(raw.get("schema_version", ""))
    if schema_version != PRESSURE_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported DQN pressure schema version: "
            f"{schema_version!r}; expected {PRESSURE_SCHEMA_VERSION!r}"
        )
    try:
        cv_zero_tol = float(raw["cv_zero_tol"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("dqn.pressure_normalization.cv_zero_tol is required") from exc
    if not math.isfinite(cv_zero_tol) or cv_zero_tol < 0.0:
        raise ValueError("dqn.pressure_normalization.cv_zero_tol must be finite and >= 0")

    refs_cfg = _mapping(raw, "refs", parent="dqn.pressure_normalization")
    expected = set(CV_COMPONENT_KEYS)
    missing = [key for key in CV_COMPONENT_KEYS if key not in refs_cfg]
    extra = sorted(set(refs_cfg) - expected)
    if missing or extra:
        details = []
        if missing:
            details.append("missing=" + ", ".join(missing))
        if extra:
            details.append("unknown=" + ", ".join(extra))
        raise ValueError(
            "DQN pressure references must match CV schema exactly ("
            + "; ".join(details)
            + ")"
        )
    refs: dict[str, float] = {}
    for key in CV_COMPONENT_KEYS:
        try:
            value = float(refs_cfg[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"DQN pressure reference {key!r} must be numeric") from exc
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"DQN pressure reference {key!r} must be finite and strictly positive"
            )
        refs[key] = value
    source = str(raw.get("reference_source", "unspecified")).strip()
    if not source:
        raise ValueError("dqn.pressure_normalization.reference_source must be non-empty")
    return PressureNormalization(refs=refs, cv_zero_tol=cv_zero_tol, reference_source=source)


def load_state_cv_total_ref(config: Mapping) -> float:
    """Return the separate explicit state scale for the global CVtotal feature."""
    dqn = _mapping(config, "dqn")
    state = _mapping(dqn, "state_normalization")
    try:
        value = float(state["cv_total_ref"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("dqn.state_normalization.cv_total_ref is required") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("dqn.state_normalization.cv_total_ref must be finite and > 0")
    return value


def extract_top_level_cv(solution) -> dict[str, float]:
    """Read one six-component raw CV vector from the authoritative report."""
    report = getattr(solution, "constraint_report", None)
    if isinstance(report, ConstraintReport):
        raw = report.components.as_dict()
    else:
        raw = {
            key: getattr(solution, f"cv_{key}", 0.0)
            for key in CV_COMPONENT_KEYS
        }
    return {key: _validate_cv_value(key, raw[key]) for key in CV_COMPONENT_KEYS}


def build_cv_pressure_snapshot(solution, ctx) -> CVPressureSnapshot:
    """Build one validated, clipped pressure snapshot for a solution."""
    normalization = load_pressure_normalization(_context_config(ctx))
    raw_cv = extract_top_level_cv(solution)
    pressure = {
        key: float(np.clip(raw_cv[key] / normalization.refs[key], 0.0, 1.0))
        for key in CV_COMPONENT_KEYS
    }
    violated = {
        key: bool(raw_cv[key] > normalization.cv_zero_tol)
        for key in CV_COMPONENT_KEYS
    }
    has_violation = any(violated.values())
    dominant = _ranked_keys(pressure)[0] if has_violation else None
    saturated = tuple(key for key in CV_COMPONENT_KEYS if pressure[key] >= 1.0)
    return CVPressureSnapshot(
        raw_cv=raw_cv,
        pressure=pressure,
        violated=violated,
        refs=dict(normalization.refs),
        dominant_component=dominant,
        max_pressure=float(max(pressure.values(), default=0.0)),
        saturated_components=saturated,
    )


def aggregate_population_pressure(
    solutions: Iterable,
    ctx,
) -> PopulationPressureSummary:
    """Aggregate snapshots once so DQN state, mask and reward agree per generation."""
    solution_list = list(solutions or [])
    if not solution_list:
        zeros = {key: 0.0 for key in CV_COMPONENT_KEYS}
        return PopulationPressureSummary(
            mean_raw_cv=zeros,
            mean_pressure=dict(zeros),
            max_pressure=dict(zeros),
            violation_rate=dict(zeros),
            saturation_rate=dict(zeros),
            dominant_component=None,
            dominant_pressure=0.0,
            second_pressure=0.0,
            dominant_margin=0.0,
            sample_count=0,
        )
    snapshots = [build_cv_pressure_snapshot(solution, ctx) for solution in solution_list]
    count = len(snapshots)
    mean_raw = {
        key: float(np.mean([snapshot.raw_cv[key] for snapshot in snapshots]))
        for key in CV_COMPONENT_KEYS
    }
    mean_pressure = {
        key: float(np.mean([snapshot.pressure[key] for snapshot in snapshots]))
        for key in CV_COMPONENT_KEYS
    }
    max_pressure = {
        key: float(max(snapshot.pressure[key] for snapshot in snapshots))
        for key in CV_COMPONENT_KEYS
    }
    violation_rate = {
        key: float(np.mean([snapshot.violated[key] for snapshot in snapshots]))
        for key in CV_COMPONENT_KEYS
    }
    saturation_rate = {
        key: float(np.mean([snapshot.pressure[key] >= 1.0 for snapshot in snapshots]))
        for key in CV_COMPONENT_KEYS
    }
    ranked = _ranked_keys(mean_pressure)
    dominant = ranked[0] if any(value > 0.0 for value in violation_rate.values()) else None
    dominant_value = float(mean_pressure[ranked[0]]) if ranked else 0.0
    second_value = float(mean_pressure[ranked[1]]) if len(ranked) > 1 else 0.0
    return PopulationPressureSummary(
        mean_raw_cv=mean_raw,
        mean_pressure=mean_pressure,
        max_pressure=max_pressure,
        violation_rate=violation_rate,
        saturation_rate=saturation_rate,
        dominant_component=dominant,
        dominant_pressure=dominant_value if dominant is not None else 0.0,
        second_pressure=second_value if dominant is not None else 0.0,
        dominant_margin=max(dominant_value - second_value, 0.0) if dominant is not None else 0.0,
        sample_count=count,
    )


def normalize_cv_components(solution, ctx) -> dict[str, float]:
    """Compatibility helper returning only the canonical six pressure values."""
    return dict(build_cv_pressure_snapshot(solution, ctx).pressure)


def _validate_cv_value(name: str, value) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} CV must be numeric") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{name} CV must be finite, got {numeric!r}")
    if numeric < 0.0:
        raise ValueError(f"{name} CV must be nonnegative, got {numeric!r}")
    return numeric


def _ranked_keys(values: Mapping[str, float]) -> list[str]:
    order = {key: index for index, key in enumerate(CV_COMPONENT_KEYS)}
    return sorted(CV_COMPONENT_KEYS, key=lambda key: (-float(values[key]), order[key]))


def _context_config(ctx) -> Mapping:
    config = getattr(ctx, "config", None)
    if not isinstance(config, Mapping):
        raise ValueError("DQN pressure normalization requires ctx.config")
    return config


def _mapping(mapping: Mapping, key: str, *, parent: str = "") -> Mapping:
    value = mapping.get(key) if isinstance(mapping, Mapping) else None
    if not isinstance(value, Mapping):
        path = f"{parent}.{key}" if parent else key
        raise ValueError(f"Configuration mapping {path!r} is required")
    return value

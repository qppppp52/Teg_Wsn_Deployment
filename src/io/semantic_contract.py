"""Versioned semantic contract attached to checkpoints and result artifacts."""
from __future__ import annotations

import hashlib
import json

from src.constraints.constraint_report import (
    AUTHORITATIVE_FIELDS_VERSION,
    COMPARATOR_VERSION,
    CONSTRAINT_SEMANTICS_VERSION,
    CV_SCHEMA_VERSION,
    ENERGY_TOLERANCE_VERSION,
)
from src.constraints.energy_constraints import ENERGY_REPAIR_SEMANTICS_VERSION
from src.constraints.repair_config import (
    REPAIR_CONFIG_SEMANTICS_VERSION,
    REPAIR_SEMANTICS_VERSION,
)
from src.constraints.link_constraints import LINK_REPAIR_SEMANTICS_VERSION
from src.decoder.deployment_decoder import DEPLOYMENT_DECODER_SEMANTICS_VERSION
from src.model.pareto_archive import PARETO_ARCHIVE_SEMANTICS_VERSION
from src.power.power_repair import POWER_REPAIR_SEMANTICS_VERSION


def build_semantic_contract(config: dict | None) -> dict:
    constraints = ((config or {}).get("constraints", {}) or {})
    return {
        "cv_schema_version": int(constraints.get("cv_schema_version", CV_SCHEMA_VERSION)),
        "constraint_semantics_version": int(
            constraints.get("semantics_version", CONSTRAINT_SEMANTICS_VERSION)
        ),
        "comparator_version": int(constraints.get("comparator_version", COMPARATOR_VERSION)),
        "energy_tolerance_version": int(
            constraints.get("energy_tolerance_version", ENERGY_TOLERANCE_VERSION)
        ),
        "authoritative_fields_version": int(
            constraints.get("authoritative_fields_version", AUTHORITATIVE_FIELDS_VERSION)
        ),
        "deployment_decoder_semantics_version": DEPLOYMENT_DECODER_SEMANTICS_VERSION,
        "link_repair_semantics_version": LINK_REPAIR_SEMANTICS_VERSION,
        "power_repair_semantics_version": POWER_REPAIR_SEMANTICS_VERSION,
        "energy_repair_semantics_version": ENERGY_REPAIR_SEMANTICS_VERSION,
        "repair_config_semantics_version": REPAIR_CONFIG_SEMANTICS_VERSION,
        "repair_semantics_version": REPAIR_SEMANTICS_VERSION,
        "pareto_archive_semantics_version": PARETO_ARCHIVE_SEMANTICS_VERSION,
        "deployment_count_mode": (config or {}).get("deployment", {}).get(
            "count_mode", "topk_up_to_max"
        ),
        "enable_energy_node_deletion_fallback": bool(
            constraints.get("enable_energy_node_deletion_fallback", False)
        ),
    }


def semantic_signature(config: dict | None) -> str:
    payload = json.dumps(build_semantic_contract(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
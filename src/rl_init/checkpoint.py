"""Checkpoint helpers for PPO initialization."""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from typing import Any

from src.rl_init.ppo_policy import torch
from src.decoder.power_decoder import INITIAL_POWER_SEMANTICS_VERSION
from src.heatsink.sink_ownership import SINK_OWNERSHIP_SEMANTICS_VERSION
from src.power.power_repair import POWER_REPAIR_SEMANTICS_VERSION
from src.power.throughput_enhancer import THROUGHPUT_ENHANCER_VERSION
from src.physics.evaluation_contract import physical_evaluation_contract

COMPATIBILITY_KEYS = [
    "scene_name",
    "num_candidates",
    "candidate_feature_dim",
    "global_feature_dim",
    "hidden_dim",
    "action_roles",
    "max_selected_sensors",
    "max_selected_aps",
    "config_hash",
]


def save_policy(policy, path):
    if torch is None or policy is None:
        return ""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(policy.state_dict(), path)
    return path


def load_policy(policy, path, map_location=None):
    if torch is None or not path or not os.path.exists(path):
        return False
    policy.load_state_dict(torch.load(path, map_location=map_location or "cpu"))
    return True


def scene_name(config: dict) -> str:
    exp_name = config.get("experiment", {}).get("name")
    if exp_name:
        return _slug(exp_name)
    temp_model = config.get("temperature", {}).get("model", "scene")
    targets = config.get("targets", {}).get("num_targets", "targets")
    spacing = config.get("discretization", {}).get("grid_spacing", "grid")
    return _slug(f"{temp_model}_{targets}_{spacing}")


def resolve_checkpoint_path(config: dict, seed: int | None = None) -> str:
    cfg = config.get("drl_init", config)
    mode = str(cfg.get("checkpoint_mode", "per_seed"))
    checkpoint_dir = cfg.get("checkpoint_dir", "experiments/checkpoints")
    values = {
        "scene_name": scene_name(config),
        "seed": int(seed if seed is not None else cfg.get("seed", 42)),
    }
    if mode == "shared_pretrain":
        template = cfg.get("checkpoint_name_template_shared", "drl_init_{scene_name}_rsum_capacity_shared.pt")
    else:
        template = cfg.get("checkpoint_name_template", "drl_init_{scene_name}_rsum_capacity_seed_{seed}.pt")
    return os.path.join(checkpoint_dir, template.format(**values))

def stable_config_hash(config: dict) -> str:
    cfg = config.get("drl_init", config)
    net = cfg.get("network", {})
    relevant = {
        "scene_name": scene_name(config),
        "space": config.get("space", {}),
        "discretization": config.get("discretization", {}),
        "targets": config.get("targets", {}),
        "temperature": config.get("temperature", {}),
        "role_actions": cfg.get("role_actions", ["sensor", "ap", "skip", "stop"]),
        "max_selected_sensors": cfg.get("max_selected_sensors"),
        "max_selected_aps": cfg.get("max_selected_aps"),
        "reward": cfg.get("reward", {}),
        "normalization": cfg.get("normalization", {}),
        "evaluation_semantics": {
            "environment": physical_evaluation_contract(config),
            "throughput_enhancer_version": THROUGHPUT_ENHANCER_VERSION,
            "heatsink_ownership_semantics_version": SINK_OWNERSHIP_SEMANTICS_VERSION,
            "initial_power_semantics_version": INITIAL_POWER_SEMANTICS_VERSION,
            "power_repair_semantics_version": POWER_REPAIR_SEMANTICS_VERSION,
        },
        "network": {
            "candidate_feature_dim": net.get("candidate_feature_dim", "auto"),
            "global_feature_dim": net.get("global_feature_dim", "auto"),
            "hidden_dim": net.get("hidden_dim", 128),
            "attention_dim": net.get("attention_dim", 128),
            "dropout": net.get("dropout", 0.1),
            "use_attention": net.get("use_attention", True),
            "use_candidate_encoder": net.get("use_candidate_encoder", True),
        },
    }
    payload = json.dumps(relevant, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def build_checkpoint_meta(agent, ctx, config: dict, checkpoint_mode: str | None = None) -> dict:
    cfg = config.get("drl_init", config)
    net_cfg = cfg.get("network", {})
    policy = getattr(agent, "policy", None)
    candidate_dim = _first_linear_in_features(getattr(getattr(policy, "encoder", None), "candidate_mlp", None))
    global_dim = _first_linear_in_features(getattr(getattr(policy, "encoder", None), "global_mlp", None))
    hidden_dim = int(getattr(getattr(policy, "actor", None), "in_features", net_cfg.get("hidden_dim", 128)))
    return {
        "scene_name": scene_name(config),
        "seed": int(cfg.get("seed", config.get("experiment", {}).get("seeds", [42])[0])),
        "checkpoint_mode": checkpoint_mode or cfg.get("checkpoint_mode", "per_seed"),
        "num_candidates": int(getattr(ctx, "num_candidates", 0)),
        "candidate_feature_dim": int(candidate_dim or net_cfg.get("candidate_feature_dim", 0) or 0),
        "global_feature_dim": int(global_dim or net_cfg.get("global_feature_dim", 0) or 0),
        "hidden_dim": hidden_dim,
        "action_roles": list(cfg.get("role_actions", ["sensor", "ap", "skip", "stop"])),
        "max_selected_sensors": int(cfg.get("max_selected_sensors", config.get("deployment", {}).get("max_sensors", 0))),
        "max_selected_aps": int(cfg.get("max_selected_aps", config.get("deployment", {}).get("max_aps", 0))),
        "config_hash": stable_config_hash(config),
        "normalization": cfg.get("normalization", {}),
        "evaluation_semantics": {
            "environment": physical_evaluation_contract(config),
            "throughput_enhancer_version": THROUGHPUT_ENHANCER_VERSION,
            "heatsink_ownership_semantics_version": SINK_OWNERSHIP_SEMANTICS_VERSION,
            "initial_power_semantics_version": INITIAL_POWER_SEMANTICS_VERSION,
            "power_repair_semantics_version": POWER_REPAIR_SEMANTICS_VERSION,
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_branch": _git_branch(),
    }


def check_checkpoint_compatibility(meta: dict | None, expected_meta: dict | None) -> tuple[bool, str]:
    if not expected_meta:
        return True, ""
    if not meta:
        return False, "missing_meta"
    for key in COMPATIBILITY_KEYS:
        if meta.get(key) != expected_meta.get(key):
            return False, f"{key}_mismatch"
    return True, ""


def save_checkpoint(agent, path, meta: dict | None = None):
    if torch is None or agent is None:
        return ""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "policy_state_dict": agent.policy.state_dict(),
        "optimizer_state_dict": agent.optimizer.state_dict() if getattr(agent, "optimizer", None) is not None else None,
        "meta": meta or {},
    }, path)
    return path


def load_checkpoint(agent, path, map_location=None, expected_meta: dict | None = None):
    result = {
        "loaded": False,
        "meta": {},
        "compatible": False,
        "skip_reason": "missing_checkpoint",
        "error": "",
    }
    if torch is None:
        result["skip_reason"] = "torch_unavailable"
        return result
    if agent is None or not path or not os.path.exists(path):
        return result
    try:
        checkpoint = torch.load(path, map_location=map_location or agent.device)
        meta = checkpoint.get("meta", {}) if isinstance(checkpoint, dict) else {}
        compatible, reason = check_checkpoint_compatibility(meta, expected_meta)
        result.update({"meta": meta, "compatible": compatible, "skip_reason": reason})
        if not compatible:
            return result
        if isinstance(checkpoint, dict) and "policy_state_dict" in checkpoint:
            agent.policy.load_state_dict(checkpoint["policy_state_dict"])
            optimizer_state = checkpoint.get("optimizer_state_dict")
            if optimizer_state is not None and getattr(agent, "optimizer", None) is not None:
                agent.optimizer.load_state_dict(optimizer_state)
        elif isinstance(checkpoint, dict) and "policy" in checkpoint:
            agent.policy.load_state_dict(checkpoint["policy"])
            if "optimizer" in checkpoint and getattr(agent, "optimizer", None) is not None:
                agent.optimizer.load_state_dict(checkpoint["optimizer"])
        else:
            agent.policy.load_state_dict(checkpoint)
        result.update({"loaded": True, "compatible": True, "skip_reason": ""})
    except Exception as exc:  # pragma: no cover - exact torch errors vary by version
        result.update({"loaded": False, "compatible": False, "skip_reason": "load_error", "error": str(exc)})
    return result


def _first_linear_in_features(module) -> int | None:
    if module is None or torch is None:
        return None
    for layer in module:
        if hasattr(layer, "in_features"):
            return int(layer.in_features)
    return None


def _slug(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "_", str(value).strip().lower()).strip("_")
    return value or "scene"


def _git_branch() -> str:
    try:
        return subprocess.check_output(["git", "branch", "--show-current"], text=True, stderr=subprocess.DEVNULL).strip() or "unknown"
    except Exception:
        return "unknown"

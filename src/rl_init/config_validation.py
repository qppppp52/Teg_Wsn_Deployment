"""Strict validation for the formal DRL initialization configuration."""
from __future__ import annotations


TOP_LEVEL_KEYS = {
    "enabled", "algorithm", "mode", "seed", "device", "train_episodes",
    "max_steps_per_episode", "rollout_steps", "update_epochs", "minibatch_size",
    "gamma", "gae_lambda", "clip_epsilon", "entropy_coef", "value_coef",
    "learning_rate", "max_grad_norm", "population_size", "drl_ratio",
    "heuristic_ratio", "random_ratio", "max_selected_sensors", "max_selected_aps",
    "min_selected_sensors", "min_selected_aps", "allow_stop_action", "action_mode",
    "role_actions", "reward", "normalization", "network",
    "diversity", "priority_noise_std", "save_checkpoint", "load_checkpoint_if_exists",
    "force_retrain", "checkpoint_mode", "checkpoint_dir", "checkpoint_name_template",
    "checkpoint_name_template_shared", "save_training_log", "save_generated_population",
    "require_drl_policy", "min_accepted_drl_ratio", "invalid_run_policy",
    "training_summary_window", "log_interval_episodes",
}

REWARD_KEYS = {
    "coverage", "rsum_capacity", "feasible_bonus", "cv_penalty", "repair_cost_penalty",
    "diversity_bonus", "invalid_action_penalty", "too_few_nodes_penalty",
    "duplicate_penalty", "coverage_step", "energy_step", "link_step", "sink_risk",
}
NORMALIZATION_KEYS = {"rsum_ref_min", "rsum_ref_max", "cv_ref", "repair_iter_ref"}
NETWORK_KEYS = {"hidden_dim", "dropout"}
DIVERSITY_KEYS = {
    "min_hamming_distance", "weak_hamming_distance", "min_priority_l2_distance",
    "min_objective_distance", "max_duplicate_retry", "max_drl_candidate_evaluations",
}

DEPRECATED_KEYS = {
    "allow_pure_drl", "evaluate_during_training", "eval_interval_steps",
    "terminal_full_evaluation", "use_repair_loop_for_reward",
}


def validate_drl_init_config(config: dict) -> dict:
    cfg = config.get("drl_init", config)
    if not isinstance(cfg, dict):
        raise ValueError("drl_init must be a mapping")
    deprecated = sorted(DEPRECATED_KEYS.intersection(cfg))
    if deprecated:
        raise ValueError("Deprecated DRL-init keys must be removed: " + ", ".join(deprecated))
    _reject_unknown(cfg, TOP_LEVEL_KEYS, "drl_init")
    _reject_unknown(cfg.get("reward", {}), REWARD_KEYS, "drl_init.reward")
    _reject_unknown(cfg.get("normalization", {}), NORMALIZATION_KEYS, "drl_init.normalization")
    _reject_unknown(cfg.get("network", {}), NETWORK_KEYS, "drl_init.network")
    _reject_unknown(cfg.get("diversity", {}), DIVERSITY_KEYS, "drl_init.diversity")

    ratios = [
        float(cfg.get("drl_ratio", 0.6)),
        float(cfg.get("heuristic_ratio", 0.2)),
        float(cfg.get("random_ratio", 0.2)),
    ]
    if any(value < 0.0 or value > 1.0 for value in ratios):
        raise ValueError("DRL-init population ratios must be in [0, 1]")
    if abs(sum(ratios) - 1.0) > 1.0e-9:
        raise ValueError("drl_ratio + heuristic_ratio + random_ratio must equal 1.0")

    for key in ("train_episodes", "max_steps_per_episode", "rollout_steps", "update_epochs", "minibatch_size"):
        if int(cfg.get(key, 1)) <= 0:
            raise ValueError(f"drl_init.{key} must be positive")
    for key in ("max_selected_sensors", "max_selected_aps", "min_selected_sensors", "min_selected_aps"):
        if int(cfg.get(key, 1)) <= 0:
            raise ValueError(f"drl_init.{key} must be positive")

    norm = cfg.setdefault("normalization", {})
    norm["rsum_ref_min"] = float(norm.get("rsum_ref_min", 0.0))
    norm["rsum_ref_max"] = float(norm.get("rsum_ref_max", 1.0))
    norm["cv_ref"] = float(norm.get("cv_ref", 10.0))
    norm["repair_iter_ref"] = float(norm.get("repair_iter_ref", 5.0))
    if norm["rsum_ref_max"] <= norm["rsum_ref_min"]:
        raise ValueError("drl_init.normalization.rsum_ref_max must exceed rsum_ref_min")

    min_ratio = float(cfg.get("min_accepted_drl_ratio", 0.5))
    if not 0.0 <= min_ratio <= 1.0:
        raise ValueError("drl_init.min_accepted_drl_ratio must be in [0, 1]")
    policy = str(cfg.get("invalid_run_policy", "fail"))
    if policy not in {"fail", "mark_and_continue"}:
        raise ValueError("drl_init.invalid_run_policy must be fail or mark_and_continue")
    if str(cfg.get("checkpoint_mode", "per_seed")) not in {"per_seed", "shared_pretrain"}:
        raise ValueError("drl_init.checkpoint_mode must be per_seed or shared_pretrain")
    return cfg


def _reject_unknown(mapping: dict, allowed: set[str], path: str) -> None:
    if not isinstance(mapping, dict):
        raise ValueError(f"{path} must be a mapping")
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ValueError(f"Unknown keys in {path}: {', '.join(unknown)}")

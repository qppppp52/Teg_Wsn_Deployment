"""Typed and validated DQN configuration."""
from __future__ import annotations

from dataclasses import dataclass


_LEGACY_KEYS = {
    "hidden_dims", "gamma", "lr", "batch_size", "replay_capacity",
    "min_replay_size", "target_update_interval", "epsilon_start",
    "epsilon_end", "epsilon_decay_generations", "train_every",
    "reward_clip", "reward_normalization", "action_mask_enabled",
}


@dataclass(frozen=True)
class DQNConfig:
    mode: str
    checkpoint_path: str
    hidden_dims: tuple[int, ...]
    gamma: float
    learning_rate: float
    batch_size: int
    grad_clip_norm: float
    replay_capacity: int
    min_replay_size: int
    target_update_interval: int
    epsilon_start: float
    epsilon_end: float
    epsilon_decay_steps: int
    updates_per_step: int
    reward_clip: tuple[float, float]
    reward_normalization: bool
    reward_warmup_steps: int
    state_normalization: bool
    state_warmup_steps: int
    state_clip: float
    action_mask_enabled: bool
    mask_thresholds: dict
    save_model: bool
    training_episodes: int
    validation_interval: int
    validation_patience: int

    @classmethod
    def from_mapping(cls, config: dict) -> "DQNConfig":
        if not isinstance(config, dict) or not isinstance(config.get("dqn"), dict):
            raise ValueError("Configuration must contain a 'dqn' mapping")
        raw = config["dqn"]
        legacy = sorted(_LEGACY_KEYS.intersection(raw))
        if legacy:
            raise ValueError(
                "Legacy flat DQN keys are not supported: " + ", ".join(legacy)
            )
        allowed = {
            "enabled", "mode", "checkpoint_path", "network", "optimizer",
            "replay", "target_update", "exploration", "training", "reward",
            "state_normalization", "action_mask", "checkpoint",
        }
        _reject_unknown(raw, allowed, "dqn")
        required_sections = [
            "network", "optimizer", "replay", "target_update", "exploration",
            "training", "reward", "state_normalization", "action_mask",
        ]
        missing = [name for name in required_sections if not isinstance(raw.get(name), dict)]
        if missing:
            raise ValueError("Missing DQN configuration sections: " + ", ".join(missing))

        network = raw["network"]
        optimizer = raw["optimizer"]
        replay = raw["replay"]
        target = raw["target_update"]
        exploration = raw["exploration"]
        training = raw["training"]
        reward = raw["reward"]
        state = raw["state_normalization"]
        action_mask = raw["action_mask"]
        checkpoint = raw.get("checkpoint", {})
        _reject_unknown(network, {"hidden_dims"}, "dqn.network")
        _reject_unknown(
            optimizer,
            {"learning_rate", "gamma", "batch_size", "grad_clip_norm"},
            "dqn.optimizer",
        )
        _reject_unknown(replay, {"capacity", "min_size"}, "dqn.replay")
        _reject_unknown(target, {"interval"}, "dqn.target_update")
        _reject_unknown(
            exploration,
            {"epsilon_start", "epsilon_end", "decay_steps"},
            "dqn.exploration",
        )
        _reject_unknown(
            training,
            {
                "updates_per_step", "episodes", "validation_interval",
                "validation_patience",
            },
            "dqn.training",
        )
        _reject_unknown(reward, {"clip", "normalize", "warmup_steps"}, "dqn.reward")
        _reject_unknown(
            state,
            {"enabled", "warmup_steps", "clip"},
            "dqn.state_normalization",
        )
        _reject_unknown(action_mask, {"enabled", "thresholds"}, "dqn.action_mask")
        _reject_unknown(checkpoint, {"save_model"}, "dqn.checkpoint")

        mode = str(raw.get("mode", "")).lower()
        if mode not in {"train", "eval"}:
            raise ValueError("dqn.mode must be either 'train' or 'eval'")
        clip = tuple(float(value) for value in reward.get("clip", [-1.0, 1.0]))
        hidden = tuple(int(value) for value in network.get("hidden_dims", []))
        if not hidden or len(clip) != 2:
            raise ValueError("DQN hidden_dims must be non-empty and reward.clip must have two values")
        result = cls(
            mode=mode,
            checkpoint_path=str(raw.get("checkpoint_path", "")),
            hidden_dims=hidden,
            gamma=float(optimizer.get("gamma", 0.95)),
            learning_rate=float(optimizer.get("learning_rate", 1.0e-3)),
            batch_size=int(optimizer.get("batch_size", 64)),
            grad_clip_norm=float(optimizer.get("grad_clip_norm", 5.0)),
            replay_capacity=int(replay.get("capacity", 5000)),
            min_replay_size=int(replay.get("min_size", 64)),
            target_update_interval=int(target.get("interval", 100)),
            epsilon_start=float(exploration.get("epsilon_start", 1.0)),
            epsilon_end=float(exploration.get("epsilon_end", 0.05)),
            epsilon_decay_steps=int(exploration.get("decay_steps", 5000)),
            updates_per_step=int(training.get("updates_per_step", 1)),
            reward_clip=clip,
            reward_normalization=bool(reward.get("normalize", True)),
            reward_warmup_steps=int(reward.get("warmup_steps", 64)),
            state_normalization=bool(state.get("enabled", True)),
            state_warmup_steps=int(state.get("warmup_steps", 64)),
            state_clip=float(state.get("clip", 5.0)),
            action_mask_enabled=bool(action_mask.get("enabled", True)),
            mask_thresholds=dict(action_mask.get("thresholds", {})),
            save_model=bool(checkpoint.get("save_model", True)),
            training_episodes=int(training.get("episodes", 120)),
            validation_interval=int(training.get("validation_interval", 5)),
            validation_patience=int(training.get("validation_patience", 20)),
        )
        result.validate()
        return result

    def validate(self):
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("dqn.optimizer.gamma must be in [0, 1]")
        if self.batch_size <= 0 or self.min_replay_size < self.batch_size:
            raise ValueError("dqn.replay.min_size must be >= optimizer.batch_size > 0")
        if self.replay_capacity < self.min_replay_size:
            raise ValueError("dqn.replay.capacity must be >= replay.min_size")
        if self.target_update_interval <= 0 or self.epsilon_decay_steps <= 0:
            raise ValueError("Target update interval and epsilon decay steps must be positive")
        if self.reward_clip[0] >= self.reward_clip[1]:
            raise ValueError("dqn.reward.clip lower bound must be smaller than upper bound")


def _reject_unknown(mapping, allowed, path):
    unknown = sorted(set(mapping) - set(allowed))
    if unknown:
        raise ValueError(f"Unknown keys in {path}: {', '.join(unknown)}")

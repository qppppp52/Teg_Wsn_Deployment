def make_dqn_config(hidden_dims=(8,), batch_size=2, epsilon=(0.0, 0.0)):
    return {
        "dqn": {
            "mode": "train",
            "checkpoint_path": "",
            "network": {"hidden_dims": list(hidden_dims)},
            "optimizer": {
                "learning_rate": 0.001,
                "gamma": 0.95,
                "batch_size": batch_size,
                "grad_clip_norm": 5.0,
            },
            "replay": {"capacity": 100, "min_size": batch_size},
            "target_update": {"interval": 2},
            "exploration": {
                "epsilon_start": epsilon[0],
                "epsilon_end": epsilon[1],
                "decay_steps": 10,
            },
            "training": {
                "updates_per_step": 1,
                "episodes": 2,
                "validation_interval": 1,
                "validation_patience": 2,
            },
            "reward": {
                "clip": [-1.0, 1.0],
                "normalize": True,
                "warmup_steps": 2,
            },
            "state_normalization": {
                "enabled": True,
                "warmup_steps": 2,
                "clip": 5.0,
            },
            "action_mask": {"enabled": True, "thresholds": {}},
            "checkpoint": {"save_model": True},
        }
    }

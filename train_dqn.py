"""Train the DQN controller before formal algorithm comparison."""
from __future__ import annotations

import argparse
import copy

from src.dqn.dqn_trainer import DQNTrainer
from src.io.config_reader import load_experiment_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/experiment_small_compare.yaml",
    )
    parser.add_argument("--episodes", type=int, default=None)
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    config = copy.deepcopy(config)
    config.setdefault("dqn", {})["mode"] = "train"
    if args.episodes is not None:
        if args.episodes <= 0:
            raise ValueError("--episodes must be positive")
        config["dqn"]["training"]["episodes"] = args.episodes
        config.setdefault("experiment", {})["dqn_training_seeds"] = list(
            range(1000, 1000 + args.episodes)
        )
    path = DQNTrainer(config).train()
    print(f"Best DQN checkpoint: {path}")


if __name__ == "__main__":
    main()

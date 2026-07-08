"""Checkpoint helpers for PPO initialization."""
from __future__ import annotations

import os
from src.rl_init.ppo_policy import torch


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


def save_checkpoint(agent, path):
    if torch is None or agent is None:
        return ""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "policy": agent.policy.state_dict(),
        "optimizer": agent.optimizer.state_dict(),
    }, path)
    return path


def load_checkpoint(agent, path, map_location=None):
    if torch is None or agent is None or not path or not os.path.exists(path):
        return False
    checkpoint = torch.load(path, map_location=map_location or agent.device)
    if "policy" in checkpoint:
        agent.policy.load_state_dict(checkpoint["policy"])
        if "optimizer" in checkpoint:
            agent.optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        agent.policy.load_state_dict(checkpoint)
    return True

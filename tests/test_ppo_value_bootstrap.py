import torch

from src.rl_init.ppo_agent import PPOAgent


class _ValuePolicy:
    def __init__(self):
        self.grad_enabled = None

    def __call__(self, state):
        self.grad_enabled = torch.is_grad_enabled()
        return {"value": torch.tensor([2.5], requires_grad=True)}


def test_value_bootstrap_runs_without_gradient_tracking():
    agent = PPOAgent.__new__(PPOAgent)
    agent.policy = _ValuePolicy()

    value = agent.value({"state": "partial_rollout"})

    assert value == 2.5
    assert agent.policy.grad_enabled is False

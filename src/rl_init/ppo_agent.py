"""PPO agent for DRL initialization."""
from __future__ import annotations

import numpy as np
from src.rl_init.action_space import flatten_action
from src.rl_init.ppo_policy import InitActorCritic, torch
from src.rl_init.rollout_buffer import RolloutBuffer


class PPOAgent:
    def __init__(self, state_sample: dict, config: dict):
        if torch is None:
            raise ImportError("PyTorch is required for PPOAgent")
        cfg = config.get("drl_init", config)
        net_cfg = cfg.get("network", {})
        device_name = cfg.get("device", "auto")
        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_name)
        self.policy = InitActorCritic(
            candidate_feature_dim=state_sample["candidate_features"].shape[1],
            global_feature_dim=state_sample["global_features"].shape[0],
            hidden_dim=int(net_cfg.get("hidden_dim", 128)),
            dropout=float(net_cfg.get("dropout", 0.1)),
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=float(cfg.get("learning_rate", 3.0e-4)))
        self.clip_epsilon = float(cfg.get("clip_epsilon", 0.2))
        self.value_coef = float(cfg.get("value_coef", 0.5))
        self.entropy_coef = float(cfg.get("entropy_coef", 0.01))
        self.max_grad_norm = float(cfg.get("max_grad_norm", 0.5))
        self.update_epochs = int(cfg.get("update_epochs", 8))
        self.minibatch_size = int(cfg.get("minibatch_size", 128))
        self.gamma = float(cfg.get("gamma", 0.98))
        self.gae_lambda = float(cfg.get("gae_lambda", 0.95))

    def act(self, state, deterministic=False):
        with torch.no_grad():
            action, log_prob, value, entropy = self.policy.act(state, deterministic=deterministic)
        return action, float(log_prob.item()), float(value.item()), float(entropy.item())

    def update(self, buffer: RolloutBuffer):
        if len(buffer) == 0:
            return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        if not buffer.advantages or not buffer.returns:
            buffer.compute_returns_and_advantages(0.0, self.gamma, self.gae_lambda)
        advantages = torch.as_tensor(buffer.advantages, dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(buffer.returns, dtype=torch.float32, device=self.device)
        old_log_probs = torch.as_tensor(buffer.log_probs, dtype=torch.float32, device=self.device)
        actions = np.asarray(buffer.actions, dtype=int)
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1.0e-8)

        n = len(buffer)
        last_stats = {}
        for _ in range(self.update_epochs):
            order = np.random.permutation(n)
            for start in range(0, n, max(self.minibatch_size, 1)):
                idx = order[start:start + max(self.minibatch_size, 1)]
                states = [buffer.states[i] for i in idx]
                batch_actions = actions[idx]
                new_log_probs, values, entropy = self.policy.evaluate_actions(states, batch_actions)
                ratio = torch.exp(new_log_probs - old_log_probs[idx])
                adv = advantages[idx]
                unclipped = ratio * adv
                clipped = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * adv
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = torch.nn.functional.mse_loss(values, returns[idx])
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                last_stats = {
                    "loss": float(loss.item()),
                    "policy_loss": float(policy_loss.item()),
                    "value_loss": float(value_loss.item()),
                    "entropy": float(entropy.item()),
                }
        return last_stats or {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}


def action_to_id(action: dict) -> int:
    return flatten_action(int(action["candidate_id"]), int(action["role"]))

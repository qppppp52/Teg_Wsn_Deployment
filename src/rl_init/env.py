"""Sequential deployment initialization environment."""
from __future__ import annotations

import numpy as np
from src.rl_init.action_space import ROLE_SENSOR, ROLE_AP, ROLE_SKIP, ROLE_STOP
from src.rl_init.action_mask import build_joint_action_mask
from src.rl_init.individual_builder import build_individual_from_orders
from src.rl_init.reward import compute_step_reward, compute_terminal_reward
from src.rl_init.state_builder import build_init_state


class InitDeploymentEnv:
    def __init__(self, ctx, config):
        self.ctx = ctx
        self.config = config
        self.cfg = config.get("drl_init", config)
        self.rng = np.random.default_rng(int(self.cfg.get("seed", 42)))
        self.selected_sensors = set()
        self.selected_aps = set()
        self.skipped_candidates = set()
        self.selection_order_sensors = []
        self.selection_order_aps = []
        self.step_count = 0
        self.done = False
        self.last_eval_metrics = {}
        self.invalid_action_count = 0
        self.step_reward_sum = 0.0
        self._last_summary = {}

    def reset(self, seed=None) -> dict:
        if seed is not None:
            self.rng = np.random.default_rng(int(seed))
        self.selected_sensors = set()
        self.selected_aps = set()
        self.skipped_candidates = set()
        self.selection_order_sensors = []
        self.selection_order_aps = []
        self.step_count = 0
        self.done = False
        self.last_eval_metrics = {}
        self.invalid_action_count = 0
        self.step_reward_sum = 0.0
        self._last_summary = self.render_state_summary()
        return build_init_state(self)

    def step(self, action: dict) -> tuple[dict, float, bool, dict]:
        if self.done:
            return build_init_state(self), 0.0, True, {"already_done": True}
        info = {"invalid_action": False, "duplicate_or_conflict": False}
        prev_summary = self.render_state_summary()
        role = int(action.get("role", ROLE_STOP))
        gid = int(action.get("candidate_id", 0))
        valid = self._is_valid_action(gid, role, info)
        if valid:
            if role == ROLE_SENSOR:
                self.selected_sensors.add(gid)
                self.selection_order_sensors.append(gid)
            elif role == ROLE_AP:
                self.selected_aps.add(gid)
                self.selection_order_aps.append(gid)
            elif role == ROLE_SKIP:
                self.skipped_candidates.add(gid)
            elif role == ROLE_STOP:
                self.done = True
        else:
            info["invalid_action"] = True
            self.invalid_action_count += 1

        self.step_count += 1
        if self._terminal_condition():
            self.done = True
        new_summary = self.render_state_summary()
        reward = compute_step_reward(prev_summary, new_summary, info, self.cfg.get("reward", {}))
        self.step_reward_sum += float(reward)
        if self.done:
            terminal_reward, metrics = compute_terminal_reward(self, self.cfg.get("reward", {}), self.cfg.get("normalization", {}))
            self.last_eval_metrics = {k: v for k, v in metrics.items() if k != "solution"}
            terminal_info = dict(self.last_eval_metrics)
            terminal_info.update({
                "terminal_reward": float(terminal_reward),
                "step_reward_sum": float(self.step_reward_sum),
                "num_sensors": len(self.selected_sensors),
                "num_aps": len(self.selected_aps),
                "invalid_action_count": int(self.invalid_action_count),
            })
            info.update(terminal_info)
            reward += terminal_reward
        self._last_summary = new_summary
        return build_init_state(self), float(reward), bool(self.done), info

    def get_action_mask(self) -> np.ndarray:
        return build_joint_action_mask(self)

    def build_current_individual(self):
        return build_individual_from_orders(
            self.ctx,
            self.selection_order_sensors,
            self.selection_order_aps,
            priority_noise_std=float(self.cfg.get("priority_noise_std", 0.03)),
            rng=self.rng,
        )

    def evaluate_current_solution(self) -> dict:
        reward, metrics = compute_terminal_reward(self, self.cfg.get("reward", {}), self.cfg.get("normalization", {}))
        metrics["reward"] = reward
        self.last_eval_metrics = {k: v for k, v in metrics.items() if k != "solution"}
        return metrics

    def render_state_summary(self) -> dict:
        state = build_init_state(self) if self.ctx.num_candidates else {"global_features": np.zeros(15)}
        gf = state["global_features"]
        return {
            "estimated_coverage": float(gf[4]),
            "estimated_energy": float(gf[5]),
            "estimated_link": float(gf[6]),
            "estimated_sink_pressure": float(gf[8]),
            "num_sensors": len(self.selected_sensors),
            "num_aps": len(self.selected_aps),
        }

    def too_few_nodes(self) -> bool:
        return (
            len(self.selected_sensors) < int(self.cfg.get("min_selected_sensors", self._deployment_limit("min_sensors", 1)))
            or len(self.selected_aps) < int(self.cfg.get("min_selected_aps", self._deployment_limit("min_aps", 1)))
        )

    def _is_valid_action(self, gid, role, info):
        n = int(self.ctx.num_candidates)
        if gid < 0 or gid >= n:
            return False
        if gid in self.skipped_candidates or gid in self.selected_sensors or gid in self.selected_aps:
            info["duplicate_or_conflict"] = True
            return False
        im = self.ctx.index_mapping
        if role == ROLE_SENSOR:
            return im.Ls_global_to_local[gid] >= 0 and len(self.selected_sensors) < int(self.cfg.get("max_selected_sensors", self._deployment_limit("max_sensors", 20)))
        if role == ROLE_AP:
            return im.La_global_to_local[gid] >= 0 and len(self.selected_aps) < int(self.cfg.get("max_selected_aps", self._deployment_limit("max_aps", 4)))
        if role == ROLE_SKIP:
            return True
        if role == ROLE_STOP:
            return not self.too_few_nodes() and bool(self.cfg.get("allow_stop_action", True))
        return False

    def _terminal_condition(self):
        max_steps = int(self.cfg.get("max_steps_per_episode", 30))
        max_s = int(self.cfg.get("max_selected_sensors", self._deployment_limit("max_sensors", 20)))
        max_a = int(self.cfg.get("max_selected_aps", self._deployment_limit("max_aps", 4)))
        if self.step_count >= max_steps:
            return True
        if len(self.selected_sensors) >= max_s and len(self.selected_aps) >= max_a:
            return True
        return not np.any(self.get_action_mask())

    def _deployment_limit(self, key, default):
        return self.config.get("deployment", {}).get(key, default)


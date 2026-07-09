"""Sequential RL initializer for DRL-Init-CR-MODE.

This module trains a policy-gradient placement policy. The policy builds an
individual step by step, evaluates it with the real decoder/repair/objective
pipeline, and updates from the resulting scalar reward. It is intentionally used
only for initialization; the generated population is still optimized by CR-MODE.
"""
from __future__ import annotations

import csv
import os
import time
import numpy as np
from src.model.individual import Individual
from src.evaluator.individual_evaluator import evaluate_individual

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None
    nn = None
    F = None


class SequentialPlacementPolicy(nn.Module if nn is not None else object):
    """MLP policy that scores available placement actions."""

    def __init__(self, feature_dim, hidden_dims=(128, 128)):
        if nn is None:
            raise ImportError("PyTorch is required for SequentialPlacementPolicy")
        super().__init__()
        layers = []
        prev = int(feature_dim)
        for hidden in hidden_dims:
            layers.extend([nn.Linear(prev, int(hidden)), nn.ReLU()])
            prev = int(hidden)
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# Backward-compatible alias used by old saved artifacts/tests.
InitPolicyNet = SequentialPlacementPolicy


class DRLPriorityInitializer:
    """Sequential policy-gradient initializer used by DRL-Init-CR-MODE."""

    def __init__(self, ctx, config):
        self.ctx = ctx
        self.config = config
        seed = int(config.get("experiment", {}).get("seeds", [42])[0])
        self.rng = np.random.default_rng(seed)
        self.policy = None
        self.training_log = []
        self.train_seconds = 0.0
        self.device = None
        self.baseline = None
        self.feature_dim = self._infer_feature_dim()

    def train_policy(self, episodes=None):
        """Train a sequential placement policy using REINFORCE."""
        cfg = self.config.get("drl_init", {})
        episodes = int(episodes if episodes is not None else cfg.get("train_episodes", 80))
        if episodes <= 0:
            return self.training_log
        if torch is None:
            self.training_log.append({
                "episode": 0,
                "loss": np.nan,
                "reward": 0.0,
                "coverage": 0.0,
                "throughput": 0.0,
                "cv": np.nan,
                "feasible": False,
                "note": "torch_unavailable",
            })
            return self.training_log

        hidden_dims = tuple(cfg.get("hidden_dims", [128, 128]))
        lr = float(cfg.get("lr", 3.0e-4))
        entropy_coef = float(cfg.get("entropy_coef", 0.01))
        baseline_momentum = float(cfg.get("baseline_momentum", 0.9))
        temperature = float(cfg.get("train_temperature", 1.0))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = SequentialPlacementPolicy(self.feature_dim, hidden_dims).to(self.device)
        optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        start = time.time()
        for episode in range(episodes):
            sample = self._sample_episode(sample_actions=True, temperature=temperature)
            sol, _ = evaluate_individual(
                sample["individual"],
                self.ctx,
                max_repair_iter=int(cfg.get("eval_repair_iter", self.config.get("constraints", {}).get("max_repair_iter", 5))),
            )
            reward, components = self._reward(sol)
            if self.baseline is None:
                self.baseline = reward
            else:
                self.baseline = baseline_momentum * self.baseline + (1.0 - baseline_momentum) * reward
            advantage = reward - self.baseline

            log_prob_sum = torch.stack(sample["log_probs"]).sum() if sample["log_probs"] else torch.tensor(0.0, device=self.device)
            entropy_sum = torch.stack(sample["entropies"]).sum() if sample["entropies"] else torch.tensor(0.0, device=self.device)
            loss = -(float(advantage) * log_prob_sum) - entropy_coef * entropy_sum

            optimizer.zero_grad()
            loss.backward()
            grad_clip = float(cfg.get("grad_clip", 1.0))
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), grad_clip)
            optimizer.step()

            self.training_log.append({
                "episode": episode + 1,
                "loss": float(loss.detach().cpu().item()),
                "reward": float(reward),
                "baseline": float(self.baseline),
                "advantage": float(advantage),
                "coverage": float(sol.coverage),
                "throughput": float(components["throughput"]),
                "cv": float(sol.cv),
                "feasible": bool(sol.feasible),
                "selected_sensors": int(len(sample["sensor_locals"])),
                "selected_aps": int(len(sample["ap_locals"])),
                "note": "sequential_reinforce",
            })
        self.train_seconds = time.time() - start
        return self.training_log

    def create_individual(self, noise=0.05):
        """Generate one individual by sequentially sampling the trained policy."""
        cfg = self.config.get("drl_init", {})
        if torch is None or self.policy is None:
            return self._heuristic_individual(noise=noise)
        temperature = float(cfg.get("init_temperature", max(0.15, noise * 8.0)))
        sample = self._sample_episode(sample_actions=True, temperature=temperature)
        return sample["individual"]

    def save_policy(self, path):
        """Save the trained sequential policy network if available."""
        if torch is None or self.policy is None:
            return None
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "policy_type": "sequential_reinforce",
            "feature_dim": self.feature_dim,
            "state_dict": self.policy.state_dict(),
            "train_seconds": self.train_seconds,
            "baseline": self.baseline,
            "config": self.config.get("drl_init", {}),
        }, path)
        return path

    def save_training_log(self, path):
        """Write policy-training reward history."""
        if not self.training_log:
            return None
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fieldnames = list(self.training_log[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.training_log)
        return path

    def _sample_episode(self, sample_actions, temperature=1.0):
        if torch is None or self.policy is None:
            raise RuntimeError("Sequential policy is not available")
        im = self.ctx.index_mapping
        selected_global = set()
        sensor_locals = []
        ap_locals = []
        log_probs = []
        entropies = []
        schedule = self._action_schedule()
        max_s = int(self.config.get("deployment", {}).get("max_sensors", 20))
        max_a = int(self.config.get("deployment", {}).get("max_aps", 4))

        for step_idx, action_type in enumerate(schedule):
            is_sensor = action_type == "sensor"
            local_to_global = im.Ls_local_to_global if is_sensor else im.La_local_to_global
            already_selected = set(sensor_locals if is_sensor else ap_locals)
            available = [
                local_idx for local_idx, gid in enumerate(local_to_global)
                if local_idx not in already_selected and int(gid) not in selected_global
            ]
            if not available:
                continue
            gids = np.asarray([local_to_global[i] for i in available], dtype=int)
            features = self._action_features(
                gids,
                is_sensor=is_sensor,
                is_ap=not is_sensor,
                selected_sensors=len(sensor_locals),
                selected_aps=len(ap_locals),
                max_sensors=max_s,
                max_aps=max_a,
                step_idx=step_idx,
                total_steps=max(len(schedule), 1),
            ).astype(np.float32)
            x = torch.as_tensor(features, dtype=torch.float32, device=self.device)
            logits = self.policy(x) / max(float(temperature), 1e-6)
            dist = torch.distributions.Categorical(logits=logits)
            if sample_actions:
                action_pos = dist.sample()
            else:
                action_pos = torch.argmax(logits)
            local_idx = int(available[int(action_pos.detach().cpu().item())])
            selected_global.add(int(local_to_global[local_idx]))
            if is_sensor:
                sensor_locals.append(local_idx)
            else:
                ap_locals.append(local_idx)
            log_probs.append(dist.log_prob(action_pos))
            entropies.append(dist.entropy())

        return {
            "individual": self._individual_from_selection(sensor_locals, ap_locals),
            "sensor_locals": sensor_locals,
            "ap_locals": ap_locals,
            "log_probs": log_probs,
            "entropies": entropies,
        }

    def _action_schedule(self):
        cfg = self.config.get("drl_init", {})
        order = str(cfg.get("placement_order", "interleaved")).lower()
        max_s = int(self.config.get("deployment", {}).get("max_sensors", 20))
        max_a = int(self.config.get("deployment", {}).get("max_aps", 4))
        if order == "sensors_first":
            return ["sensor"] * max_s + ["ap"] * max_a
        if order == "aps_first":
            return ["ap"] * max_a + ["sensor"] * max_s
        schedule = []
        for i in range(max(max_s, max_a)):
            if i < max_a:
                schedule.append("ap")
            if i < max_s:
                schedule.append("sensor")
        return schedule

    def _individual_from_selection(self, sensor_locals, ap_locals):
        im = self.ctx.index_mapping
        cfg = self.config.get("drl_init", {})
        low_noise = float(cfg.get("background_noise", 0.02))
        ind = Individual(im.num_sensor_candidates, im.num_ap_candidates)
        ind.rho_s = self.rng.uniform(0.0, low_noise, im.num_sensor_candidates)
        ind.rho_a = self.rng.uniform(0.0, low_noise, im.num_ap_candidates)
        for rank, local_idx in enumerate(sensor_locals):
            ind.rho_s[local_idx] = max(0.0, 1.0 - 0.01 * rank)
        for rank, local_idx in enumerate(ap_locals):
            ind.rho_a[local_idx] = max(0.0, 1.0 - 0.01 * rank)
        ind.clip()
        return ind

    def _reward(self, sol):
        cfg = self.config.get("drl_init", {})
        reward_cfg = cfg.get("reward", {})
        metric = self.config.get("objectives", {}).get("throughput_metric", "actual")
        throughput_key = "throughput_capacity" if metric == "capacity" else "throughput_actual"
        throughput = float(sol.metadata.get(throughput_key, sol.throughput))
        ref = float(reward_cfg.get("throughput_ref", self.config.get("evaluation", {}).get("rsum_ref_max", 2.0e7)))
        ref = max(ref, 1.0)
        throughput_norm = min(np.log1p(max(throughput, 0.0)) / np.log1p(ref), 2.0)
        cv_ref = max(float(reward_cfg.get("cv_ref", self.config.get("constraints", {}).get("cv_pressure_refs", {}).get("total", 20.0))), 1e-12)
        cv_norm = min(float(sol.cv) / cv_ref, 3.0)
        reward = (
            float(reward_cfg.get("coverage_weight", 1.0)) * float(sol.coverage)
            + float(reward_cfg.get("throughput_weight", 0.35)) * throughput_norm
            + (float(reward_cfg.get("feasible_bonus", 0.25)) if sol.feasible else 0.0)
            - float(reward_cfg.get("cv_penalty", 0.45)) * cv_norm
        )
        return float(reward), {"throughput": throughput, "throughput_norm": throughput_norm, "cv_norm": cv_norm}

    def _infer_feature_dim(self):
        im = self.ctx.index_mapping
        gids = np.asarray(im.Ls_local_to_global[:1] or im.La_local_to_global[:1], dtype=int)
        return int(self._action_features(
            gids,
            is_sensor=True,
            is_ap=False,
            selected_sensors=0,
            selected_aps=0,
            max_sensors=1,
            max_aps=1,
            step_idx=0,
            total_steps=1,
        ).shape[1])

    def _action_features(self, gids, is_sensor, is_ap, selected_sensors, selected_aps, max_sensors, max_aps, step_idx, total_steps):
        base = self._candidate_features(gids, is_sensor=is_sensor, is_ap=is_ap)
        n = len(gids)
        state = np.column_stack([
            np.full(n, selected_sensors / max(float(max_sensors), 1.0)),
            np.full(n, selected_aps / max(float(max_aps), 1.0)),
            np.full(n, max(max_sensors - selected_sensors, 0) / max(float(max_sensors), 1.0)),
            np.full(n, max(max_aps - selected_aps, 0) / max(float(max_aps), 1.0)),
            np.full(n, step_idx / max(float(total_steps - 1), 1.0)),
            np.ones(n),
        ])
        return np.hstack([base, state])

    def _candidate_features(self, gids, is_sensor, is_ap):
        pts_all = self.ctx.candidate_points_full
        pts = pts_all[gids]
        space = self.config.get("space", {})
        scale = np.asarray([
            max(float(space.get("Lx", 3.0)), 1e-12),
            max(float(space.get("Ly", 3.0)), 1e-12),
            max(float(space.get("Lz", 3.0)), 1e-12),
        ])
        xyz = pts[:, :3].astype(float) / scale
        face_ids = pts[:, 3].astype(int)
        face_one_hot = np.zeros((len(gids), 6), dtype=float)
        face_one_hot[np.arange(len(gids)), np.clip(face_ids, 0, 5)] = 1.0
        T = self.ctx.T_r[gids]
        T_norm = ((T - self.ctx.T_r.min()) / (self.ctx.T_r.max() - self.ctx.T_r.min() + 1e-12)).reshape(-1, 1)
        P = self.ctx.P_grid[gids]
        P_norm = ((P - self.ctx.P_grid.min()) / (self.ctx.P_grid.max() - self.ctx.P_grid.min() + 1e-12)).reshape(-1, 1)
        coverage = self._coverage_scores(gids).reshape(-1, 1)
        link = self._link_quality_scores(gids).reshape(-1, 1)
        flags = np.column_stack([
            np.full(len(gids), 1.0 if is_sensor else 0.0),
            np.full(len(gids), 1.0 if is_ap else 0.0),
        ])
        return np.hstack([xyz, face_one_hot, T_norm, P_norm, coverage, link, flags])

    def _coverage_scores(self, gids):
        pts = self.ctx.candidate_points_full[gids, :3].astype(float)
        radius = float(self.config.get("targets", {}).get("sensing_radius", 1.0))
        scores = []
        for point in pts:
            dist = np.linalg.norm(self.ctx.target_coords - point, axis=1)
            scores.append(np.mean(dist <= radius))
        return np.asarray(scores, dtype=float)

    def _link_quality_scores(self, gids):
        im = self.ctx.index_mapping
        ap_gids = np.asarray(im.La_local_to_global, dtype=int)
        if len(ap_gids) == 0:
            return np.zeros(len(gids), dtype=float)
        distances = self.ctx.distance_matrix[np.ix_(gids, ap_gids)]
        quality = np.mean(1.0 / (distances + 1e-6), axis=1)
        return (quality - quality.min()) / (quality.max() - quality.min() + 1e-12)

    def _heuristic_individual(self, noise=0.05):
        im = self.ctx.index_mapping
        ind = Individual(im.num_sensor_candidates, im.num_ap_candidates)
        sensor_gid = np.asarray(im.Ls_local_to_global, dtype=int)
        ap_gid = np.asarray(im.La_local_to_global, dtype=int)
        ind.rho_s = np.clip(self._heuristic_scores(sensor_gid) + self.rng.normal(0.0, noise, len(sensor_gid)), 0.0, 1.0)
        ind.rho_a = np.clip(self._heuristic_scores(ap_gid) + self.rng.normal(0.0, noise, len(ap_gid)), 0.0, 1.0)
        return ind

    def _heuristic_scores(self, gids):
        pgrid = self.ctx.P_grid[gids]
        p_norm = (pgrid - pgrid.min()) / (pgrid.max() - pgrid.min() + 1e-12)
        coverage = self._coverage_scores(gids)
        link = self._link_quality_scores(gids)
        score = 0.55 * p_norm + 0.30 * coverage + 0.15 * link
        return (score - score.min()) / (score.max() - score.min() + 1e-12)

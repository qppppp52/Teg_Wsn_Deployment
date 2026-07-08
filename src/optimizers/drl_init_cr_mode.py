"""DRL-initialized CR-MODE optimizer."""
from __future__ import annotations

import csv
import os
import time
import numpy as np
from src.optimizers.cr_mode import CRMode, _copy_solution_metrics
from src.model.population import Population
from src.model.individual import create_random_individual, create_greedy_energy_individual
from src.init.drl_priority_initializer import DRLPriorityInitializer
from src.evaluator.individual_evaluator import evaluate_individual
from src.operators.mode_mutation import mutate
from src.operators.mode_crossover import crossover
from src.operators.mode_selection import select_better


class DRLInitCRMode(CRMode):
    """CR-MODE with a sequential policy-gradient initializer."""

    def __init__(self, ctx, config):
        super().__init__(ctx, config)
        self.init_metrics = []
        self.init_training_log = []
        self.init_train_seconds = 0.0
        self.init_policy_path = None

    def run(self):
        self._initialize_mixed_drl_population()
        self.archive.update(self.population.solutions)
        self._record_convergence(0)
        self._save_init_metrics()
        for gen in range(self.Tmax):
            new_inds, new_sols = [], []
            for i, ind in enumerate(self.population.individuals):
                mutant = mutate(ind, self.strategy.F, self.strategy.mutation_strategy, self.population.individuals, i, self.ctx)
                trial = crossover(ind, mutant, self.strategy.crossover_rate)
                trial_sol, rep_ind = evaluate_individual(trial, self.ctx)
                if rep_ind is not None:
                    trial = rep_ind
                    trial_sol, _ = evaluate_individual(trial, self.ctx)
                winner, winner_sol, _ = select_better(trial, trial_sol, ind, self.population.solutions[i])
                _copy_solution_metrics(winner, winner_sol)
                new_inds.append(winner)
                new_sols.append(winner_sol)
            self.population.individuals = new_inds
            self.population.solutions = new_sols
            self.archive.update(new_sols)
            self._record_convergence(gen + 1)
        return self.archive

    def _initialize_mixed_drl_population(self):
        im = self.ctx.index_mapping
        cfg = self.config.get("drl_init", {})
        gen_ratio = float(cfg.get("generated_ratio", 0.4))
        heu_ratio = float(cfg.get("heuristic_ratio", 0.3))
        n_gen = int(round(self.NP * gen_ratio))
        n_heu = int(round(self.NP * heu_ratio))
        n_rand = max(0, self.NP - n_gen - n_heu)
        initializer = DRLPriorityInitializer(self.ctx, self.config)

        train_start = time.time()
        self.init_training_log = initializer.train_policy(cfg.get("train_episodes", 50))
        self.init_train_seconds = initializer.train_seconds or (time.time() - train_start)
        self._save_init_training_artifacts(initializer)

        individuals = []
        sources = []
        noise = float(cfg.get("noise", 0.05))
        for _ in range(n_gen):
            individuals.append(initializer.create_individual(noise=noise))
            sources.append("sequential_drl_policy")
        for _ in range(n_heu):
            individuals.append(create_greedy_energy_individual(self.ctx))
            sources.append("heuristic")
        for _ in range(n_rand):
            individuals.append(create_random_individual(im.num_sensor_candidates, im.num_ap_candidates))
            sources.append("random")

        self.population = Population(self.NP, im.num_sensor_candidates, im.num_ap_candidates)
        self.population.individuals = individuals[:self.NP]
        self.population.solutions = []
        for idx, ind in enumerate(self.population.individuals):
            sol, rep_ind = evaluate_individual(ind, self.ctx)
            if rep_ind is not None:
                ind = rep_ind
                self.population.individuals[idx] = ind
            _copy_solution_metrics(ind, sol)
            self.population.solutions.append(sol)
            self.init_metrics.append({
                "source_type": sources[idx],
                "individual_id": idx,
                "feasible_before_repair": getattr(sol, "feasible_before_repair", False),
                "cv_before_repair": getattr(sol, "cv_before_repair", sol.cv),
                "feasible": sol.feasible,
                "cv": sol.cv,
                "coverage": sol.coverage,
                "rsum": sol.throughput,
                "throughput_actual": sol.metadata.get("throughput_actual", sol.throughput),
                "throughput_capacity": sol.metadata.get("throughput_capacity", sol.throughput),
                "repair_iter": getattr(sol, "repair_iter", 0),
            })

    def _save_init_training_artifacts(self, initializer):
        base = self.config.get("runtime", {}).get("output_dir", "results")
        data_dir = os.path.join(base, "data")
        model_dir = os.path.join(base, "models")
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        initializer.save_training_log(os.path.join(data_dir, "init_training_log.csv"))
        self.init_policy_path = initializer.save_policy(os.path.join(model_dir, "init_policy.pt"))

    def _save_init_metrics(self):
        if not self.init_metrics:
            return
        base = self.config.get("runtime", {}).get("output_dir", "results")
        os.makedirs(os.path.join(base, "data"), exist_ok=True)
        path = os.path.join(base, "data", "init_population_metrics.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.init_metrics[0].keys()))
            writer.writeheader()
            writer.writerows(self.init_metrics)


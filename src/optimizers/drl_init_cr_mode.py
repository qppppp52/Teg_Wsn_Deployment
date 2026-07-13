"""DRL-initialized CR-MODE optimizer."""
from __future__ import annotations

from src.optimizers.cr_mode import CRMode
from src.rl_init.evaluation_counter import DRLInitEvaluationCounter
from src.rl_init.population_generator import DRLInitPopulationGenerator
from src.utils.logger import get_logger


class DRLInitCRMode(CRMode):
    """CR-MODE whose only algorithmic difference is PPO-generated Gen0."""

    def __init__(self, ctx, config):
        super().__init__(ctx, config)
        self.logger = get_logger("DRL-INIT-CR-MODE")
        self.drl_cost_counter = DRLInitEvaluationCounter()
        self.generator = None
        self.init_metrics = []
        self.init_training_log = []
        self.init_train_seconds = 0.0
        self.init_generation_seconds = 0.0
        self.init_policy_path = ""
        self.accepted_counts = {"drl": 0, "heuristic": 0, "random": 0}
        self.loaded_checkpoint = False
        self.policy_source = ""
        self.checkpoint_path = ""
        self.torch_available = False
        self.drl_init_summary = {}
        self.checkpoint_compatible = False
        self.checkpoint_skip_reason = ""
        self.checkpoint_error = ""
        self.checkpoint_mode = ""
        self.config_hash = ""
        self.checkpoint_load_seconds = 0.0
        self.pretrain_seconds = 0.0
        self.drl_init_valid = False
        self.drl_init_invalid_reasons = []

    def initialize_population(self):
        self.generator = DRLInitPopulationGenerator(
            self.ctx,
            self.config,
            evaluation_counter=self.drl_cost_counter,
        )
        population = self.generator.generate(self.NP)
        self._sync_generator_state()
        output_dir = self.config.get("runtime", {}).get("output_dir", "results")
        self.generator.save_artifacts(output_dir)
        self.drl_init_summary = self.generator.summary

        if (
            self.config.get("drl_init", {}).get("require_drl_policy", True)
            and not self.drl_init_valid
            and self.config.get("drl_init", {}).get("invalid_run_policy", "fail") == "fail"
        ):
            reasons = ", ".join(self.drl_init_invalid_reasons) or "unknown"
            raise RuntimeError(f"Invalid formal DRL initialization: {reasons}")
        return population

    def run(self):
        archive = super().run()
        if self.generator is not None:
            self.generator.cost_timing = {
                "gen0_evaluation_seconds": self.gen0_evaluation_seconds,
                "cr_mode_search_seconds": self.cr_mode_search_seconds,
                "online_optimization_seconds": self.online_optimization_seconds,
                "total_end_to_end_seconds": self.total_end_to_end_seconds,
            }
            self.generator.save_artifacts(
                self.config.get("runtime", {}).get("output_dir", "results")
            )
            self.drl_init_summary = self.generator.summary
            self._sync_generator_state()
        return archive

    def _on_gen0_evaluated(self, count):
        self.drl_cost_counter.increment("gen0_re_evaluations", count)

    def _on_trials_evaluated(self, count):
        self.drl_cost_counter.increment("cr_mode_trial_evaluations", count)

    def _sync_generator_state(self):
        generator = self.generator
        self.init_metrics = generator.init_metrics
        self.init_training_log = generator.training_log
        self.init_train_seconds = float(generator.train_seconds)
        self.init_generation_seconds = float(generator.population_generation_seconds)
        self.init_policy_path = generator.policy_path
        self.accepted_counts = generator.accepted_counts
        self.loaded_checkpoint = bool(generator.loaded_checkpoint)
        self.policy_source = generator.policy_source
        self.checkpoint_path = generator.policy_path
        self.torch_available = bool(generator.torch_available)
        self.checkpoint_compatible = bool(generator.loaded_checkpoint_compatible)
        self.checkpoint_skip_reason = generator.checkpoint_load_skip_reason
        self.checkpoint_error = generator.checkpoint_load_error
        self.checkpoint_mode = generator.checkpoint_mode
        self.config_hash = generator.config_hash
        self.checkpoint_load_seconds = float(generator.checkpoint_load_seconds)
        self.pretrain_seconds = float(generator.pretrain_seconds)
        self.drl_init_valid = bool(generator.drl_init_valid)
        self.drl_init_invalid_reasons = list(generator.drl_init_invalid_reasons)

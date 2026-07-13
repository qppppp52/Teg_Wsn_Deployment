

## Small Center-Heat Scenario for Algorithm Validation

This project includes a small reproducible validation scenario for comparing CR-MODE, DQN-CR-MODE, and DRL-Init-CR-MODE.

- Scene size: 3 m x 3 m x 3 m enclosed cube.
- Heat source: internal center source at [1.5, 1.5, 1.5] m.
- Heat-source temperature: 343.15 K (70 C).
- Temperature model: `center_air_convection_synthetic`, a simplified center-air convection field with buoyancy, face multipliers, and plume-direction terms. It is for algorithm validation and does not replace high-fidelity ANSYS thermal simulation.
- Train the DQN controller first: `conda run -n huanjin3.12 python train_dqn.py --config configs/experiment_small_compare.yaml`.
- Run the frozen-policy comparison: `conda run -n huanjin3.12 python main.py --experiment configs/experiment_small_compare.yaml`.
- A short pipeline check may use `--episodes 2`; formal training uses the configured 120 episodes.
- Formal DQN evaluation fails clearly when the configured checkpoint is missing or incompatible. It never falls back to a random policy.
- CR-MODE and DQN-CR-MODE reuse the same checksummed raw Gen0 snapshot for each seed. DRL-Init-CR-MODE keeps its PPO-generated initialization because initialization is its experimental variable.
- Every evolutionary generation evaluates exactly one trial per population member and records the cumulative evaluation count.
- DQN training artifacts are written to `experiments/dqn_training`; the best checkpoint is written to `experiments/checkpoints`.
- `main.py` runs DQN inference only, so DQN training time is not mixed into the online optimization runtime comparison.
- DRL-Init-CR-MODE uses a per-seed PPO policy only to generate Gen0; generations 1-60 use the same CR-MODE search core as the baseline.
- PPO setup/training, population generation, Gen0 evaluation, and CR-MODE search times are reported separately.
- Formal DRL-Init runs require PyTorch, a trained or compatible loaded PPO policy, and enough accepted DRL individuals; invalid runs fail instead of silently becoming heuristic/random results.
- `drl_init_summary.json` records source-wise DRL/heuristic/random quality, actual evaluation counts, fallback reasons, PPO update evidence, and checkpoint load/save status.
- Per-seed PPO uses information from that seed's scenario and must not be described as unseen-seed generalization.
- A separate `shared_pretrain` study would be required for frozen PPO generalization; this project keeps the current per-seed initialization research design.
- Main outputs: `experiments/small_center_heat_compare/<timestamp>/scene_preprocess/shared` for scene preprocessing, plus `<algorithm>/seed_<seed>/data`, `figures`, and `pareto` for algorithm results.
- Recommended Pareto solution: selected by normalized distance to the ideal point using Coverage and normalized Rsum.

The small scenario exports shared scene files (`candidate_temperature.csv`, `temperature_faces.png`, and `pgrid_distribution.png`) once per experiment, while algorithm-specific files include `convergence.csv`, `pareto_solutions.csv`, and run summaries.

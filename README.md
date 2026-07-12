

## Small Center-Heat Scenario for Algorithm Validation

This project includes a small reproducible validation scenario for comparing CR-MODE, DQN-CR-MODE, and DRL-Init-CR-MODE.

- Scene size: 3 m x 3 m x 3 m enclosed cube.
- Heat source: internal center source at [1.5, 1.5, 1.5] m.
- Heat-source temperature: 343.15 K (70 C).
- Temperature model: `center_air_convection_synthetic`, a simplified center-air convection field with buoyancy, face multipliers, and plume-direction terms. It is for algorithm validation and does not replace high-fidelity ANSYS thermal simulation.
- Run command: `python main.py --experiment configs/experiment_small_compare.yaml`.
- Main outputs: `experiments/small_center_heat_compare/<timestamp>/scene_preprocess/shared` for scene preprocessing, plus `<algorithm>/seed_<seed>/data`, `figures`, and `pareto` for algorithm results.
- Recommended Pareto solution: selected by normalized distance to the ideal point using Coverage and normalized Rsum.

The small scenario exports shared scene files (`candidate_temperature.csv`, `temperature_faces.png`, and `pgrid_distribution.png`) once per experiment, while algorithm-specific files include `convergence.csv`, `pareto_solutions.csv`, and run summaries.

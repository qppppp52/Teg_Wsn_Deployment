"""Raw Gen0 population snapshots for fair optimizer comparison."""
from __future__ import annotations

import hashlib
import json
import os
import numpy as np

from src.model.individual import Individual
from src.model.population import Population


def save_population_snapshot(population, path, seed):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rho_s = np.stack([individual.rho_s for individual in population.individuals])
    rho_a = np.stack([individual.rho_a for individual in population.individuals])
    metadata = {
        "schema_version": 1,
        "seed": int(seed),
        "population_size": len(population.individuals),
        "num_sensor_candidates": int(population.num_s),
        "num_ap_candidates": int(population.num_a),
    }
    digest = hashlib.sha256(rho_s.tobytes() + rho_a.tobytes()).hexdigest()
    metadata["sha256"] = digest
    np.savez_compressed(
        path,
        rho_s=rho_s,
        rho_a=rho_a,
        metadata=json.dumps(metadata, sort_keys=True),
    )
    return metadata


def load_population_snapshot(path, expected_size, num_s, num_a):
    data = np.load(path, allow_pickle=False)
    rho_s = np.asarray(data["rho_s"], dtype=float)
    rho_a = np.asarray(data["rho_a"], dtype=float)
    expected_s = (int(expected_size), int(num_s))
    expected_a = (int(expected_size), int(num_a))
    if rho_s.shape != expected_s or rho_a.shape != expected_a:
        raise ValueError(
            "Initial population snapshot shape mismatch: "
            f"rho_s={rho_s.shape}, rho_a={rho_a.shape}, "
            f"expected={expected_s}/{expected_a}"
        )
    population = Population(int(expected_size), int(num_s), int(num_a))
    for sensor_priorities, ap_priorities in zip(rho_s, rho_a):
        individual = Individual(int(num_s), int(num_a))
        individual.rho_s = sensor_priorities.copy()
        individual.rho_a = ap_priorities.copy()
        population.individuals.append(individual)
    digest = hashlib.sha256(rho_s.tobytes() + rho_a.tobytes()).hexdigest()
    metadata = json.loads(str(data["metadata"]))
    if metadata.get("sha256") != digest:
        raise ValueError("Initial population snapshot checksum mismatch")
    return population, metadata

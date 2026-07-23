"""Experiment result serialization."""
from __future__ import annotations

import json
import os
import pickle

import numpy as np

from src.heatsink.sink_ownership import build_sink_ownership
from src.io.semantic_contract import build_semantic_contract, semantic_signature


RESULT_SCHEMA_VERSION = 3


def save_pareto(solutions, path: str, ctx=None):
    """Save feasible Pareto solutions with reconstructable sink ownership."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    semantic_contract = build_semantic_contract(ctx.config) if ctx is not None else {}
    artifact_semantic_signature = semantic_signature(ctx.config) if ctx is not None else ""
    feasible = [solution for solution in solutions if solution.feasible]
    objectives = np.asarray(
        [[solution.coverage, solution.rsum_capacity] for solution in feasible],
        dtype=float,
    ).reshape((-1, 2))
    if not feasible:
        np.savez_compressed(
            path,
            objectives=objectives,
            result_schema_version=np.asarray(RESULT_SCHEMA_VERSION, dtype=np.int32),
            solution_count=np.asarray(0, dtype=np.int32),
            semantic_contract=np.asarray(json.dumps(semantic_contract, sort_keys=True)),
            semantic_signature=np.asarray(artifact_semantic_signature),
        )
        return

    count = len(feasible)
    K = len(feasible[0].x)
    x_all = np.asarray([solution.x for solution in feasible], dtype=np.int8)
    y_all = np.asarray([solution.y for solution in feasible], dtype=np.int8)
    c_all = np.asarray([solution.c for solution in feasible], dtype=np.int8)
    ptx_all = np.asarray([solution.p_tx for solution in feasible], dtype=np.float64)
    n_sensor = np.asarray(
        [solution.n_sink_sensor for solution in feasible], dtype=np.int32
    )
    n_ap = np.asarray(
        [solution.n_sink_ap for solution in feasible], dtype=np.int32
    )
    sensor_by_owner = np.zeros((count, K, K), dtype=bool)
    ap_by_owner = np.zeros((count, K, K), dtype=bool)
    effective_sensor = np.zeros((count, K), dtype=np.int32)
    effective_ap = np.zeros((count, K), dtype=np.int32)
    sensor_harvest = np.zeros((count, K), dtype=np.float64)
    ap_harvest = np.zeros((count, K), dtype=np.float64)

    for solution_index, solution in enumerate(feasible):
        if ctx is not None:
            ownership = build_sink_ownership(solution, ctx)
            sensor_positions = ownership.effective_sensor_positions
            ap_positions = ownership.effective_ap_positions
        else:
            sensor_positions = tuple(
                tuple(sorted(set(int(grid) for grid in positions if 0 <= int(grid) < K)))
                for positions in solution.z_sink_sensor
            )
            ap_positions = tuple(
                tuple(sorted(set(int(grid) for grid in positions if 0 <= int(grid) < K)))
                for positions in solution.z_sink_ap
            )
        for owner_id, positions in enumerate(sensor_positions):
            sensor_by_owner[solution_index, owner_id, list(positions)] = True
            effective_sensor[solution_index, owner_id] = len(positions)
            if ctx is not None:
                sensor_harvest[solution_index, owner_id] = (
                    len(positions) * float(ctx.P_grid[owner_id])
                )
        for owner_id, positions in enumerate(ap_positions):
            ap_by_owner[solution_index, owner_id, list(positions)] = True
            effective_ap[solution_index, owner_id] = len(positions)
            if ctx is not None:
                ap_harvest[solution_index, owner_id] = (
                    len(positions) * float(ctx.P_grid[owner_id])
                )

    sensor_aggregate = np.any(sensor_by_owner, axis=1).astype(np.int8)
    ap_aggregate = np.any(ap_by_owner, axis=1).astype(np.int8)
    overlap_policy = json.dumps(
        (ctx.config.get("heatsink", {}) if ctx is not None else {}),
        sort_keys=True,
    )
    np.savez_compressed(
        path,
        objectives=objectives,
        x=x_all,
        y=y_all,
        c=c_all,
        p_tx=ptx_all,
        n_sink_sensor=n_sensor,
        n_sink_ap=n_ap,
        z_sink_sensor=sensor_aggregate,
        z_sink_ap=ap_aggregate,
        z_sink_sensor_by_owner=sensor_by_owner,
        z_sink_ap_by_owner=ap_by_owner,
        n_effective_sensor=effective_sensor,
        n_effective_ap=effective_ap,
        nmax_static=(
            np.asarray(ctx.nmax, dtype=np.int32)
            if ctx is not None
            else np.zeros(K, dtype=np.int32)
        ),
        sensor_power_consumption=np.asarray(
            [solution.sensor_power_consumption for solution in feasible],
            dtype=np.float64,
        ),
        ap_power_consumption=np.asarray(
            [solution.ap_power_consumption for solution in feasible],
            dtype=np.float64,
        ),
        sensor_harvest_power=(
            sensor_harvest
            if ctx is not None
            else np.asarray(
                [solution.sensor_harvest_power for solution in feasible],
                dtype=np.float64,
            )
        ),
        ap_harvest_power=(
            ap_harvest
            if ctx is not None
            else np.asarray(
                [solution.ap_harvest_power for solution in feasible],
                dtype=np.float64,
            )
        ),
        cv=np.asarray([solution.cv for solution in feasible], dtype=np.float64),
        feasible_mask=np.ones(count, dtype=bool),
        overlap_policy=np.asarray(overlap_policy),
        semantic_contract=np.asarray(json.dumps(semantic_contract, sort_keys=True)),
        semantic_signature=np.asarray(artifact_semantic_signature),
        boost_applied=np.asarray(
            [bool(solution.metadata.get("boost_applied", False)) for solution in feasible],
            dtype=bool,
        ),
        boost_sinks_added=np.asarray(
            [int(solution.metadata.get("boost_sinks_added", 0)) for solution in feasible],
            dtype=np.int32,
        ),
        rsum_capacity_before_boost=np.asarray(
            [
                float(
                    solution.metadata.get(
                        "rsum_capacity_before_boost", solution.rsum_capacity
                    )
                )
                for solution in feasible
            ],
            dtype=np.float64,
        ),
        rsum_capacity_boost_gain=np.asarray(
            [
                float(solution.metadata.get("rsum_capacity_boost_gain", 0.0))
                for solution in feasible
            ],
            dtype=np.float64,
        ),
        result_schema_version=np.asarray(RESULT_SCHEMA_VERSION, dtype=np.int32),
        solution_count=np.asarray(count, dtype=np.int32),
    )


def load_pareto(path: str) -> np.ndarray:
    data = np.load(path)
    return data["objectives"]


def save_solutions_raw(solutions, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as file:
        pickle.dump(solutions, file)


def save_log(text: str, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        file.write(text)

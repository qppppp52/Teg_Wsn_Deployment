"""Calibrate frozen DQN CV-pressure references from training-pilot CSV logs.

Only use logs generated from DQN training pilots. The tool never changes a
configuration in place: review its YAML output, then copy the approved values
into configs/dqn.yaml before formal retraining.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.constraints.cv_schema import CV_COMPONENT_KEYS, PRESSURE_SCHEMA_VERSION


FIELD_ALIASES = {
    key: (f"cv_{key}", f"cv_{key}_mean", f"CV_{key}", f"CV_{key}_mean")
    for key in CV_COMPONENT_KEYS
}
TOTAL_FIELD_ALIASES = ("cv_total", "cv_total_mean", "CV_total", "CV_after_repair")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="One or more CSV paths or glob patterns from training-pilot runs.",
    )
    parser.add_argument("--output-dir", default="artifacts/cv_pressure_calibration")
    parser.add_argument("--quantile", type=float, default=0.90)
    parser.add_argument("--cv-zero-tol", type=float, default=1.0e-12)
    parser.add_argument("--min-nonzero-samples", type=int, default=100)
    parser.add_argument("--min-reference", type=float, default=1.0e-6)
    parser.add_argument(
        "--fallback-ref",
        action="append",
        default=[],
        metavar="COMPONENT=VALUE",
        help="Required only when a component is always zero in the pilot logs.",
    )
    parser.add_argument(
        "--source-label",
        default="training_pilot_only",
        help="Provenance recorded in the calibration artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_args(args)
    paths = expand_paths(args.input)
    if not paths:
        raise FileNotFoundError("No calibration CSV files matched --input")
    values = collect_values(paths)
    fallbacks = parse_fallbacks(args.fallback_ref)
    report = build_report(values, paths, args, fallbacks)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_outputs(output_dir, report)
    print(f"Calibration report: {output_dir / 'cv_pressure_calibration.json'}")
    print(f"Proposed refs: {output_dir / 'cv_pressure_refs.yaml'}")


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 < args.quantile < 1.0:
        raise ValueError("--quantile must be in (0, 1)")
    if not math.isfinite(args.cv_zero_tol) or args.cv_zero_tol < 0.0:
        raise ValueError("--cv-zero-tol must be finite and >= 0")
    if args.min_nonzero_samples <= 0:
        raise ValueError("--min-nonzero-samples must be positive")
    if not math.isfinite(args.min_reference) or args.min_reference <= 0.0:
        raise ValueError("--min-reference must be finite and > 0")
    if not str(args.source_label).strip():
        raise ValueError("--source-label must be non-empty")


def expand_paths(patterns: Iterable[str]) -> list[Path]:
    matches: set[Path] = set()
    for pattern in patterns:
        resolved = [Path(path) for path in glob.glob(pattern, recursive=True)]
        if not resolved and Path(pattern).is_file():
            resolved = [Path(pattern)]
        matches.update(path.resolve() for path in resolved if path.is_file())
    return sorted(matches)


def collect_values(paths: Iterable[Path]) -> dict[str, list[float]]:
    keys = (*CV_COMPONENT_KEYS, "total")
    values = {key: [] for key in keys}
    rows_with_components = 0
    for path in paths:
        with path.open("r", newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            if not reader.fieldnames:
                continue
            field_map = {
                key: resolve_field(reader.fieldnames, FIELD_ALIASES[key])
                for key in CV_COMPONENT_KEYS
            }
            total_field = resolve_field(reader.fieldnames, TOTAL_FIELD_ALIASES)
            if not any(field_map.values()):
                continue
            for row_number, row in enumerate(reader, start=2):
                row_has_component = False
                for key, field in field_map.items():
                    if field is None or row.get(field, "") == "":
                        continue
                    values[key].append(parse_cv(row[field], path, row_number, field))
                    row_has_component = True
                if total_field is not None and row.get(total_field, "") != "":
                    values["total"].append(
                        parse_cv(row[total_field], path, row_number, total_field)
                    )
                if row_has_component:
                    rows_with_components += 1
    if rows_with_components == 0:
        required = ", ".join(f"cv_{key}_mean" for key in CV_COMPONENT_KEYS)
        raise ValueError(f"No usable component rows found; expected columns such as {required}")
    return values


def resolve_field(fieldnames: Iterable[str], aliases: Iterable[str]) -> str | None:
    available = set(fieldnames)
    return next((field for field in aliases if field in available), None)


def parse_cv(raw: str, path: Path, row_number: int, field: str) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}:{row_number} has nonnumeric {field!r}") from exc
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{path}:{row_number} has invalid nonnegative finite {field!r}")
    return value


def parse_fallbacks(entries: Iterable[str]) -> dict[str, float]:
    allowed = {*CV_COMPONENT_KEYS, "total"}
    fallbacks: dict[str, float] = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError("--fallback-ref must use COMPONENT=VALUE")
        key, raw = entry.split("=", 1)
        key = key.strip()
        if key not in allowed:
            raise ValueError(f"Unknown fallback component {key!r}")
        value = float(raw)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"Fallback reference {key!r} must be finite and > 0")
        fallbacks[key] = value
    return fallbacks


def build_report(values, paths, args, fallbacks) -> dict:
    components = {}
    for key in CV_COMPONENT_KEYS:
        components[key] = summarize_component(values[key], key, args, fallbacks)
    total = summarize_component(values["total"], "total", args, fallbacks)
    return {
        "schema_version": PRESSURE_SCHEMA_VERSION,
        "source": str(args.source_label),
        "input_files": [str(path) for path in paths],
        "quantile": float(args.quantile),
        "cv_zero_tol": float(args.cv_zero_tol),
        "min_nonzero_samples": int(args.min_nonzero_samples),
        "min_reference": float(args.min_reference),
        "components": components,
        "total": total,
    }


def summarize_component(values, key, args, fallbacks) -> dict:
    data = np.asarray(values, dtype=float)
    nonzero = data[data > args.cv_zero_tol]
    if len(nonzero) >= args.min_nonzero_samples:
        proposed = float(np.quantile(nonzero, args.quantile))
        selection = f"q{int(args.quantile * 100)}_nonzero"
    elif len(nonzero):
        proposed = float(np.max(nonzero))
        selection = "max_nonzero_insufficient_samples"
    elif key in fallbacks:
        proposed = float(fallbacks[key])
        selection = "explicit_fallback_all_zero"
    else:
        raise ValueError(
            f"Component {key!r} has no nonzero pilot samples; provide "
            f"--fallback-ref {key}=VALUE after a modelling review"
        )
    reference = max(proposed, args.min_reference)
    return {
        "sample_count": int(len(data)),
        "nonzero_count": int(len(nonzero)),
        "nonzero_ratio": float(len(nonzero) / len(data)) if len(data) else 0.0,
        "mean": float(np.mean(data)) if len(data) else 0.0,
        "median_nonzero": float(np.median(nonzero)) if len(nonzero) else None,
        "q90_nonzero": float(np.quantile(nonzero, 0.90)) if len(nonzero) else None,
        "q95_nonzero": float(np.quantile(nonzero, 0.95)) if len(nonzero) else None,
        "max": float(np.max(data)) if len(data) else 0.0,
        "selection": selection,
        "reference": float(reference),
    }


def write_outputs(output_dir: Path, report: dict) -> None:
    json_path = output_dir / "cv_pressure_calibration.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    csv_path = output_dir / "cv_pressure_calibration.csv"
    rows = []
    for key in (*CV_COMPONENT_KEYS, "total"):
        rows.append({"component": key, **report["components"].get(key, report["total"])})
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    refs = {
        key: report["components"][key]["reference"]
        for key in CV_COMPONENT_KEYS
    }
    yaml_payload = {
        "dqn": {
            "pressure_normalization": {
                "method": "fixed_reference",
                "schema_version": PRESSURE_SCHEMA_VERSION,
                "cv_zero_tol": report["cv_zero_tol"],
                "reference_source": "training_pilot_calibrated",
                "refs": refs,
            },
            "state_normalization": {
                "cv_total_ref": report["total"]["reference"],
            },
        }
    }
    with (output_dir / "cv_pressure_refs.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(yaml_payload, file, allow_unicode=False, sort_keys=False)


if __name__ == "__main__":
    main()

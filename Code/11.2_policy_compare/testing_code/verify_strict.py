"""Verify causal forecast boundaries and preserve all earlier experiment artifacts."""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import bootstrap as paths
import numpy as np
from experiment import save_csv, save_json
from evaluation import distribution_metrics
from strict_protocol import load_records
from verify_compare import compare_csv


def read(path):
    return json.loads(Path(path).read_text())


def verify(root):
    config, manifest = read(root / "config_frozen.json"), read(root / "manifest.json")
    checks = dict(unit_tests=read(root / "unit_tests.json")["passed"])
    protected = {p: paths.sha256(paths.WORKSPACE / p) == digest for p, digest in manifest["protected_inputs_sha256"].items()}
    sources = {p: paths.sha256(paths.ROOT / p) == digest for p, digest in manifest["source_sha256"].items()}
    checks.update(previous_experiments_unchanged=all(protected.values()), sources_match_run=all(sources.values()))
    all_metrics = []
    previous = paths.ROOT / manifest["prior_three_policy_run"]
    for window in sorted((root / "windows").iterdir()):
        protocol = read(window / "protocol.json")
        cutoff = protocol["forecast_origin_h"]
        checks[window.name + "_numerics"] = read(window / "numerical_verification.json")["passed"]
        for policy in ["history_only"] + config["policies"]:
            folder = window / policy
            metrics = read(folder / "task_metrics.json")
            checks[f"{window.name}_{policy}_frozen"] = read(folder / "forecast_verification.json")["passed"]
            if policy != "history_only":
                checks[f"{window.name}_{policy}_mission"] = read(folder / "verification.json")["passed"]
                checks[f"{window.name}_{policy}_execution"] = read(folder / "execution.json")["status"] == "SUCCESS"
                with (folder / "online_measurements.csv").open() as stream:
                    acquired = list(csv.DictReader(stream))
                ids = {int(r["record_id"]) for r in acquired}
                checks[f"{window.name}_{policy}_cutoff"] = all(float(r["t_h"]) <= cutoff for r in acquired)
            else:
                ids = set()
            for metric in metrics:
                task = metric["task"]
                refs = load_records(window / "common" / f"{task}_records.csv")
                prediction = load_records(folder / f"{task}_predictions.csv")
                checks[f"{window.name}_{policy}_{task}_same_records"] = [r["record_id"] for r in refs] == [r["record_id"] for r in prediction]
                checks[f"{window.name}_{policy}_{task}_heldout"] = not ids.intersection(r["record_id"] for r in refs)
                if task.startswith("forecast"):
                    lo, hi = [int(v.rstrip("m")) for v in task.split("_")[1:]]
                    checks[f"{window.name}_{policy}_{task}_future"] = all(cutoff + lo / 60 < r["t_h"] <= cutoff + hi / 60 for r in refs)
                truth = np.array([r["temperature_c"] for r in prediction])
                mean = np.array([float(r["predicted_temperature_c"]) for r in prediction])
                variance = np.array([float(r["observation_variance_c2"]) for r in prediction])
                recalculated = distribution_metrics(truth, mean, variance)
                checks[f"{window.name}_{policy}_{task}_metrics"] = all(np.isclose(value, metric[name], atol=1e-10, rtol=1e-10) for name, value in recalculated.items())
                all_metrics.append(metric)
            if window.name == "w00" and policy in ("mi", "emse", "fusion"):
                for name in ("online_measurements.csv", "robot_trajectory.csv", "metrics.csv"):
                    checks[f"legacy_{policy}_{name}"] = compare_csv(previous / policy / name, folder / name, 1e-7)
                a, b = read(previous / policy / "planned_routes.json"), read(folder / "planned_routes.json")
                checks[f"legacy_{policy}_routes"] = [p["route"] for p in a] == [p["route"] for p in b]
    grouped = defaultdict(list)
    for row in all_metrics:
        grouped[row["policy"], row["task"]].append(row)
    aggregate = []
    for (policy, task), values in grouped.items():
        aggregate.append(dict(policy=policy, task=task, window_count=len(values),
            mean_window_rmse_c=float(np.mean([r["rmse_c"] for r in values])),
            mean_window_mae_c=float(np.mean([r["mae_c"] for r in values])),
            mean_window_sensor_balanced_rmse_c=float(np.mean([r["sensor_balanced_rmse_c"] for r in values])),
            mean_window_coverage_95=float(np.mean([r["coverage_95"] for r in values])),
            record_count=sum(r["record_count"] for r in values)))
    save_csv(root / "evaluation.csv", all_metrics)
    save_json(root / "evaluation.json", all_metrics)
    save_csv(root / "aggregate.csv", aggregate)
    save_json(root / "aggregate.json", aggregate)
    save_json(root / "verification.json", dict(passed=all(checks.values()), checks=checks, protected_inputs=protected, sources=sources))
    if not all(checks.values()):
        raise AssertionError([key for key, passed in checks.items() if not passed])
    save_json(root / "artifacts_sha256.json", {str(p.relative_to(root)): paths.sha256(p) for p in root.rglob("*") if p.is_file() and p.name != "artifacts_sha256.json"})
    print("Strict causal and artifact checks passed:", len(checks), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    verify(parser.parse_args().output)

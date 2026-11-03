"""Independent posterior checks, historical MI reproduction, and artifact audit."""

import argparse
import csv
import json
from pathlib import Path

import bootstrap as paths
import numpy as np
from experiment import save_csv, save_json
from gp import GPModel
from grg import information_gain
from interfaces import Measurement
from evaluation import distribution_metrics
from objectives import FrozenCatalog


def read(path):
    return json.loads(Path(path).read_text())


def verify_numerics(root):
    gp = GPModel.load(paths.CHECKPOINT)
    catalog = FrozenCatalog(root / "common/objective_catalog.npz")
    chosen = np.unique(np.linspace(0, len(catalog.events) - 1, 4, dtype=int))
    rows = []
    for count in (0, 2):
        if count:
            for event in catalog.events[chosen[:count]]:
                gp.append(Measurement(*event, 0., "numerical_fixture", "ok"))
        catalog.condition(gp)
        queries = catalog.events[chosen]
        joint = gp.joint(np.vstack([catalog.target, queries]))[1]
        covariance = joint[-len(queries):, -len(queries):]
        cross = joint[:-len(queries), -len(queries):]
        expected_mi = float(information_gain(covariance, gp.noise))
        expected_reduction = float(np.sum(cross * np.linalg.solve(
            covariance + gp.noise * np.eye(len(queries)), cross.T).T) / len(catalog.target))
        mi, reduction = catalog.components(gp, [queries])
        errors = dict(mi_error_nats=abs(float(mi[0]) - expected_mi),
            emse_error_c2=abs(float(reduction[0]) - expected_reduction),
            target_variance_error_c2=float(np.max(np.abs(np.diag(joint)[:-len(queries)] - catalog.current_target_variance))))
        assert max(errors.values()) < 1e-7, errors
        rows.append(dict(online_fixture_events=count, **errors))
    save_json(root / "numerical_verification.json", dict(passed=True, cases=rows,
        reference="Direct GPyTorch joint posterior and independent Schur complement, before and after two numerical-fixture observations. No fixture records enter a mission."))


def compare_csv(first, second, tolerance):
    with Path(first).open(newline="") as stream:
        a = list(csv.DictReader(stream))
    with Path(second).open(newline="") as stream:
        b = list(csv.DictReader(stream))
    if len(a) != len(b) or any(set(x) != set(y) for x, y in zip(a, b)):
        return False
    for x, y in zip(a, b):
        for key in x:
            try:
                if not np.isclose(float(x[key]), float(y[key]), atol=tolerance, rtol=tolerance):
                    return False
            except ValueError:
                if x[key] != y[key]:
                    return False
    return True


def verify_mi_control(root):
    tolerance = read(root / "protocol.json")["comparison_config"]["historical_baseline_tolerance"]
    a, b = read(paths.HISTORICAL / "summary.json"), read(root / "mi/summary.json")
    checks = {key: bool(np.isclose(a[key], b[key], atol=tolerance, rtol=tolerance)) for key in
        ["measurements", "planning_calls", "elapsed_s", "travel_distance_m", "sampling_time_s", "information_gain_nats"]}
    for name in ["online_measurements.csv", "robot_trajectory.csv", "metrics.csv"]:
        checks[name] = compare_csv(paths.HISTORICAL / name, root / "mi" / name, tolerance)
    old, new = read(paths.HISTORICAL / "planned_routes.json"), read(root / "mi/planned_routes.json")
    checks["routes"] = [p["route"] for p in old] == [p["route"] for p in new]
    checks["sampling_times"] = len(old) == len(new) and all(
        np.allclose(a["arrivals_s"], b["arrivals_s"], atol=tolerance, rtol=0) for a, b in zip(old, new))
    save_json(root / "mi_baseline_verification.json", dict(passed=all(checks.values()), tolerance=tolerance,
        checks=checks, excluded="Wall-clock timing is machine dependent and is not expected to reproduce."))
    if not all(checks.values()):
        raise AssertionError(f"MI control failed historical reproduction: {checks}")


def verify_artifacts(root):
    protocol, manifest = read(root / "protocol.json"), read(root / "manifest.json")
    unchanged = {p: paths.sha256(paths.WORKSPACE / p) == digest for p, digest in manifest["protected_inputs_sha256"].items()}
    unchanged_sources = {p: paths.sha256(paths.ROOT / p) == digest for p, digest in manifest["source_sha256"].items()}
    checks = dict(protected_inputs_unchanged=all(unchanged.values()), new_sources_match_run=all(unchanged_sources.values()),
        objective_unit_checks=read(root / "unit_tests.json")["passed"],
        numerical_reference=read(root / "numerical_verification.json")["passed"],
        mi_control_reproduced=read(root / "mi_baseline_verification.json")["passed"])
    rows = []
    for policy in protocol["comparison_config"]["policies"]:
        folder = root / policy
        summary = read(folder / "summary.json")
        checks[policy + "_mission_checks"] = read(folder / "verification.json")["passed"]
        checks[policy + "_worker_success"] = read(folder / "execution.json")["status"] == "SUCCESS"
        checks[policy + "_shared_checkpoint"] = summary["initial_gp_sha256"] == paths.sha256(paths.CHECKPOINT)
        checks[policy + "_shared_catalog"] = summary["initial_catalog_sha256"] == paths.sha256(root / "common/objective_catalog.npz")
        with np.load(folder / "reference_predictions.npz") as data:
            metrics = distribution_metrics(data["truth"], data["final_mean"], data["final_observed_var"])
            checks[policy + "_metrics_recomputed"] = all(np.isclose(v, summary["final_metrics"][k], atol=1e-10, rtol=1e-10) for k, v in metrics.items())
        rows.append(dict(policy=policy, measurements=summary["measurements"], planning_calls=summary["planning_calls"],
            final_rmse_c=summary["final_metrics"]["rmse_c"], final_mae_c=summary["final_metrics"]["mae_c"],
            coverage_95=summary["final_metrics"]["coverage_95"], nlpd=summary["final_metrics"]["nlpd"],
            initial_rmse_c=summary["initial_metrics"]["rmse_c"],
            mean_target_variance_c2=summary["final_mean_target_variance_c2"],
            heldout_mean_variance_c2=summary["final_metrics"]["mean_variance_c2"],
            executed_joint_mi_nats=summary["executed_joint_mi_nats"],
            travel_distance_m=summary["travel_distance_m"], elapsed_s=summary["elapsed_s"],
            measured_plan_wall_s=summary["measured_plan_wall_s"], worker_wall_s=summary["worker_wall_s"],
            returned_home=summary["returned_home"]))
    save_csv(root / "comparison.csv", rows)
    save_json(root / "comparison.json", rows)
    save_json(root / "verification.json", dict(passed=all(checks.values()), checks=checks,
        protected_inputs=unchanged, source_files=unchanged_sources))
    if not all(checks.values()):
        raise AssertionError(checks)
    hashes = {str(p.relative_to(root)): paths.sha256(p) for p in root.rglob("*")
        if p.is_file() and p.name != "artifacts_sha256.json"}
    save_json(root / "artifacts_sha256.json", hashes)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    verify_artifacts(args.output)
    print("Artifact, metric, source, and historical-control checks passed.")

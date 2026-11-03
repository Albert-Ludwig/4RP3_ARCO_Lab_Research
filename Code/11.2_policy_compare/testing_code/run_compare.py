"""Execute three paired objective arms without modifying the reused experiment."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import bootstrap as paths
import numpy as np
import torch
from experiment import save_csv, save_json
from gp import GPModel
from grg import information_gain
from mission import MissionController
from navigation import PythonNavigationSimulator, travel_matrix
from replay import RecordedEvaluation, setup
from objectives import FrozenCatalog, create_catalog
from planner import ObjectivePlanner

torch.set_num_threads(1)


def event_writer(path):
    def write(kind, **values):
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(dict(event=kind, timestamp_s=time.time(), **values)) + "\n")
    return write


def worker(folder, policy):
    started = time.perf_counter()
    root = folder.parent
    protocol = json.loads((root / "protocol.json").read_text())
    config, grid, locations, excluded, points, source, reference = setup()
    np.testing.assert_array_equal(travel := travel_matrix(
        nav := PythonNavigationSimulator(grid, config["mission"]["speed_m_s"], config["mission"]["step_s"]), points),
        np.load(root / "common/travel_time_s.npy"))
    assert config == protocol["baseline_config"]
    gp = GPModel.load(paths.CHECKPOINT)
    parameters = gp.parameters()
    catalog = FrozenCatalog(root / "common/objective_catalog.npz")
    evaluation = RecordedEvaluation(reference)
    field_initial = gp.predict(catalog.target)
    target_metrics = []

    def observe(model, elapsed, count):
        result = evaluation.record(model, elapsed, count)
        catalog.condition(model)
        target_metrics.append(dict(elapsed_s=elapsed, measurement_count=count,
            mean_target_variance_c2=float(catalog.current_target_variance.mean())))
        return result

    planner = ObjectivePlanner(points, travel, config["mission"]["start_h"], config["mission"]["sample_s"],
        depth=protocol["comparison_config"]["depth"], time_splits=config["grg"]["time_splits"],
        source=source, policy=policy, catalog=catalog,
        weights=protocol["comparison_config"]["fusion_weights"], event_log=event_writer(folder / "events.jsonl"))
    controller = MissionController(gp, planner, nav, source, config["mission"], observe)
    summary = controller.run()
    catalog.condition(gp)
    final_target = gp.predict(catalog.target)
    np.testing.assert_allclose(final_target["latent_var"], catalog.current_target_variance, atol=1e-7, rtol=1e-7)
    initial = GPModel.load(paths.CHECKPOINT)
    observed_query = np.array([[r.x_m, r.y_m, r.t_h] for r in controller.online]).reshape(-1, 3)
    joint = float(information_gain(initial.joint(observed_query)[1], initial.noise)) if len(observed_query) else 0.
    checks = dict(returned_home=bool(summary["returned_home"]),
        budget_respected=bool(summary["elapsed_s"] <= config["mission"]["budget_s"] + 1e-7),
        clock_accounting=bool(abs(summary["mission_time_s"] - summary["elapsed_s"]) < 1e-7),
        original_zero_computation_charge=bool(summary["online_computation_time_s"] == 0),
        unique_record_ids=len(source.used) == len(set(source.used)),
        no_heldout_record_acquired=not set(source.used).intersection(r["record_id"] for r in reference),
        parameters_frozen=all(np.array_equal(v, gp.parameters()[k]) for k, v in parameters.items()),
        successful_updates_only=gp.version == len(controller.online),
        observation_count=len(gp.x) == catalog.history_count + len(controller.online),
        all_plans_feasible=all(p["finish_s"] <= config["mission"]["budget_s"] + 1e-7 for p in controller.plans),
        chronological_measurements=bool(np.all(np.diff(observed_query[:, 2]) > 0)),
        mi_chain_rule=abs(joint - summary["information_gain_nats"]) < 1e-6,
        target_variance_nonincreasing=bool(np.max(np.diff([r["mean_target_variance_c2"] for r in target_metrics]), initial=0) <= 1e-8),
        trajectory_collision_free=all(grid.segment_free([a["x_m"], a["y_m"]], [b["x_m"], b["y_m"]])
            for a, b in zip(nav.trajectory, nav.trajectory[1:])))
    maximum_schedule_error = 0.
    for p in controller.plans:
        _, arrivals, finish = planner.schedule(p["route"], p["elapsed_s"])
        np.testing.assert_allclose(arrivals, p["arrivals_s"], atol=1e-8, rtol=0)
        maximum_schedule_error = max(maximum_schedule_error, abs(finish - p["finish_s"]))
    checks["same_planning_execution_schedule"] = maximum_schedule_error < 1e-7
    plans = []
    for p, component in zip(controller.plans, planner.components_log):
        row = dict(p)
        row["objective_score"] = row.pop("gain")
        row.update(component)
        plans.append(row)
    logs = []
    for raw in controller.log:
        row = dict(raw)
        row["route_objective_score"] = row.pop("route_gain_nats")
        row["policy"] = policy
        logs.append(row)
    save_csv(folder / "online_measurements.csv", [dict(**asdict(r), record_id=rid)
        for r, rid in zip(controller.online, source.used)])
    save_csv(folder / "robot_trajectory.csv", nav.trajectory)
    save_csv(folder / "planning_log.csv", logs)
    save_csv(folder / "metrics.csv", evaluation.rows)
    save_csv(folder / "target_metrics.csv", target_metrics)
    save_csv(folder / "objective_components.csv", planner.components_log)
    save_json(folder / "planned_routes.json", plans)
    np.savez_compressed(folder / "reference_predictions.npz", query=evaluation.query, truth=evaluation.truth,
        initial_mean=evaluation.predictions[0]["mean"], final_mean=evaluation.predictions[-1]["mean"],
        initial_latent_var=evaluation.predictions[0]["latent_var"], final_latent_var=evaluation.predictions[-1]["latent_var"],
        final_observed_var=evaluation.predictions[-1]["observed_var"])
    np.savez_compressed(folder / "target_predictions.npz", query=catalog.target,
        initial_mean=field_initial["mean"], initial_variance=field_initial["latent_var"],
        final_mean=final_target["mean"], final_variance=final_target["latent_var"])
    gp.save(folder / "final_gp.pt")
    summary.update(policy=policy, status="SUCCESS" if all(checks.values()) else "CHECK_FAILED",
        depth=protocol["comparison_config"]["depth"], initial_metrics=evaluation.rows[0],
        final_metrics=evaluation.rows[-1], initial_mean_target_variance_c2=float(field_initial["latent_var"].mean()),
        final_mean_target_variance_c2=float(final_target["latent_var"].mean()), target_count=len(catalog.target),
        target_h=protocol["catalog"]["target_h"], evaluation_count=len(reference),
        initial_gp_sha256=paths.sha256(paths.CHECKPOINT),
        initial_catalog_sha256=paths.sha256(root / "common/objective_catalog.npz"),
        measured_plan_wall_s=sum(planner.plan_wall_s), covariance_refresh_wall_s=catalog.refresh_wall_s,
        worker_wall_s=time.perf_counter() - started,
        executed_joint_mi_nats=joint, maximum_schedule_error_s=maximum_schedule_error,
        objective_units="nats" if policy == "mi" else ("C^2" if policy == "emse" else "dimensionless"))
    save_json(folder / "summary.json", summary)
    save_json(folder / "verification.json", dict(passed=all(checks.values()), checks=checks))
    print(json.dumps(summary, indent=2), flush=True)
    if not all(checks.values()):
        raise AssertionError(checks)


def supervised(policy, folder, settings):
    folder.mkdir()
    command = [sys.executable, "-B", str(Path(__file__).resolve()), "--worker", policy, "--output", str(folder)]
    started, reason = time.time(), None
    with (folder / "stdout.log").open("w", encoding="utf-8") as out, (folder / "stderr.log").open("w", encoding="utf-8") as err:
        process = subprocess.Popen(command, stdout=out, stderr=err)
        while process.poll() is None:
            rows = []
            if (folder / "events.jsonl").exists():
                for line in (folder / "events.jsonl").read_text().splitlines():
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
            active = None
            for row in rows:
                active = row if row["event"] == "plan_start" else None
            if active and time.time() - active["timestamp_s"] > settings["planning_wall_limit_s"]:
                reason = "planning_wall_limit"
            if time.time() - started > settings["mission_wall_limit_s"]:
                reason = "mission_wall_limit"
            if reason:
                process.kill()
                process.wait()
                break
            time.sleep(.1)
    outcome = dict(status="TIMEOUT" if reason else ("SUCCESS" if process.returncode == 0 else "ERROR"),
        reason=reason, returncode=process.returncode, worker_wall_s=time.time() - started)
    save_json(folder / "execution.json", outcome)
    print(policy, outcome, flush=True)
    if outcome["status"] != "SUCCESS":
        raise RuntimeError(f"{policy} did not complete; preserved logs in {folder}")


def main(output=None):
    settings = json.loads(paths.CONFIG.read_text())
    if settings["policies"] != ["mi", "emse", "fusion"]:
        raise ValueError("This paired protocol requires exactly MI, EMSE, and fusion")
    weights = settings["fusion_weights"]
    if any(v < 0 for v in weights.values()) or not np.isclose(sum(weights.values()), 1):
        raise ValueError("Invalid fusion weights")
    root = Path(output) if output else paths.ROOT / "11.2_result" / datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    root = root.resolve()
    if not root.is_relative_to(paths.ROOT / "11.2_result"):
        raise ValueError("Outputs must stay inside this experiment's 11.2_result")
    root.mkdir(parents=True, exist_ok=False)
    (root / "common").mkdir()
    protected = paths.protected_inputs()
    sources = {str(p.relative_to(paths.ROOT)): paths.sha256(p) for p in paths.HERE.glob("*.py")
               if p.name != "build_report.py"}
    sources[str(paths.CONFIG.relative_to(paths.ROOT))] = paths.sha256(paths.CONFIG)
    environment = dict(python=sys.version, platform=platform.platform(), executable=sys.executable,
        torch_threads=torch.get_num_threads(), packages={name: importlib.metadata.version(name) for name in
            ["torch", "gpytorch", "numpy", "scipy", "networkx", "shapely", "matplotlib", "nbformat", "nbclient", "ipykernel"]})
    save_json(root / "manifest.json", dict(protected_inputs_sha256=protected, source_sha256=sources, environment=environment))
    with (root / "unit_tests.log").open("w", encoding="utf-8") as stream:
        completed = subprocess.run([sys.executable, "-B", str(paths.HERE / "test_objectives.py")],
                                   stdout=stream, stderr=subprocess.STDOUT)
    save_json(root / "unit_tests.json", dict(passed=completed.returncode == 0, returncode=completed.returncode))
    if completed.returncode:
        raise AssertionError("Objective unit checks failed; see unit_tests.log")
    config, grid, locations, excluded, points, source, reference = setup()
    assert config["grg"]["depth"] == settings["depth"] == 2
    travel = travel_matrix(PythonNavigationSimulator(grid, config["mission"]["speed_m_s"], config["mission"]["step_s"]), points)
    np.save(root / "common/travel_time_s.npy", travel)
    save_csv(root / "common/candidates.csv", [dict(node=i, sensor_id=sensor, x_m=xy[0], y_m=xy[1]) for i, (sensor, xy) in enumerate(locations)])
    save_csv(root / "common/excluded_sensors.csv", excluded)
    save_csv(root / "common/evaluation_records.csv", reference)
    gp = GPModel.load(paths.CHECKPOINT)
    catalog_info = create_catalog(gp, source, points, grid, config, root / "common/objective_catalog.npz")
    save_json(root / "protocol.json", dict(comparison_config=settings, baseline_config=config, catalog=catalog_info,
        evaluation="Unchanged 28 held-out real records in the final five minutes. Retrospective reconstruction, not future forecasting. No dense measured truth.",
        policy_target="Uniform mean latent variance over the original reachable grid, at the fixed mission deadline. No measured labels enter this target.",
        normalization_weights_note="50/50 is an explicit initial hypothesis, not a validated optimum. Fixed initial single-event scales give equal relative weights, not equal physical units."))
    print("Prepared", root, catalog_info, flush=True)
    from verify_compare import verify_numerics, verify_mi_control, verify_artifacts
    verify_numerics(root)
    for policy in settings["policies"]:
        supervised(policy, root / policy, settings)
        if policy == "mi":
            verify_mi_control(root)
    verify_artifacts(root)
    save_json(paths.ROOT / "11.2_result/latest.json", dict(run=root.name))
    print("Completed all three objective arms:", root, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", choices=["mi", "emse", "fusion"])
    args = parser.parse_args()
    if args.worker:
        worker(args.output, args.worker)
    else:
        main(args.output)

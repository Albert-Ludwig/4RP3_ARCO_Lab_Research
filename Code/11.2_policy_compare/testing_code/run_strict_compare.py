"""Five preset objectives, three time windows, and frozen-origin forecasting."""

import argparse
from datetime import datetime, timezone
from functools import partial
import json
import importlib.metadata
import platform
from pathlib import Path
import subprocess
import sys
import time

import bootstrap as paths
import numpy as np
import run_compare as legacy
from experiment import save_csv, save_json
from gp import GPModel
from navigation import PythonNavigationSimulator, travel_matrix
from objectives import create_catalog
from strict_protocol import (STRICT_CONFIG, settings, setup_window, split_records,
    day4, StrictPlanner, load_records, evaluate_records)


def evaluate_model(window, policy, checkpoint):
    folder = window / policy
    folder.mkdir(exist_ok=True)
    protocol = json.loads((window / "protocol.json").read_text())
    cutoff = protocol["forecast_origin_h"]
    gp = GPModel.load(checkpoint)
    scaler = gp.fitted.scaler
    train_times = gp.x[:, 2] * scaler.x_std[2] + scaler.x_mean[2]
    assert train_times.max() <= cutoff + 1e-9
    parameters = gp.parameters()
    initial_x, initial_y = gp.x.copy(), gp.y.copy()
    metrics = []
    task_names = ["reconstruction"] + [f"forecast_{lo}_{hi}m" for lo, hi in settings()["forecast_lead_bins_min"]]
    for task in task_names:
        records = load_records(window / "common" / f"{task}_records.csv")
        if task.startswith("forecast"):
            assert all(r["t_h"] > cutoff for r in records)
        else:
            assert all(r["t_h"] <= cutoff for r in records)
        result = evaluate_records(gp, records, folder, task, cutoff)
        metrics.append(dict(window=window.name, policy=policy, **result))
    checks = dict(training_times_not_after_forecast_origin=bool(train_times.max() <= cutoff + 1e-9),
        no_forecast_conditioning=np.array_equal(gp.x, initial_x) and np.array_equal(gp.y, initial_y),
        parameters_not_changed=all(np.array_equal(v, gp.parameters()[k]) for k, v in parameters.items()))
    save_json(folder / "forecast_verification.json", dict(passed=all(checks.values()), checks=checks,
        training_count=len(gp.x), latest_training_h=float(train_times.max()), forecast_origin_h=cutoff))
    save_json(folder / "task_metrics.json", metrics)
    if not all(checks.values()):
        raise AssertionError(checks)


def worker(window, policy):
    protocol = json.loads((window / "protocol.json").read_text())
    start_h = protocol["baseline_config"]["mission"]["start_h"]
    # Dependency injection is process-local; original source files stay intact.
    legacy.setup = partial(setup_window, start_h)
    legacy.ObjectivePlanner = StrictPlanner
    legacy.worker(window / policy, policy)
    evaluate_model(window, policy, window / policy / "final_gp.pt")


def supervise(window, policy, config):
    folder = window / policy
    folder.mkdir()
    command = [sys.executable, "-B", str(Path(__file__).resolve()), "--worker", policy, "--window", str(window)]
    begin, reason = time.time(), None
    with (folder / "stdout.log").open("w") as out, (folder / "stderr.log").open("w") as err:
        process = subprocess.Popen(command, stdout=out, stderr=err)
        while process.poll() is None:
            active = None
            if (folder / "events.jsonl").exists():
                for line in (folder / "events.jsonl").read_text().splitlines():
                    try:
                        event = json.loads(line)
                        active = event if event["event"] == "plan_start" else None
                    except json.JSONDecodeError:
                        pass
            if active and time.time() - active["timestamp_s"] > config["planning_wall_limit_s"]:
                reason = "planning_wall_limit"
            if time.time() - begin > config["mission_wall_limit_s"]:
                reason = "mission_wall_limit"
            if reason:
                process.kill()
                process.wait()
                break
            time.sleep(.1)
    outcome = dict(status="TIMEOUT" if reason else ("SUCCESS" if process.returncode == 0 else "ERROR"),
        reason=reason, returncode=process.returncode, wall_s=time.time() - begin)
    save_json(folder / "execution.json", outcome)
    print(window.name, policy, outcome["status"], round(outcome["wall_s"], 2), "s", flush=True)
    if outcome["status"] != "SUCCESS":
        raise RuntimeError(f"Inspect preserved logs in {folder}")


def prepare_window(root, start_h, config):
    window = root / "windows" / f"w{int(start_h - 72):02d}"
    (window / "common").mkdir(parents=True)
    baseline, grid, locations, excluded, points, source, reference = setup_window(start_h)
    cutoff = start_h + baseline["mission"]["budget_s"] / 3600
    _, reconstruction, future = split_records(day4(), {s for s, _ in locations}, start_h,
        baseline["mission"]["budget_s"], config["forecast_lead_bins_min"], config["reconstruction_window_min"])
    assert reference == reconstruction
    save_csv(window / "common/reconstruction_records.csv", reference)
    save_csv(window / "common/evaluation_records.csv", reference)
    for task, records in future.items():
        save_csv(window / "common" / f"{task}_records.csv", records)
    save_csv(window / "common/candidates.csv", [dict(node=i, sensor_id=s, x_m=xy[0], y_m=xy[1]) for i, (s, xy) in enumerate(locations)])
    save_csv(window / "common/excluded_sensors.csv", excluded)
    nav = PythonNavigationSimulator(grid, baseline["mission"]["speed_m_s"], baseline["mission"]["step_s"])
    np.save(window / "common/travel_time_s.npy", travel_matrix(nav, points))
    catalog = create_catalog(GPModel.load(paths.CHECKPOINT), source, points, grid, baseline,
                             window / "common/objective_catalog.npz")
    save_json(window / "protocol.json", dict(baseline_config=baseline,
        comparison_config=dict(depth=config["depth"], fusion_weights={"mi": .5, "emse_reduction": .5}),
        weights_mi_emse=config["weights_mi_emse"], catalog=catalog, forecast_origin_h=cutoff,
        reconstruction="Latest held-out actual record per reachable sensor in the final five minutes; retrospective window reconstruction, not simultaneous dense truth.",
        forecasting="Frozen final GP; only actual held-out records strictly after the common deadline; no additional conditioning.",
        future_counts={task: len(records) for task, records in future.items()}))
    print("Prepared", window.name, "cutoff", cutoff, "future records", {k: len(v) for k, v in future.items()}, flush=True)
    evaluate_model(window, "history_only", paths.CHECKPOINT)
    from verify_compare import verify_numerics
    verify_numerics(window)
    return window


def main(output=None):
    config = settings()
    root = (Path(output) if output else paths.ROOT / "11.2_result" /
            datetime.now(timezone.utc).strftime("strict_run_%Y%m%dT%H%M%SZ")).resolve()
    if not root.is_relative_to(paths.ROOT / "11.2_result"):
        raise ValueError("Output must remain inside this experiment")
    root.mkdir(parents=True, exist_ok=False)
    protected = paths.protected_inputs()
    prior = paths.ROOT / "11.2_result" / json.loads((paths.ROOT / "11.2_result/latest.json").read_text())["run"]
    # Reports are regenerated at one canonical location; protect raw runs instead.
    for p in list(prior.rglob("*")) + [paths.ROOT / "11.2_result/latest.json"]:
        if p.is_file():
            protected[str(p.relative_to(paths.WORKSPACE))] = paths.sha256(p)
    sources = {str(p.relative_to(paths.ROOT)): paths.sha256(p) for p in paths.HERE.glob("*.py")
               if not p.name.startswith("build_")}
    sources[str(STRICT_CONFIG.relative_to(paths.ROOT))] = paths.sha256(STRICT_CONFIG)
    save_json(root / "manifest.json", dict(protected_inputs_sha256=protected, source_sha256=sources,
        prior_three_policy_run=str(prior.relative_to(paths.ROOT)),
        historical_checkpoint_sha256=paths.sha256(paths.CHECKPOINT),
        environment=dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
            packages={name: importlib.metadata.version(name) for name in
                ["torch", "gpytorch", "numpy", "scipy", "networkx", "shapely", "matplotlib", "nbformat", "nbclient"]}),
        selection_note="The five ratios and three windows are preset. Results are exploratory; no best ratio is selected and revalidated on the same labels."))
    save_json(root / "config_frozen.json", config)
    for test_file in ["test_objectives.py", "test_strict_protocol.py"]:
        with (root / f"{Path(test_file).stem}.log").open("w") as stream:
            outcome = subprocess.run([sys.executable, "-B", str(paths.HERE / test_file)], stdout=stream, stderr=subprocess.STDOUT)
        if outcome.returncode:
            raise AssertionError(f"Read {test_file} test log")
    save_json(root / "unit_tests.json", dict(passed=True, suites=["test_objectives.py", "test_strict_protocol.py"]))
    for start_h in config["window_start_hours"]:
        window = prepare_window(root, start_h, config)
        for policy in config["policies"]:
            supervise(window, policy, config)
    from verify_strict import verify
    verify(root)
    save_json(paths.ROOT / "11.2_result/latest_strict.json", dict(run=root.name))
    print("Completed strict comparison:", root, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", choices=settings()["policies"])
    parser.add_argument("--window", type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(args.window, args.worker)
    else:
        main(args.output)

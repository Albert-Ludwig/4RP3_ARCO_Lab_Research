"""Isolated runs using the unchanged baseline GP, GRG and mission controller."""

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "testing_code" / "baseline"
RESULT = ROOT / "10.18_result" / "grg_depth_benchmark"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "10.18_result" / ".cache" / "matplotlib"))
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
sys.path.insert(0, str(BASE))

import numpy as np
import torch

from evaluate import Evaluation
from experiment import save_csv, save_json
from gp import GPModel, arrays
from grg import GRGPlanner, information_gain
from interfaces import Measurement
from mission import MissionController
from navigation import GridMap, PythonNavigationSimulator, travel_matrix
from temperature import SyntheticTemperatureSource

torch.set_num_threads(1)


def read_json(path):
    return json.loads(path.read_text())


def read_history(path):
    with path.open(newline="") as file:
        return [Measurement(float(r["x_m"]), float(r["y_m"]), float(r["t_h"]),
                            float(r["temperature_c"]), r["source"], r["status"])
                for r in csv.DictReader(file)]


def peak_mb():
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return usage / (1024**2 if sys.platform == "darwin" else 1024)


def event(path, kind, **data):
    with path.open("a") as file:
        file.write(json.dumps({"event": kind, "wall_s": time.perf_counter(), **data}) + "\n")
        file.flush()


def prepare(seed):
    folder = RESULT / "shared" / f"seed_{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "manifest.json").is_file():
        manifest = read_json(folder / "manifest.json")
        assert all(hashlib.sha256((folder / name).read_bytes()).hexdigest() == digest
                   for name, digest in manifest.items())
        return
    config = read_json(BASE / "config.json")
    config["seed"] = seed
    grid = GridMap(config["map"])
    c = config["mission"]
    points = grid.candidates(c["candidate_count"], seed)
    nav = PythonNavigationSimulator(grid, c["speed_m_s"], c["step_s"])
    travel = travel_matrix(nav, points)
    rng = np.random.default_rng(seed + 100)
    count = config["history"]["count"]
    xy = grid.free_xy[rng.choice(len(grid.free_xy), count, replace=False)]
    x = np.column_stack([xy, rng.uniform(*config["history"]["time_h"], count)])
    sensor = SyntheticTemperatureSource(x, config["history"]["noise_std_c"],
                                        config["sensor_noise_std_c"], seed)
    history = sensor.load_history()
    original = ROOT / "10.18_result" / "baseline" / ("main" if seed == 18 else f"seed_{seed}")
    if (original / "initial_gp.pt").is_file():
        gp = GPModel.load(original / "initial_gp.pt")
        tx, ty = arrays(history)
        np.testing.assert_array_equal(gp.x, gp.fitted.scaler.transform_x(tx))
        np.testing.assert_array_equal(gp.y, gp.fitted.scaler.transform_y(ty))
        shutil.copy2(original / "initial_gp.pt", folder / "initial_gp.pt")
        origin = "loaded existing baseline checkpoint; no retraining"
    else:
        gp = GPModel.train(history, c["start_h"], config["gp"], seed)
        gp.save(folder / "initial_gp.pt")
        origin = "trained once for this seed using unchanged baseline settings"
    evaluation = Evaluation(grid, config["evaluation"]["resolution_m"],
                            c["start_h"] + c["budget_s"] / 3600)
    evaluation.record(gp, 0, 0)
    evaluation.save(folder / "initial_reconstruction.npz")
    save_json(folder / "config.json", config)
    save_json(folder / "gp_training_summary.json", gp.fitted.hyperparameters())
    save_json(folder / "initial_state.json", {"seed": seed, "origin": origin,
                                              "observations": len(gp.x), "online_count": 0,
                                              "elapsed_s": 0,
                                              "initial_rmse_c": evaluation.rows[0]["rmse_c"]})
    save_csv(folder / "historical_measurements.csv", [asdict(r) for r in history])
    np.save(folder / "candidates.npy", points)
    np.save(folder / "travel_time_s.npy", travel)
    save_json(folder / "manifest.json", {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(folder.iterdir()) if p.is_file() and p.name != "manifest.json"})


class LoggedPlanner(GRGPlanner):
    def __init__(self, *args, folder, **kwargs):
        super().__init__(*args, **kwargs)
        self.folder, self.calls = folder, []

    def plan(self, gp, current, elapsed, remaining, visited, blocked=()):
        index = len(self.calls)
        event(self.folder / "events.jsonl", "plan_start", call=index,
              elapsed_s=elapsed, gp_version=gp.version, remaining_s=remaining)
        plan = super().plan(gp, current, elapsed, remaining, visited, blocked)
        samples = [v for v in plan.route[1:] if v != self.home]
        row = {"call": index, "elapsed_s": elapsed, **asdict(plan),
               "sampling_waypoints": len(samples), "unique_sampling_nodes": len(set(samples)),
               "repeated_planned_visits": len(samples) - len(set(samples)),
               "planned_travel_s": float(sum(self.travel[a, b]
                                             for a, b in zip(plan.route[:-1], plan.route[1:]))),
               "budget_feasible": bool(plan.route and plan.finish_s <= elapsed + remaining + 1e-7),
               "peak_memory_mb": peak_mb()}
        self.calls.append(row)
        event(self.folder / "events.jsonl", "plan_end", **row)
        save_csv(self.folder / "planning_runtime.csv", self.calls)
        return plan


def execute(seed, depth, mode, folder):
    folder.mkdir(parents=True, exist_ok=True)
    shared = RESULT / "shared" / f"seed_{seed}"
    config = read_json(shared / "config.json")
    config["grg"]["depth"] = depth
    c = config["mission"]
    gp = GPModel.load(shared / "initial_gp.pt")
    parameters, scaler = gp.parameters(), gp.fitted.scaler.as_dict()
    history = read_history(shared / "historical_measurements.csv")
    points, travel = np.load(shared / "candidates.npy"), np.load(shared / "travel_time_s.npy")
    grid = GridMap(config["map"])
    np.testing.assert_array_equal(grid.candidates(c["candidate_count"], seed), points)
    initial_x, _ = arrays(history)
    sensor = SyntheticTemperatureSource(initial_x, config["history"]["noise_std_c"],
                                        config["sensor_noise_std_c"], seed)
    np.testing.assert_array_equal(arrays(sensor.load_history())[1], arrays(history)[1])
    nav = PythonNavigationSimulator(grid, c["speed_m_s"], c["step_s"])
    evaluation = Evaluation(grid, config["evaluation"]["resolution_m"],
                            c["start_h"] + c["budget_s"] / 3600)
    initial = np.load(shared / "initial_reconstruction.npz")
    np.testing.assert_array_equal(evaluation.query, initial["query"])
    planner = LoggedPlanner(points, travel, c["start_h"], c["sample_s"], depth,
                            config["grg"]["time_splits"], allow_revisit=c["allow_revisit"], folder=folder)
    save_json(folder / "config.json", config)
    shutil.copy2(shared / "initial_gp.pt", folder / "initial_gp.pt")
    initial_hash = hashlib.sha256((folder / "initial_gp.pt").read_bytes()).hexdigest()
    event(folder / "events.jsonl", "worker_ready", seed=seed, depth=depth,
          initial_gp_sha256=initial_hash, observations=len(gp.x), elapsed_s=0)
    if mode == "screen":
        plan = planner.plan(gp, 0, 0, c["budget_s"], set())
        save_json(folder / "summary.json", {"seed": seed, "depth": depth, "status": "SUCCESS",
                                            **planner.calls[0]})
        return

    def observe(model, elapsed, count):
        metrics = evaluation.record(model, elapsed, count)
        save_csv(folder / "metrics.csv", evaluation.rows)
        save_csv(folder / "online_measurements.csv", [asdict(r) for r in controller.online])
        save_csv(folder / "robot_trajectory.csv", nav.trajectory)
        save_csv(folder / "planning_log.csv", controller.log)
        save_json(folder / "partial_state.json", {"elapsed_s": elapsed, "measurements": count,
                                                   "gp_version": gp.version, **metrics})
        return metrics

    controller = MissionController(gp, planner, nav, sensor, c, observe)
    started = time.perf_counter()
    summary = controller.run()
    mission_wall = time.perf_counter() - started
    observe(gp, summary["elapsed_s"], len(controller.online))
    gp.save(folder / "final_gp.pt")
    evaluation.save(folder / "reconstruction.npz")
    save_json(folder / "planned_routes.json", controller.plans)
    saved_initial = GPModel.load(shared / "initial_gp.pt")
    if controller.online:
        q, _ = arrays(controller.online)
        realized_gain = float(information_gain(saved_initial.joint(q)[1], saved_initial.noise))
    else:
        realized_gain = 0.0
    sampled = [r for r in controller.log if r["gp_updated"]]
    runtimes = np.array([p["runtime_s"] for p in planner.calls])
    first, last = evaluation.rows[0], evaluation.rows[-1]
    checks = {
        "initial_predictions_identical": bool(np.array_equal(evaluation.means[0], initial["means"][0])
                                               and np.array_equal(evaluation.variances[0], initial["variances"][0])),
        "fixed_parameters": all(np.array_equal(v, gp.parameters()[k]) for k, v in parameters.items()),
        "fixed_scaler": scaler == gp.fitted.scaler.as_dict(),
        "all_observations_retained": len(gp.x) == len(history) + len(controller.online),
        "variance_nonincreasing": bool(np.max(np.diff(evaluation.variances, axis=0)) <= 1e-9),
        "actual_arrival_measurements": all(np.allclose([m.x_m, m.y_m, m.t_h],
                                                        [r["arrival_x_m"], r["arrival_y_m"], r["arrival_t_h"]],
                                                        atol=1e-10, rtol=0)
                                           for m, r in zip(controller.online, sampled)),
        "replan_after_updates": all(any(p["gp_version"] == i for p in controller.plans)
                                    for i in range(1, len(controller.online) + 1)),
        "all_routes_budget_feasible": all(r["budget_feasible"] for r in planner.calls),
        "collision_free": all(grid.segment_free([a["x_m"], a["y_m"]], [b["x_m"], b["y_m"]])
                              for a, b in zip(nav.trajectory[:-1], nav.trajectory[1:])),
        "mission_budget": summary["elapsed_s"] <= c["budget_s"] + 1e-7,
        "returned_home": summary["returned_home"],
        "at_least_three_samples": len(controller.online) >= 3,
        "gain_chain_rule": abs(realized_gain - summary["information_gain_nats"]) < 1e-6,
    }
    summary.update(seed=seed, depth=depth, status="SUCCESS" if all(checks.values()) else "FAILED",
                   initial_gp_sha256=initial_hash, initial_rmse_c=first["rmse_c"], final_rmse_c=last["rmse_c"],
                   rmse_reduction_c=first["rmse_c"] - last["rmse_c"],
                   rmse_reduction_pct=100 * (1 - last["rmse_c"] / first["rmse_c"]),
                   initial_variance_c2=first["mean_variance_c2"], final_variance_c2=last["mean_variance_c2"],
                   variance_reduction_pct=100 * (1 - last["mean_variance_c2"] / first["mean_variance_c2"]),
                   final_coverage95=last["coverage_95"], realized_information_gain_nats=realized_gain,
                   sampling_time_s=len(controller.online) * c["sample_s"],
                   remaining_budget_s=c["budget_s"] - summary["elapsed_s"],
                   navigation_failures=sum(r.get("status") == "navigation_failed" for r in controller.log),
                   mean_planning_s=float(runtimes.mean()), median_planning_s=float(np.median(runtimes)),
                   max_planning_s=float(runtimes.max()), p95_planning_s=float(np.percentile(runtimes, 95)),
                   total_planning_s=float(runtimes.sum()), mission_wall_s=mission_wall,
                   peak_memory_mb=peak_mb(), target_h=float(evaluation.query[0, 2]),
                   evaluation_count=len(evaluation.query),
                   max_planned_waypoints=max(r["sampling_waypoints"] for r in planner.calls),
                   repeated_planned_visits=sum(r["repeated_planned_visits"] for r in planner.calls))
    checks = {name: bool(value) for name, value in checks.items()}
    save_json(folder / "verification.json", {"passed": all(checks.values()), "checks": checks})
    save_json(folder / "summary.json", summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prepare", "screen", "mission"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.seed)
    else:
        execute(args.seed, args.depth, args.mode, args.output)

"""Run one configured mission; write every artifact below 10.18_result."""

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "10.18_result" / ".cache" / "matplotlib"))
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np

from evaluate import Evaluation
from gp import GPModel
from grg import GRGPlanner, information_gain
from mission import MissionController
from navigation import GridMap, PythonNavigationSimulator, travel_matrix
from plots import save_figures
from temperature import SyntheticTemperatureSource


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def save_csv(path, rows):
    if not rows:
        path.write_text("")
        return
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()})


def run_experiment(config, output, figures=True):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    seed, mission = config["seed"], config["mission"]
    map_started = time.perf_counter()
    grid = GridMap(config["map"])
    points = grid.candidates(mission["candidate_count"], seed)
    nav = PythonNavigationSimulator(grid, mission["speed_m_s"], mission["step_s"])
    travel = travel_matrix(nav, points)
    map_s = time.perf_counter() - map_started
    rng = np.random.default_rng(seed + 100)
    count = config["history"]["count"]
    xy = grid.free_xy[rng.choice(len(grid.free_xy), count, replace=False)]
    times = rng.uniform(*config["history"]["time_h"], count)
    sensor = SyntheticTemperatureSource(np.column_stack([xy, times]),
                                        config["history"]["noise_std_c"],
                                        config["sensor_noise_std_c"], seed)
    history = sensor.load_history()
    gp = GPModel.train(history, mission["start_h"], config["gp"], seed)
    gp.save(output / "initial_gp.pt")
    parameters = gp.parameters()
    target = mission["start_h"] + mission["budget_s"] / 3600
    evaluation = Evaluation(grid, config["evaluation"]["resolution_m"], target)
    planner = GRGPlanner(points, travel, mission["start_h"], mission["sample_s"],
                         config["grg"]["depth"], config["grg"]["time_splits"],
                         allow_revisit=mission["allow_revisit"])
    controller = MissionController(gp, planner, nav, sensor, mission, evaluation.record)
    summary = controller.run()
    evaluation.record(gp, summary["elapsed_s"], len(controller.online))
    gp.save(output / "final_gp.pt")
    summary.update(seed=seed, target_h=target, historical_count=count,
                   candidate_count=len(points), evaluation_count=len(evaluation.query),
                   initial_rmse_c=evaluation.rows[0]["rmse_c"],
                   final_rmse_c=evaluation.rows[-1]["rmse_c"],
                   initial_variance_c2=evaluation.rows[0]["mean_variance_c2"],
                   final_variance_c2=evaluation.rows[-1]["mean_variance_c2"],
                   initial_coverage95=evaluation.rows[0]["coverage_95"],
                   final_coverage95=evaluation.rows[-1]["coverage_95"],
                   map_and_travel_precompute_s=map_s,
                   gp_training_s=gp.fitted.fit_seconds,
                   grg_covariance_query_s=gp.query_s,
                   covariance_batches=gp.query_batches,
                   simulation_wall_s=time.perf_counter() - started)
    sampled = [r for r in controller.log if r["gp_updated"]]
    initial_gp = GPModel.load(output / "initial_gp.pt")
    if controller.online:
        q = np.array([[r.x_m, r.y_m, r.t_h] for r in controller.online])
        joint_gain = float(information_gain(initial_gp.joint(q)[1], initial_gp.noise))
    else:
        joint_gain = 0.0
    checks = {
        "history_no_future": all(r.t_h <= mission["start_h"] for r in history),
        "at_least_three_samples": len(controller.online) >= 3,
        "all_observations_retained": len(gp.x) == count + len(controller.online),
        "parameters_fixed": all(np.array_equal(v, gp.parameters()[k]) for k, v in parameters.items()),
        "variance_nonincreasing": bool(np.max(np.diff(evaluation.variances, axis=0)) <= 1e-9),
        "budget_respected": summary["elapsed_s"] <= mission["budget_s"] + 1e-7,
        "returned_home": summary["returned_home"],
        "trajectory_collision_free": all(grid.segment_free([a["x_m"], a["y_m"]],
                                                            [b["x_m"], b["y_m"]])
                                         for a, b in zip(nav.trajectory[:-1], nav.trajectory[1:])),
        "all_plans_feasible": all(p["finish_s"] <= mission["budget_s"] + 1e-7 for p in controller.plans),
        "replan_after_every_measurement": all(any(p["gp_version"] == i for p in controller.plans)
                                               for i in range(1, len(controller.online) + 1)),
        "actual_arrivals_used": all(np.allclose([m.x_m, m.y_m, m.t_h],
                                               [r["arrival_x_m"], r["arrival_y_m"], r["arrival_t_h"]],
                                               atol=1e-10, rtol=0)
                                    for m, r in zip(controller.online, sampled)),
        "travel_estimate_matches_execution": all(abs(r["estimated_travel_s"] - r["actual_travel_s"]) < 1e-7
                                                 for r in sampled),
        "gain_chain_rule": abs(joint_gain - summary["information_gain_nats"]) < 1e-6,
        "multiple_gp_updates": gp.version == len(controller.online) >= 3,
    }
    checks = {key: bool(value) for key, value in checks.items()}
    save_json(output / "config.json", config)
    save_json(output / "mission_summary.json", summary)
    save_json(output / "verification.json", {"passed": all(checks.values()), "checks": checks})
    save_json(output / "gp_training_summary.json", gp.fitted.hyperparameters())
    save_json(output / "planned_routes.json", controller.plans)
    save_json(output / "map_summary.json", {"obstacle_areas_m2": [p.area for p in grid.obstacles],
                                            "footprint_radius_with_margin_m": grid.radius,
                                            "reachable_grid_nodes": len(grid.graph),
                                            "edge_count": grid.graph.number_of_edges()})
    save_csv(output / "historical_measurements.csv", [asdict(r) for r in history])
    save_csv(output / "online_measurements.csv", [asdict(r) for r in controller.online])
    save_csv(output / "robot_trajectory.csv", nav.trajectory)
    save_csv(output / "planning_log.csv", controller.log)
    save_csv(output / "metrics.csv", evaluation.rows)
    save_csv(output / "training_history.csv", gp.fitted.history)
    save_csv(output / "candidates.csv", [{"node": i, "x_m": p[0], "y_m": p[1]} for i, p in enumerate(points)])
    np.save(output / "travel_time_s.npy", travel)
    evaluation.save(output / "reconstruction.npz")
    if figures:
        save_figures(output / "figures", grid, points, nav, controller, evaluation, summary)
    summary["total_wall_s"] = time.perf_counter() - started
    save_json(output / "mission_summary.json", summary)
    manifest = {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(output.rglob("*")) if p.is_file() and p.name != "manifest.json"}
    save_json(output / "manifest.json", manifest)
    if not all(checks.values()):
        raise AssertionError({k: v for k, v in checks.items() if not v})
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--output", type=Path, default=ROOT / "10.18_result" / "baseline" / "current" / "main")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.seed is not None:
        config["seed"] = args.seed
    if args.debug:
        config["mission"].update(candidate_count=12, budget_s=300)
        config["gp"]["iterations"] = 80
    print(json.dumps(run_experiment(config, args.output, not args.no_figures), indent=2))

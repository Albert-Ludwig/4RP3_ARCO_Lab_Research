"""Recorded sensor backend and timestamp-aligned GRG; no temperature interpolation."""

from collections import defaultdict
from dataclasses import asdict
import json

from data import BASE, ROOT, load_split, measurements
from interfaces import Measurement
from grg import GRGPlanner, information_gain
from navigation import GridMap, PythonNavigationSimulator, travel_matrix
from gp import GPModel
from experiment import save_csv, save_json
from mission import MissionController
from evaluation import distribution_metrics

import numpy as np


class IntelSource:
    def __init__(self, rows, start_h=72):
        groups = defaultdict(list)
        for row in rows:
            groups[(row["x_m"], row["y_m"])].append(row)
        self.rows = {xy: sorted(group, key=lambda r: r["t_h"]) for xy, group in groups.items()}
        self.times = {xy: np.array([r["t_h"] for r in group]) for xy, group in self.rows.items()}
        self.start_h = start_h
        self.used = []

    def next_time(self, position, timestamp_h):
        times = self.times.get(tuple(position), np.array([]))
        index = np.searchsorted(times, timestamp_h - 1e-12)
        return float(times[index]) if index < len(times) else np.inf

    def measure(self, position, timestamp_h):
        xy = tuple(position)
        times = self.times.get(xy, np.array([]))
        index = np.searchsorted(times, timestamp_h - 1e-12)
        if index >= len(times) or abs(times[index] - timestamp_h) > 1e-10:
            return Measurement(*position, timestamp_h, np.nan, "intel_lab", "missing")
        row = self.rows[xy][index]
        if row["record_id"] in self.used:
            raise AssertionError("The same recorded observation cannot be acquired twice")
        self.used.append(row["record_id"])
        return measurements([row])[0]


class IntelPlanner(GRGPlanner):
    def __init__(self, *args, source, **kwargs):
        self.record_times = source.times  # public availability only; no temperatures
        super().__init__(*args, sampling_time=self.next_sample_s, **kwargs)

    def next_sample_s(self, node, elapsed):
        if not np.isfinite(elapsed):
            return np.inf
        times = self.record_times.get(tuple(self.points[node]), np.array([]))
        index = np.searchsorted(times, self.start_h + elapsed / 3600 - 1e-12)
        return float((times[index] - self.start_h) * 3600) if index < len(times) else np.inf


def setup():
    config = json.loads((BASE / "config.json").read_text())
    config["map"]["home_m"] = [1.5, 2.0]  # Sensor 16; nearest to the old Home.
    config["mission"]["start_h"] = 72.
    _, test = load_split()
    grid = GridMap(config["map"])
    by_sensor = {r["sensor_id"]: r for r in test}
    locations, excluded = [], []
    for sensor, row in sorted(by_sensor.items()):
        xy = np.array([row["x_m"], row["y_m"]])
        if grid.node(xy) in grid.graph:
            locations.append((sensor, xy))
        else:
            excluded.append(dict(sensor_id=sensor, x_m=row["x_m"], y_m=row["y_m"],
                                 reason="outside_current_map_or_not_footprint_reachable"))
    locations.sort(key=lambda item: (item[0] != 16, item[0]))
    assert locations[0][0] == 16
    points = np.array([xy for _, xy in locations])
    config["mission"]["candidate_count"] = len(points)
    # Split only by record index/availability, not temperature. Odd records
    # form an independent reference pool and are never sampled by the robot.
    groups = defaultdict(list)
    for row in test:
        groups[row["sensor_id"]].append(row)
    available, heldout = [], []
    for sensor in sorted(groups):
        ordered = sorted(groups[sensor], key=lambda r: (r["t_h"], r["record_id"]))
        available.extend(ordered[::2]); heldout.extend(ordered[1::2])
    source = IntelSource(available)
    # A real-data evaluation target consists of actual held-out observations
    # immediately before the 72.25 h mission target, not invented grid truth.
    target = 72 + config["mission"]["budget_s"] / 3600
    evaluation = []
    for sensor, xy in locations:
        choices = [r for r in heldout if r["sensor_id"] == sensor and target - 5/60 <= r["t_h"] <= target]
        if choices:
            evaluation.append(max(choices, key=lambda r: r["t_h"]))
    return config, grid, locations, excluded, points, source, evaluation


class RecordedEvaluation:
    def __init__(self, records):
        self.records = records
        self.query = np.array([[r["x_m"], r["y_m"], r["t_h"]] for r in records])
        self.truth = np.array([r["temperature_c"] for r in records])
        self.rows, self.predictions = [], []

    def record(self, gp, elapsed, count):
        pred = gp.predict(self.query)
        row = dict(elapsed_s=elapsed, measurement_count=count,
                   **distribution_metrics(self.truth, pred["mean"], pred["observed_var"]),
                   mean_variance_c2=float(pred["latent_var"].mean()))
        self.rows.append(row); self.predictions.append(pred)
        return {"rmse_c": row["rmse_c"], "mean_variance_c2": row["mean_variance_c2"]}


def execute(folder, depth=2, screen=False, planner_class=IntelPlanner):
    folder.mkdir(parents=True, exist_ok=True)
    config, grid, locations, excluded, points, source, reference = setup()
    config["grg"]["depth"] = depth
    c = config["mission"]
    nav = PythonNavigationSimulator(grid, c["speed_m_s"], c["step_s"])
    travel = travel_matrix(nav, points)
    checkpoint = ROOT / "10.18_result/baseline/intel_lab/forecast/initial_gp.pt"
    gp = GPModel.load(checkpoint)
    parameters = gp.parameters()
    planner = planner_class(points, travel, c["start_h"], c["sample_s"], depth=depth,
                            time_splits=config["grg"]["time_splits"], source=source)
    save_json(folder / "config.json", config)
    save_csv(folder / "candidates.csv", [dict(node=i, sensor_id=sensor, x_m=xy[0], y_m=xy[1])
                                         for i, (sensor, xy) in enumerate(locations)])
    save_csv(folder / "excluded_sensors.csv", excluded)
    save_csv(folder / "evaluation_records.csv", reference)
    if screen:
        plan = planner.plan(gp, 0, 0, c["budget_s"])
        save_json(folder / "summary.json", dict(status="SUCCESS", depth=depth, **asdict(plan)))
        return
    evaluation = RecordedEvaluation(reference)
    controller = MissionController(gp, planner, nav, source, c, evaluation.record)
    summary = controller.run()
    gp.save(folder / "final_gp.pt")
    save_csv(folder / "online_measurements.csv", [dict(**asdict(r), record_id=rid)
              for r, rid in zip(controller.online, source.used)])
    save_csv(folder / "planning_log.csv", controller.log)
    save_csv(folder / "robot_trajectory.csv", nav.trajectory)
    save_csv(folder / "metrics.csv", evaluation.rows)
    save_json(folder / "planned_routes.json", controller.plans)
    initial = GPModel.load(checkpoint)
    q = np.array([[r.x_m, r.y_m, r.t_h] for r in controller.online])
    joint = float(information_gain(initial.joint(q)[1], initial.noise)) if len(q) else 0.
    errors = []
    for plan in controller.plans:
        _, arrivals, finish = planner.schedule(plan["route"], plan["elapsed_s"])
        errors.append(abs(finish - plan["finish_s"]))
        np.testing.assert_allclose(arrivals, plan["arrivals_s"], atol=1e-8)
    checks = dict(returned_home=summary["returned_home"],
                  within_budget=summary["elapsed_s"] <= c["budget_s"] + 1e-7,
                  clock_accounting=abs(summary["mission_time_s"] - summary["elapsed_s"]) < 1e-7,
                  unique_record_ids=len(source.used) == len(set(source.used)),
                  no_evaluation_leakage=not set(source.used).intersection(r["record_id"] for r in reference),
                  gp_observation_count=len(gp.x) == 2000 + len(controller.online),
                  only_successful_gp_updates=gp.version == len(controller.online),
                  parameters_frozen=all(np.array_equal(v, gp.parameters()[k]) for k,v in parameters.items()),
                  all_plans_feasible=all(p["finish_s"] <= c["budget_s"] + 1e-7 for p in controller.plans),
                  reward_chain_rule=abs(joint - summary["information_gain_nats"]) < 1e-6,
                  chronological_measurements=bool(np.all(np.diff(q[:,2]) > 0)) if len(q) else True,
                  path_collision_free=all(grid.segment_free([a["x_m"],a["y_m"]],[b["x_m"],b["y_m"]])
                    for a,b in zip(nav.trajectory,nav.trajectory[1:])))
    checks = {name: bool(value) for name, value in checks.items()}
    np.savez_compressed(folder / "evaluation.npz", query=evaluation.query, truth=evaluation.truth,
                        initial_mean=evaluation.predictions[0]["mean"], final_mean=evaluation.predictions[-1]["mean"],
                        initial_latent_var=evaluation.predictions[0]["latent_var"],
                        final_latent_var=evaluation.predictions[-1]["latent_var"])
    summary.update(status="SUCCESS" if all(checks.values()) else "CHECK_FAILED", depth=depth,
                   train_count=2000, candidate_count=len(points), evaluation_count=len(reference),
                   evaluation_time_range_h=[float(evaluation.query[:,2].min()),float(evaluation.query[:,2].max())],
                   initial_metrics=evaluation.rows[0], final_metrics=evaluation.rows[-1],
                   joint_mi_nats=joint, maximum_plan_schedule_error_s=max(errors, default=0.))
    save_json(folder / "summary.json", summary)
    save_json(folder / "verification.json", dict(passed=all(checks.values()), checks=checks))
    print(json.dumps(summary, indent=2), flush=True)
    assert all(checks.values()), checks
    return summary


if __name__ == "__main__":
    execute(ROOT / "10.18_result/baseline/intel_lab/replay")

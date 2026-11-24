"""Fit and validate a per-leg Nav2 time estimator from saved ROS runs."""

from pathlib import Path
import csv
import hashlib
import json
import re

import numpy as np

RUNS = {
    "MI_Test": (
        "run_20261123T204448Z", "run_20261123T205840Z", "run_20261123T213446Z",
        "run_20261123T220640Z", "run_20261123T223950Z",
    ),
    "EMSE_Test": (
        "run_20261123T211629Z", "run_20261123T214029Z", "run_20261123T221211Z",
        "run_20261123T224551Z",
    ),
    "WIVR_test": (
        "run_20261120T145549Z", "run_20261123T210449Z", "run_20261123T214611Z",
        "run_20261123T221750Z", "run_20261123T225159Z",
    ),
}
# The last listed run from each policy is a frozen holdout, independent of latest.json.
SHORT_LEG_M = 10.0


def read_json(path):
    return json.loads(path.read_text(encoding="utf8"))


def shared_nav_settings(case):
    params = (case / "nav2_params.yaml").read_text(encoding="utf8")
    params = re.sub(r"^\s*yaml_filename:.*$", "", params, flags=re.M)
    return params, read_json(case / "navigation_profile.json")


def extract_case(case, policy, split):
    graph = {(edge["from"], edge["to"]): edge for edge in read_json(case / "graph.json")}
    points = read_json(case / "candidates.json")
    events = [json.loads(line) for line in (case / "events.jsonl").read_text(encoding="utf8").splitlines()]
    arrivals = {event["goal_epoch"]: event for event in events if event["event"] == "arrived"}
    ended = {event["goal_epoch"] for event in events if event["event"] == "sampling_goal_ended"}
    run = case.parent.name
    state = read_json(case / "mission.json")
    if state["status"] != "window_completed" or state["navigation_failures"]:
        raise ValueError(f"Run {run} has a navigation or mission failure")
    config = read_json(case / "protocol.json")
    speed = config["travel_estimator_speed_m_s"]
    allowance = config["travel_estimator_segment_allowance_s"]
    trajectory = np.load(case / "trajectory.npy", allow_pickle=False)
    times = trajectory[:, 0]
    steps = np.diff(times)
    if not np.isfinite(trajectory).all() or not len(steps) or not np.all(steps > 0):
        raise ValueError(f"Invalid trajectory for {run}")
    cumulative_distance = np.r_[0, np.linalg.norm(np.diff(trajectory[:, 1:3], axis=0), axis=1).cumsum()]
    sample_interval = float(np.median(steps))
    current = 0
    legs, excluded = [], []
    for goal in (e for e in events if e["event"] == "nav_goal"):
        epoch = goal["goal_epoch"]
        target = goal["target"]
        arrival = arrivals.get(epoch)
        if arrival is not None and arrival["target"] != target:
            raise ValueError(f"Arrival mismatch in {run}, goal {epoch}")
        t0 = float(goal["event_t"])
        t1 = float(arrival["event_t"]) if arrival else None
        actual = t1 - t0 if t1 is not None else None
        if actual is not None and actual <= 0:
            raise ValueError(f"Nonpositive elapsed time in {run}, goal {epoch}")

        row = dict(
            run=run, policy=policy, split=split, goal_epoch=epoch,
            start_node=current, target_node=target, start_x_m=points[current][0],
            start_y_m=points[current][1], target_x_m=points[target][0],
            target_y_m=points[target][1], dispatch_sim_s=t0, arrival_sim_s=t1,
            actual_time_s=actual,
        )
        if goal["returning"] or arrival is None:
            row["reason"] = ("final_return" if goal["returning"] else
                             "ended_without_logged_arrival" if epoch in ended else "missing_arrival")
            excluded.append(row)
        else:
            edge = graph[current, target]
            xy = np.asarray(edge["xy"], dtype=float)
            length = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
            old = travel_time_s(length, speed, allowance)
            if length <= 0 or not np.isclose(old, edge["duration_s"], rtol=0, atol=1e-8):
                raise ValueError(f"Invalid path length or baseline mismatch in {run}, goal {epoch}")
            if t0 < times[0] - sample_interval or t1 > times[-1] + sample_interval:
                raise ValueError(f"Trajectory does not cover {run}, goal {epoch}")
            # A goal boundary can precede/follow the trace by at most one odometry sample.
            start_distance, end_distance = np.interp([t0, t1], times, cumulative_distance)
            measured_length = float(end_distance - start_distance)
            if measured_length <= 0:
                raise ValueError(f"No measured displacement for {run}, goal {epoch}")
            legs.append(dict(row, path_length_m=length, actual_distance_m=measured_length, baseline_s=old))
        if arrival is not None:
            current = target
    completed_returns = sum(r["reason"] == "final_return" and r["arrival_sim_s"] is not None for r in excluded)
    if state["segments"] not in (len(legs), len(legs) + completed_returns):
        raise ValueError(f"Completed-leg count mismatch for {run}")
    return legs, excluded


def columns(rows):
    return (np.array([r["path_length_m"] for r in rows]),
            np.array([r["actual_time_s"] for r in rows]))


def travel_time_s(length_m, speed_m_s, delta_s):
    return length_m / speed_m_s + delta_s


def fit(rows):
    """Minimize squared time residuals for L / v + delta, with v > 0 and delta >= 0."""
    length, actual = columns(rows)
    # The substitution beta = 1 / v makes this a linear least-squares problem.
    design = np.column_stack((np.ones(len(length)), length))
    delta_s, beta = np.linalg.lstsq(design, actual, rcond=None)[0]
    if delta_s < 0 or beta < 0:
        boundaries = [
            (0.0, max(0.0, (length @ actual) / (length @ length))),
            (max(0.0, actual.mean()), 0.0),
        ]
        delta_s, beta = min(boundaries, key=lambda pair: np.sum((design @ pair - actual) ** 2))
    if beta <= 0:
        raise ValueError("The training set did not yield a finite positive effective speed")
    return float(1 / beta), float(delta_s)


def fit_fixed_speed(rows):
    """Fix speed from measured motion, then minimize MAE subject to delta >= 0."""
    length, actual = columns(rows)
    distance_m = sum(row["actual_distance_m"] for row in rows)
    time_s = float(actual.sum())
    speed_m_s = distance_m / time_s
    signed_median_s = float(np.median(actual - length / speed_m_s))
    return dict(effective_speed_m_s=speed_m_s, delta_s=max(0.0, signed_median_s),
                unconstrained_delta_s=signed_median_s,
                training_distance_m=distance_m, training_elapsed_s=time_s)


def metrics(rows, column):
    err = np.array([r[column] - r["actual_time_s"] for r in rows])
    under = err < 0
    return dict(n=len(rows), mae_s=float(np.abs(err).mean()),
                rmse_s=float(np.sqrt(np.mean(err**2))), mean_bias_s=float(err.mean()),
                median_bias_s=float(np.median(err)), underestimated_legs=int(under.sum()),
                underestimation_rate=float(under.mean()), worst_underestimate_s=max(0.0, -float(err.min())))


def write_csv(path, rows):
    with path.open("w", encoding="utf8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    output = Path(__file__).resolve().parent
    verification = output.parent.parent / "Verification tests"
    reference = verification / "WIVR_test" / "results" / RUNS["WIVR_test"][-1] / "latest_gp"
    settings = shared_nav_settings(reference)
    rows, excluded, source_runs = [], [], []
    for folder, run_ids in RUNS.items():
        policy = folder.split("_")[0].lower()
        for run_id in run_ids:
            case = verification / folder / "results" / run_id / "latest_gp"
            split = "holdout" if run_id == run_ids[-1] else "train"
            if shared_nav_settings(case) != settings:
                raise ValueError(f"Navigation settings changed for {run_id}")
            legs, other = extract_case(case, policy, split)
            rows.extend(legs)
            excluded.extend(other)
            source_runs.append(dict(
                run=run_id, policy=policy, split=split,
                source=case.relative_to(verification).as_posix(),
                legs=len(legs), excluded_goals=len(other),
                source_sha256={name: hashlib.sha256((case / name).read_bytes()).hexdigest()
                               for name in ("events.jsonl", "graph.json", "candidates.json", "trajectory.npy")},
            ))

    train = [r for r in rows if r["split"] == "train"]
    holdout = [r for r in rows if r["split"] == "holdout"]
    if (len(train), len(holdout)) != (113, 33):
        raise ValueError("Archived data no longer matches the frozen split")
    fixed_speed = fit_fixed_speed(train)
    speed_m_s, delta_s = fit(train)
    held_back_errors = []
    for run_id in sorted({r["run"] for r in train}):
        fold_speed, fold_delta = fit([r for r in train if r["run"] != run_id])
        for r in train:
            if r["run"] == run_id:
                r["train_oof_typical_s"] = travel_time_s(r["path_length_m"], fold_speed, fold_delta)
                held_back_errors.append(r["actual_time_s"] - r["train_oof_typical_s"])
    margin = max(0.0, float(np.quantile(held_back_errors, .90, method="higher")))
    budget_delta_s = delta_s + margin
    median_delta_s = max(0.0, delta_s + float(np.median(held_back_errors)))
    for r in rows:
        r.setdefault("train_oof_typical_s", None)
        r["typical_s"] = travel_time_s(r["path_length_m"], speed_m_s, delta_s)
        r["guarded_s"] = travel_time_s(r["path_length_m"], speed_m_s, budget_delta_s)
        r["median_adjusted_s"] = travel_time_s(r["path_length_m"], speed_m_s, median_delta_s)
        r["fixed_speed_s"] = travel_time_s(r["path_length_m"], fixed_speed["effective_speed_m_s"],
                                            fixed_speed["delta_s"])

    def evaluation(batch):
        return {key: metrics(batch, key) for key in (
            "baseline_s", "typical_s", "guarded_s", "median_adjusted_s", "fixed_speed_s")}

    model = dict(
        domain="Bunker Mini Gazebo simulation with the common Nav2 fixture settings",
        input="Nav2 planned polyline length in metres",
        target="Successful sampling-goal arrival minus dispatch in ROS simulation seconds",
        formula="d_ij = L_ij / v + delta",
        baseline=dict(effective_speed_m_s=0.65, delta_s=8.0),
        fixed_speed_method="Total recorded ground-truth trajectory distance / total dispatch-to-arrival time on training legs",
        fixed_speed_objective="Minimize sum(abs(actual_s - (length_m / v_fixed + delta))) over delta >= 0",
        fixed_speed_estimate=fixed_speed,
        fit_objective="Minimize sum((actual_s - (length_m / v + delta))**2) over v > 0 and delta >= 0",
        fit_method="Constrained least squares using beta = 1 / v",
        typical_estimate=dict(effective_speed_m_s=speed_m_s, delta_s=delta_s),
        median_adjusted_estimate=dict(effective_speed_m_s=speed_m_s, delta_s=median_delta_s),
        budget_estimate=dict(effective_speed_m_s=speed_m_s, delta_s=budget_delta_s),
        budget_margin_s=margin,
        budget_margin_method="Nonnegative empirical P90 of actual minus predicted using leave-one-run-out training predictions",
        train_legs=len(train), holdout_legs=len(holdout), excluded_goals=len(excluded),
        short_leg_threshold_m=SHORT_LEG_M,
        validation=dict(all=evaluation(holdout),
                        short=evaluation([r for r in holdout if r["path_length_m"] <= SHORT_LEG_M]),
                        by_policy={policy: evaluation([r for r in holdout if r["policy"] == policy])
                                   for policy in ("mi", "emse", "wivr")}),
        runs=source_runs,
    )
    write_csv(output / "legs.csv", rows)
    (output / "model.json").write_text(json.dumps(model, indent=2) + "\n", encoding="utf8", newline="\n")
    print(json.dumps({k: model[k] for k in ("fixed_speed_estimate", "typical_estimate", "median_adjusted_estimate")}, indent=2))


if __name__ == "__main__":
    main()

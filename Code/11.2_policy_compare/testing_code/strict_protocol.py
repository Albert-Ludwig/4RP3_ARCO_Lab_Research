"""Timestamp-only splits and policy-weight dispatch for protocol version two."""

from collections import defaultdict
from functools import lru_cache
import json

import bootstrap as paths
from data import load_split
from replay import setup as original_setup, IntelSource
from planner import ObjectivePlanner
from evaluation import distribution_metrics
from experiment import save_csv, save_json
import numpy as np

STRICT_CONFIG = paths.HERE / "strict_config.json"


def settings():
    return json.loads(STRICT_CONFIG.read_text())


def split_records(rows, sensors, start_h, budget_s, bins, reconstruction_min=5):
    """Keep the old per-sensor even/odd split; truncate acquisitions at cutoff."""
    cutoff = start_h + budget_s / 3600
    groups = defaultdict(list)
    for row in rows:
        if row["sensor_id"] in sensors:
            groups[row["sensor_id"]].append(row)
    acquisition, reconstruction = [], []
    future = {f"forecast_{lo}_{hi}m": [] for lo, hi in bins}
    for sensor in sorted(groups):
        ordered = sorted(groups[sensor], key=lambda r: (r["t_h"], r["record_id"]))
        acquisition.extend(r for r in ordered[::2] if start_h <= r["t_h"] <= cutoff)
        heldout = ordered[1::2]
        prior = [r for r in heldout if cutoff - reconstruction_min / 60 <= r["t_h"] <= cutoff]
        if prior:
            reconstruction.append(max(prior, key=lambda r: r["t_h"]))
        for lo, hi in bins:
            future[f"forecast_{lo}_{hi}m"].extend(r for r in heldout
                if cutoff + lo / 60 < r["t_h"] <= cutoff + hi / 60)
    return acquisition, reconstruction, future


@lru_cache(maxsize=1)
def day4():
    return load_split()[1]


def setup_window(start_h):
    config, grid, locations, excluded, points, _, _ = original_setup()
    config["mission"]["start_h"] = start_h
    spec = settings()
    acquired, reference, _ = split_records(day4(), {sensor for sensor, _ in locations}, start_h,
        config["mission"]["budget_s"], spec["forecast_lead_bins_min"], spec["reconstruction_window_min"])
    return config, grid, locations, excluded, points, IntelSource(acquired, start_h), reference


class StrictPlanner(ObjectivePlanner):
    def __init__(self, *args, policy, weights, **kwargs):
        self.public_policy = policy
        mi, emse = settings()["weights_mi_emse"][policy]
        objective = policy if policy in ("mi", "emse") else "fusion"
        super().__init__(*args, policy=objective, weights={"mi": mi, "emse_reduction": emse}, **kwargs)

    def plan(self, *args, **kwargs):
        result = super().plan(*args, **kwargs)
        self.components_log[-1]["policy"] = self.public_policy
        return result


def load_records(path):
    import csv
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key in ("record_id", "sensor_id"):
            row[key] = int(row[key])
        for key in ("x_m", "y_m", "t_h", "temperature_c"):
            row[key] = float(row[key])
    return rows


def evaluate_records(gp, rows, output, task, origin):
    if not rows:
        raise ValueError(f"No measured records for {task}")
    query = np.array([[r["x_m"], r["y_m"], r["t_h"]] for r in rows])
    truth = np.array([r["temperature_c"] for r in rows])
    prediction = gp.predict(query)
    error = prediction["mean"] - truth
    sensors = np.array([r["sensor_id"] for r in rows])
    metrics = distribution_metrics(truth, prediction["mean"], prediction["observed_var"])
    metrics["sensor_balanced_rmse_c"] = float(np.sqrt(np.mean([
        np.mean(error[sensors == sensor] ** 2) for sensor in np.unique(sensors)])))
    metrics.update(task=task, record_count=len(rows), sensor_count=len(np.unique(sensors)),
        forecast_origin_h=origin, min_record_h=float(query[:, 2].min()), max_record_h=float(query[:, 2].max()),
        mean_latent_variance_c2=float(prediction["latent_var"].mean()))
    save_csv(output / f"{task}_predictions.csv", [dict(**r,
        predicted_temperature_c=float(prediction["mean"][i]), latent_variance_c2=float(prediction["latent_var"][i]),
        observation_variance_c2=float(prediction["observed_var"][i]), lead_min=(r["t_h"] - origin) * 60)
        for i, r in enumerate(rows)])
    save_json(output / f"{task}_metrics.json", metrics)
    return metrics

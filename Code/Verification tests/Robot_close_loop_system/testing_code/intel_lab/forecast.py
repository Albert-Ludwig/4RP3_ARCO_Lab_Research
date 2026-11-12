"""Fit the existing RBF GP to 2,000 past records; forecast all valid day-4 rows."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

from data import BASE, ROOT, PROCESSED, load_split, measurements
from experiment import save_csv, save_json
from gp import GPModel
from evaluation import distribution_metrics, calibration_curve

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULT = ROOT / "10.18_result/baseline/intel_lab/forecast"


def main():
    RESULT.mkdir(parents=True, exist_ok=True)
    train, test = load_split()
    settings = json.loads((BASE / "config.json").read_text())["gp"]
    started = time.perf_counter()
    print("Training 2000 observations, 300 steps, unchanged 3D RBF GP", flush=True)
    gp = GPModel.train(measurements(train), 72, settings, 18)
    gp.save(RESULT / "initial_gp.pt")
    save_json(RESULT / "hyperparameters.json", gp.fitted.hyperparameters())
    save_csv(RESULT / "training_history.csv", gp.fitted.history)
    save_json(RESULT / "config.json", dict(gp=settings, seed=18, split=str(PROCESSED),
                                          test_used_for_selection=False))
    print(f"Training completed in {gp.fitted.fit_seconds:.2f} s; predicting {len(test)} test rows", flush=True)
    query = np.array([[r["x_m"], r["y_m"], r["t_h"]] for r in test])
    truth = np.array([r["temperature_c"] for r in test])
    before = time.perf_counter()
    pred = gp.predict(query)
    prediction_s = time.perf_counter() - before
    metrics = distribution_metrics(truth, pred["mean"], pred["observed_var"])
    horizons = query[:, 2] - 72
    hourly = []
    for hour in range(24):
        mask = (horizons >= hour) & (horizons < hour + 1)
        if mask.any():
            hourly.append(dict(horizon_start_h=hour, count=int(mask.sum()),
                               **distribution_metrics(truth[mask], pred["mean"][mask], pred["observed_var"][mask])))
    sensor_metrics = []
    for sensor in sorted({r["sensor_id"] for r in test}):
        mask = np.array([r["sensor_id"] == sensor for r in test])
        sensor_metrics.append(dict(sensor_id=sensor, count=int(mask.sum()),
                                   **distribution_metrics(truth[mask], pred["mean"][mask], pred["observed_var"][mask])))
    save_csv(RESULT / "hourly_metrics.csv", hourly)
    save_csv(RESULT / "sensor_metrics.csv", sensor_metrics)
    save_csv(RESULT / "predictions.csv", [dict(**row, predicted_c=float(pred["mean"][i]),
              latent_var_c2=float(pred["latent_var"][i]), observation_var_c2=float(pred["observed_var"][i]),
              lower95_c=float(pred["mean"][i] - 1.95996398454 * np.sqrt(pred["observed_var"][i])),
              upper95_c=float(pred["mean"][i] + 1.95996398454 * np.sqrt(pred["observed_var"][i])))
             for i, row in enumerate(test)])
    np.savez_compressed(RESULT / "predictions.npz", query=query, truth=truth, **pred)
    summary = dict(train_count=len(train), test_count=len(test), start_h=72,
                   training_s=gp.fitted.fit_seconds, prediction_s=prediction_s,
                   total_s=time.perf_counter() - started, metrics=metrics,
                   calibration_variance="observation variance = latent variance + learned likelihood noise",
                   experiment="Fixed origin 24-hour forecast. No day-4 observation enters the GP.")
    save_json(RESULT / "summary.json", summary)
    figures = RESULT / "figures"
    figures.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].plot([r["horizon_start_h"] + .5 for r in hourly], [r["rmse_c"] for r in hourly], "o-")
    axes[0].set(xlabel="Forecast horizon (h)", ylabel="RMSE (°C)")
    nominal, empirical = calibration_curve(truth, pred["mean"], pred["observed_var"])
    axes[1].plot(nominal, empirical, "o-", label="Observed coverage")
    axes[1].plot([0, 1], [0, 1], "--", color="grey", label="Nominal")
    axes[1].set(xlabel="Nominal coverage", ylabel="Empirical coverage", xlim=(0, 1), ylim=(0, 1))
    axes[1].legend()
    fig.tight_layout(); fig.savefig(figures / "forecast_metrics.png", dpi=160); plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(11, 6), sharex=True)
    for ax, sensor in zip(axes.ravel(), [1, 16, 35, 50]):
        indices = np.array([i for i, r in enumerate(test) if r["sensor_id"] == sensor])[::5]
        x, y, mean = horizons[indices], truth[indices], pred["mean"][indices]
        std = np.sqrt(pred["observed_var"][indices])
        ax.plot(x, y, lw=1, label="Measured")
        ax.plot(x, mean, lw=1, label="GP mean")
        ax.fill_between(x, mean - 1.96*std, mean + 1.96*std, alpha=.2, label="95% observation interval")
        ax.set(title=f"Sensor {sensor}", xlabel="Forecast horizon (h)", ylabel="Temperature (°C)")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(figures / "sensor_forecasts.png", dpi=160); plt.close(fig)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

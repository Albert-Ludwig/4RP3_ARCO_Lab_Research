"""Phase 3: real single-location Dashboard gap reconstruction and forecasting.

Reuses gp_model.fit_gp with one temporal input and the existing RBF/ExactGP
training and posterior machinery. No invented spatial coordinates, synthetic
labels, periodic kernels, GRG or robot trajectories are used.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parent
SYNTHETIC_CODE = ROOT.parent / "3D_time&spatial temp field rebuild & prediction"
if str(SYNTHETIC_CODE) not in sys.path:
    sys.path.insert(0, str(SYNTHETIC_CODE))

import numpy as np

from evaluation import calibration_curve, distribution_metrics
from experiment_paths import DASHBOARD_RESULT
from gp_model import fit_gp
from plots import plt, setup_style, finish

DEFAULT_OUTPUT = DASHBOARD_RESULT


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def save_rows(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_data(path, temperature_column):
    """Strictly audit the supplied export; do not impute or silently drop rows."""
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows, columns = list(reader), reader.fieldnames
    dates, y = [], []
    for row in rows:
        dates.append(datetime.fromisoformat(row["Timestamp"].replace("Z", "+00:00")))
        y.append(float(row[temperature_column]))
    y = np.asarray(y)
    if len(y) < 3 or not np.isfinite(y).all():
        raise ValueError("Export is too short or contains non-finite temperatures")
    elapsed = np.array([(d - dates[0]).total_seconds() / 3600 for d in dates])
    if not (np.diff(elapsed) > 0).all():
        raise ValueError("Require strictly increasing, unique timestamps; inspect export")
    interval = np.diff(elapsed) * 60
    audit = {
        "source_name": Path(path).name, "source_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "columns": columns, "observations": len(y), "temperature_variable": temperature_column,
        "first_timestamp": dates[0].isoformat(), "last_timestamp": dates[-1].isoformat(),
        "span_h": float(elapsed[-1]), "sampling_interval_min": np.unique(np.round(interval, 8)).tolist(),
        "missing_or_invalid_temperature_count": 0, "duplicate_timestamp_count": 0,
        "unique_temperature_values": int(len(np.unique(y))),
        "adjacent_equal_temperature_fraction": float(np.mean(np.diff(y) == 0)),
        "temperature_min_c": float(y.min()), "temperature_max_c": float(y.max()),
        "groups": sorted({r.get("Group", "") for r in rows}),
        "timezone": "unspecified in export" if dates[0].tzinfo is None else str(dates[0].tzinfo),
        "time_representation": "elapsed hours from first exported timestamp; no timezone assigned",
        "spatial_metadata": "no coordinates or explicit station identifier in supplied export",
        "interpretation": "one exported Weather air-temperature series; physical sensor count unverified",
        "limitations": "sensor accuracy, raw-vs-processed status and noise independence are unverified",
    }
    return rows, dates, elapsed[:, None], y, audit


def split_indices(x, config):
    t = x[:, 0]
    gap = config["gap_reconstruction"]
    vg, tg = gap["validation_gap_h"], gap["test_gap_h"]
    val = np.flatnonzero((t >= vg[0]) & (t < vg[1]))
    test = np.flatnonzero((t >= tg[0]) & (t < tg[1]))
    train = np.flatnonzero(~np.isin(np.arange(len(t)), np.r_[val, test]))
    fc = config["forecast"]
    dev = np.flatnonzero(t < fc["development_train_end_h"])
    fv = np.flatnonzero((t >= fc["development_train_end_h"]) & (t < fc["validation_end_h"]))
    ft = np.flatnonzero(t >= fc["test_start_h"])
    refit = np.r_[dev, fv]
    for parts in [(train, val, test), (dev, fv, ft)]:
        combined = np.concatenate(parts)
        if len(combined) != len(t) or len(np.unique(combined)) != len(t):
            raise ValueError("Split configuration must partition the export without overlap")
    if not all(len(a) for a in [train, val, test, dev, fv, ft]):
        raise ValueError("All split parts must be nonempty")
    if x[dev].max() >= x[fv].min() or x[refit].max() >= x[ft].min():
        raise ValueError("Forecast split leaks future observations")
    return {"gap": {"train": train, "validation": val, "test": test},
            "forecast": {"development_train": dev, "validation": fv,
                         "refit_train": refit, "test": ft}}


def save_split(path, indices, rows, x, y, role):
    save_rows(path, [{"source_row": int(i), "source_id": rows[i].get("ID", ""),
                      "timestamp": rows[i]["Timestamp"], "elapsed_h": float(x[i, 0]),
                      "observed_c": float(y[i]), "role": role} for i in indices])



def evaluate(case, fitted, x, y, indices, output):
    test_x, actual = x[indices], y[indices]
    pred = fitted.predict(test_x)
    metrics = [{"case": case, "model": "temporal_rbf_gp", "test_count": len(actual),
                **distribution_metrics(actual, pred["mean"], pred["observed_var"])}]
    save_rows(output / "metrics.csv", metrics)
    save_rows(output / "predictions.csv", [
        {"source_row": int(i), "elapsed_h": float(x[i, 0]), "observed_c": float(y[i]),
         "gp_mean_c": float(pred["mean"][j]), "latent_variance_c2": float(pred["latent_var"][j]),
         "observation_variance_c2": float(pred["observed_var"][j]),
         "lower_95_c": float(pred["mean"][j] - 1.95996398454 * np.sqrt(pred["observed_var"][j])),
         "upper_95_c": float(pred["mean"][j] + 1.95996398454 * np.sqrt(pred["observed_var"][j]))}
        for j, i in enumerate(indices)])
    np.savez_compressed(output / "test_predictions.npz", time_h=test_x[:, 0],
                        observed_c=actual, **pred)
    return metrics, pred


def plot_raw(dates, y, out):
    import matplotlib.dates as mdates
    fig, ax = plt.subplots(figsize=(11, 3.7), layout="constrained")
    ax.plot(dates, y, lw=1.5, marker=".", ms=2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
    ax.set(title="REAL Dashboard export — air temperature", ylabel="Temperature (°C)",
           xlabel="Exported clock timestamps (timezone unspecified)")
    finish(fig, out / "raw_temperature.png")


def plot_case(case, fitted, x, y, parts, config, out):
    pred = fitted.predict(x)
    mean, std = pred["mean"], np.sqrt(pred["observed_var"])
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), layout="constrained")
    test = parts["test"]
    for ax in axes:
        ax.plot(x[:, 0], y, color="0.55", lw=1, label="Actual exported observations")
        ax.plot(x[:, 0], mean, c="tab:blue", lw=1.5, label="GP posterior mean")
        ax.fill_between(x[:, 0], mean - 1.96 * std, mean + 1.96 * std,
                        color="tab:blue", alpha=0.18, label="95% observation predictive interval")
        ax.scatter(x[test, 0], y[test], c="tab:red", s=13, label="Held-out test observations", zorder=4)
        ax.set(ylabel="Air temperature (°C)", xlabel="Hours since first exported timestamp")
    if case == "gap":
        lo, hi = config["gap_reconstruction"]["test_gap_h"]
        axes[0].axvspan(lo, hi, color="tab:green", alpha=0.12, label="Test gap")
        vlo, vhi = config["gap_reconstruction"]["validation_gap_h"]
        axes[0].axvspan(vlo, vhi, color="tab:orange", alpha=0.15, label="Validation gap")
        axes[1].set_xlim(lo - 1, hi + 1)
        zoom = (x[:, 0] >= lo - 1) & (x[:, 0] <= hi + 1)
        axes[1].set_ylim(min(y[zoom].min(), (mean - 1.96 * std)[zoom].min()) - 0.03,
                         max(y[zoom].max(), (mean + 1.96 * std)[zoom].max()) + 0.03)
        fig.suptitle("REAL data: offline gap reconstruction (observations on both sides are available)")
    else:
        boundary = config["forecast"]["test_start_h"]
        for ax in axes:
            ax.axvline(boundary, c="black", ls="--", lw=1, label="Test starts; model frozen")
        axes[1].set_xlim(boundary - 0.5, x[-1, 0] + 0.2)
        fig.suptitle("REAL data: fixed-origin future forecast — fit through Oct 15, test Oct 16")
    for ax in axes:
        ax.legend(fontsize=8, loc="best")
    finish(fig, out / ("gap_reconstruction.png" if case == "gap" else "future_forecast.png"))


def plot_calibration(cases, out):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), layout="constrained")
    for ax, (name, actual, pred) in zip(axes, cases):
        nominal, empirical = calibration_curve(actual, pred["mean"], pred["observed_var"])
        ax.plot(nominal, empirical, "o-", label="Observed coverage")
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Ideal")
        ax.set(title=f"{name}: noisy observations", xlabel="Nominal coverage", ylabel="Empirical coverage",
               xlim=(0, 1), ylim=(0, 1.02))
        ax.legend(fontsize=8)
    fig.suptitle("REAL Dashboard data — prediction interval calibration")
    finish(fig, out / "calibration.png")


def metric_table(metrics):
    lines = ["| Experiment | Predictor | RMSE °C | MAE °C | Cov95 | NLPD | Width95 °C |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for r in metrics:
        extra = (f"{r['coverage_95']:.1%} | {r['nlpd']:.3f} | {r['interval_width_95_c']:.4f}"
                 if "coverage_95" in r else "— | — | —")
        lines.append(f"| {r['case']} | {r['model']} | {r['rmse_c']:.4f} | {r['mae_c']:.4f} | {extra} |")
    return "\n".join(lines)


def run(config_path, output):
    config = json.loads(Path(config_path).read_text())
    source = (ROOT / config["input_csv"]).resolve()
    out = Path(output).resolve()
    if (out / "manifest.json").exists():
        raise FileExistsError(f"Completed results exist at {out}; choose a different --output")
    (out / "figures").mkdir(parents=True, exist_ok=True)
    rows, dates, x, y, audit = load_data(source, config["temperature_column"])
    parts = split_indices(x, config)
    save_json(out / "config.json", config)
    save_json(out / "data_audit.json", audit)
    shutil.copy2(source, out / source.name)
    save_rows(out / "split_manifest.csv", [
        {"experiment": name, "role": role, "count": len(ids),
         "first_timestamp": rows[ids[0]]["Timestamp"], "last_timestamp": rows[ids[-1]]["Timestamp"]}
        for name, roles in parts.items() for role, ids in roles.items()])
    setup_style()
    plot_raw(dates, y, out / "figures")
    metrics, calibration_cases = [], []
    for case, roles in parts.items():
        folder = out / case
        folder.mkdir(parents=True, exist_ok=True)
        for role, ids in roles.items():
            save_split(folder / f"{role}.csv", ids, rows, x, y, role)
        train_key = "train" if case == "gap" else "development_train"
        tr, va = roles[train_key], roles["validation"]
        selected = fit_gp(x[tr], y[tr], x[va], y[va], config["training"], config["seed"])
        selected.save(folder / "selection_model.pt")
        save_json(folder / "selection_hyperparameters.json", selected.hyperparameters())
        save_rows(folder / "selection_history.csv", selected.history)
        if case == "forecast":
            # Validation is now historical data. Freeze the selected iteration
            # budget, refit on the first two days, and never inspect test labels.
            settings = {**config["training"], "iterations": selected.selected_iteration}
            tr = roles["refit_train"]
            fitted = fit_gp(x[tr], y[tr], None, None, settings, config["seed"], select_by_validation=False)
            save_rows(folder / "refit_history.csv", fitted.history)
        else:
            fitted = selected
        fitted.save(folder / "model.pt")
        save_json(folder / "hyperparameters.json", fitted.hyperparameters())
        values, pred = evaluate(case, fitted, x, y, roles["test"], folder)
        metrics.extend(values)
        calibration_cases.append((case, y[roles["test"]], pred))
        plot_case(case, fitted, x, y, roles, config, out / "figures")
        if case == "forecast":
            horizons = x[roles["test"], 0] - x[tr[-1], 0]
            bins = []
            width = config["forecast"]["horizon_bin_h"]
            for lo in np.arange(0, horizons.max(), width):
                mask = (horizons > lo) & (horizons <= lo + width)
                if mask.any():
                    bins.append({"horizon_start_h": float(lo), "horizon_end_h": float(lo + width),
                                 "n": int(mask.sum()), **distribution_metrics(
                                     y[roles["test"]][mask], pred["mean"][mask], pred["observed_var"][mask])})
            save_rows(folder / "forecast_horizons.csv", bins)
        print(f"{case}: {values[0]}", flush=True)
    save_rows(out / "metrics.csv", metrics)
    plot_calibration(calibration_cases, out / "figures")
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "source_sha256": audit["source_sha256"],
                "python": sys.version, "packages": {name: importlib.metadata.version(name) for name in
                                                      ["torch", "gpytorch", "numpy", "scipy", "matplotlib"]},
                "source_code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                       [Path(__file__), SYNTHETIC_CODE / "gp_model.py",
                                        SYNTHETIC_CODE / "evaluation.py"]},
                "phase": 3, "spatial_model": False, "grg_run": False}
    save_json(out / "manifest.json", manifest)
    print(f"Saved Phase 3 results: {out}", flush=True)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "dashboard_config.json")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run(args.config, args.output)

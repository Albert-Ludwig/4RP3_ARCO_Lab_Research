"""Run only Phases 0–2. CLI and Notebook share this experiment implementation."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import sys
import time

import numpy as np

from evaluation import distribution_metrics
from experiment_paths import SYNTHETIC_RESULT, WORKSPACE_ROOT
from gp_model import fit_gp
from synthetic_field import sample_measurements, save_measurements, temperature_true
from plots import (setup_style, plot_ground_truth, plot_reconstruction, plot_sampling,
                   plot_calibration, plot_training, plot_generalization, plot_robustness)

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = SYNTHETIC_RESULT


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def save_rows(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def config_load(path=None):
    return json.loads(Path(path or ROOT / "config.json").read_text())


def repository_provenance():
    repo = WORKSPACE_ROOT / "GP-Planning_reference"
    entries = {}
    for relative in ["notebooks/05_real_field_simulation.ipynb",
                     "notebooks/06_temporal_nonstationary.ipynb",
                     "notebooks/07_continuous_simulation.ipynb", "src/planners.py"]:
        p = repo / relative
        entries[relative] = hashlib.sha256(p.read_bytes()).hexdigest()
    return entries


def independent_splits(config, scenario, seed, noise=None, max_train=None):
    """Explicit disjoint sample draws; spatial and temporal holdouts use blocks."""
    noise = config["noise_std_c"] if noise is None else noise
    bounds = [list(map(list, config["domain"])) for _ in range(3)]
    if scenario == "spatial":
        for b, key in zip(bounds, ["train_x_m", "validation_x_m", "test_x_m"]):
            b[0] = config["spatial_split"][key]
    elif scenario == "temporal":
        for b, key in zip(bounds, ["train_t_h", "validation_t_h", "test_t_h"]):
            b[2] = config["temporal_split"][key]
    elif scenario != "main":
        raise ValueError(scenario)
    counts = [max_train or config["train_count"], config["validation_count"], config["test_count"]]
    parts = [sample_measurements(n, seed * 1000 + 100 + i, b, noise)
             for i, (n, b) in enumerate(zip(counts, bounds))]
    combined = np.vstack([p.x for p in parts])
    if len(np.unique(combined, axis=0)) != len(combined):
        raise AssertionError("Train/validation/test coordinates overlap")
    return tuple(parts)


def evaluate_case(name, model, train, test):
    pred = model.predict(test.x)
    rows = []
    for target, truth, variance in [("latent_truth", test.truth, pred["latent_var"]),
                                     ("noisy_observation", test.observed, pred["observed_var"])]:
        rows.append({"scenario": name, "model": "3d_ard_rbf_gp", "target": target,
                     **distribution_metrics(truth, pred["mean"], variance)})
    return rows, pred


def run_case(config, scenario, out, seed=None):
    seed = config["seed"] + {"main": 0, "spatial": 10, "temporal": 20}[scenario] if seed is None else seed
    case = out / scenario
    case.mkdir(parents=True, exist_ok=True)
    train, validation, test = independent_splits(config, scenario, seed)
    for split, data in zip(["train", "validation", "test"], [train, validation, test]):
        save_measurements(case / f"{split}.csv", data, split)
    model = fit_gp(train.x, train.observed, validation.x, validation.observed,
                   config["training"], seed)
    model.save(case / "model.pt")
    save_json(case / "hyperparameters.json", model.hyperparameters())
    save_rows(case / "training_history.csv", model.history)
    rows, pred = evaluate_case(scenario, model, train, test)
    save_rows(case / "metrics.csv", rows)
    np.savez_compressed(case / "test_predictions.npz", xyt=test.x, truth=test.truth,
                        observed=test.observed, **pred)
    if scenario == "temporal":
        by_horizon = []
        for lo, hi in [(18, 20), (20, 22), (22, 24)]:
            mask = (test.x[:, 2] >= lo) & (test.x[:, 2] < hi)
            by_horizon.append({"test_start_h": lo, "test_end_h": hi,
                               "horizon_from_training_end_h": hi - 16,
                               "n": int(mask.sum()),
                               **distribution_metrics(test.truth[mask], pred["mean"][mask],
                                                      pred["latent_var"][mask])})
        save_rows(case / "forecast_horizons.csv", by_horizon)
    plot_sampling(train, validation, test, scenario, out / "figures" / f"samples_{scenario}.png")
    print(f"{scenario}: latent RMSE={rows[0]['rmse_c']:.4f}°C, "
          f"coverage={rows[0]['coverage_95']:.1%}, fit={model.fit_seconds:.1f}s", flush=True)
    return model, train, validation, test, rows, pred


def run_robustness(config, out):
    robust = config["robustness"]
    rows = []
    for seed in robust["seeds"]:
        # Fixed independent test/validation draws within a seed; nested training
        # coordinates and common standard-normal noise across count/noise settings.
        base = independent_splits(config, "main", seed + 100, noise=0,
                                   max_train=max(robust["train_counts"]))
        for noise in robust["noise_std_c"]:
            for count in robust["train_counts"]:
                train = base[0].prefix(count).with_noise(noise)
                validation, test = (p.with_noise(noise) for p in base[1:])
                case = out / "robustness" / f"seed{seed}_n{count}_noise{noise:g}"
                case.mkdir(parents=True, exist_ok=True)
                for split, data in zip(["train", "validation", "test"], [train, validation, test]):
                    save_measurements(case / f"{split}.csv", data, split)
                model = fit_gp(train.x, train.observed, validation.x, validation.observed,
                               config["training"], seed)
                model.save(case / "model.pt")
                save_json(case / "hyperparameters.json", model.hyperparameters())
                save_rows(case / "training_history.csv", model.history)
                metrics, pred = evaluate_case("robustness", model, train, test)
                for row in metrics:
                    rows.append({"seed": seed, "train_count": count, "noise_std_c": noise, **row})
                np.savez_compressed(case / "test_predictions.npz", xyt=test.x, truth=test.truth,
                                    observed=test.observed, **pred)
                print(f"robustness seed={seed} n={count} noise={noise:g}: "
                      f"RMSE={metrics[0]['rmse_c']:.4f}°C", flush=True)
    save_rows(out / "robustness_runs.csv", rows)
    summary_all = []
    for noise in robust["noise_std_c"]:
        for count in robust["train_counts"]:
            for target in ["latent_truth", "noisy_observation"]:
                group = [r for r in rows if r["noise_std_c"] == noise and r["train_count"] == count
                         and r["target"] == target and r["model"] == "3d_ard_rbf_gp"]
                summary = {"train_count": count, "noise_std_c": noise, "target": target,
                           "runs": len(group)}
                for key in ["rmse_c", "mae_c", "coverage_95", "nlpd", "interval_width_95_c"]:
                    values = [r[key] for r in group]
                    summary[key + "_mean"] = float(np.mean(values))
                    summary[key + "_std"] = float(np.std(values, ddof=1))
                summary_all.append(summary)
    save_rows(out / "robustness_summary.csv", summary_all)
    plot_robustness([r for r in summary_all if r["target"] == "latent_truth"],
                    out / "figures" / "robustness.png")
    return summary_all


def metric_table(rows):
    text = ["| Scenario | Model | Target | RMSE °C | MAE °C | Cov95 | NLPD | Width95 °C |",
            "|---|---|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        extra = (f"{r['coverage_95']:.1%} | {r['nlpd']:.3f} | {r['interval_width_95_c']:.3f}"
                 if "coverage_95" in r else "— | — | —")
        text.append(f"| {r['scenario']} | {r['model']} | {r['target']} | "
                    f"{r['rmse_c']:.4f} | {r['mae_c']:.4f} | {extra} |")
    return "\n".join(text)



def run_all(config=None, output=None):
    config = config or config_load()
    out = Path(output or DEFAULT_OUTPUT).resolve()
    if (out / "manifest.json").exists():
        raise FileExistsError(f"Completed results already exist: {out}. Use a new --output directory.")
    (out / "figures").mkdir(parents=True, exist_ok=True)
    setup_style()
    save_json(out / "config.json", config)
    before = repository_provenance()
    started = time.perf_counter()
    plot_ground_truth(config, out / "figures" / "ground_truth.png")
    main = run_case(config, "main", out)
    model, train, validation, test, rows, pred = main
    plot_reconstruction(config, model, train, out / "figures")
    plot_calibration(test, pred, out / "figures" / "calibration.png")
    plot_training(model.history, model.selected_iteration, out / "figures" / "training.png")
    all_rows, cases = list(rows), []
    for scenario in ["spatial", "temporal"]:
        fit, tr, val, te, metrics, predictions = run_case(config, scenario, out)
        all_rows.extend(metrics)
        cases.append((scenario, te, predictions))
    plot_generalization(cases, out / "figures" / "generalization.png")
    save_rows(out / "metrics.csv", all_rows)
    robustness = run_robustness(config, out)
    after = repository_provenance()
    if before != after:
        raise AssertionError("Reference source changed during experiment")
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(),
                "python": sys.version, "platform": platform.platform(),
                "packages": {p: importlib.metadata.version(p) for p in
                             ["torch", "gpytorch", "numpy", "scipy", "matplotlib", "nbformat", "nbclient"]},
                "reference_sha256_before": before, "reference_sha256_after": after,
                "reference_unchanged": True,
                "experiment_source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                              for p in ROOT.glob("*.py")},
                "elapsed_seconds": time.perf_counter() - started,
                "scope": "Phases 0–2, synthetic data only"}
    save_json(out / "manifest.json", manifest)
    print(f"Completed Phases 0–2. Results: {out}", flush=True)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_all(config_load(args.config), args.output)

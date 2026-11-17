"""Greedy / active GP sampling experiment runner: greedy MI + 2-sensor cold start.

Runs every strategy under matched conditions, fits the same GP, and writes a
timestamped, self-contained results directory:

    results/<timestamp>/
        config.yaml                  resolved config (reproducibility)
        metrics.csv                  per-strategy per-budget scalar metrics
        field_eval.csv               dense true field (x, y, truth)
        final_maps.csv               per-strategy dense mean/var/abs_error at max budget
        selected_points.csv          per-strategy selection-order points at max budget
        greedy_gains.csv             greedy marginal / cumulative MI objective
        random_seeds.csv             raw random Monte-Carlo (per seed, per budget)
        outputscale_comparison.csv   oracle vs pilot output-scale (fixed mode)
        border_fraction.csv          fraction of selected points on the domain edge

The downstream scripts in scripts/ regenerate all figures and tables from these
CSVs WITHOUT importing torch/gpytorch, so results are reproducible from raw data
alone.

Usage:
    python experiments/run_experiment.py [--config PATH] [--full] [--output-root DIR]
"""

import argparse
import math
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.fields import create_grid, get_field
from src.samplers import random_sampler
from src.metrics import path_length
from src.solvers import (
    FixedKernel,
    greedy_information_sampler,
    sequential_active_sampler,
    border_fraction,
)
from src.gp_model import fit_gp_predict


# ----------------------------------------------------------------------------- #
# Setup
# ----------------------------------------------------------------------------- #
def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def lawnmower_with_stride(x_axis, y_axis, n_samples, row_stride=1, column_stride=1):
    x_sub = x_axis[:: max(1, int(column_stride))]
    y_sub = y_axis[:: max(1, int(row_stride))]
    points = []
    for row_idx, y_val in enumerate(y_sub):
        row_x = x_sub if row_idx % 2 == 0 else x_sub[::-1]
        for x_val in row_x:
            points.append((x_val, y_val))
    path = np.array(points, dtype=float)
    if len(path) < n_samples:
        raise ValueError("Not enough lawnmower points for requested n_samples")
    return path[:n_samples]


def _key(p):
    return (round(float(p[0]), 8), round(float(p[1]), 8))


class World:
    """Field, grids, and the shared noisy-measurement lookup."""

    def __init__(self, cfg):
        self.cfg = cfg
        (self.x_eval, self.y_eval, self.x1_eval, self.x2_eval, self.eval_points) = create_grid(
            cfg["evaluation_grid_size"]
        )
        self.eval_field = get_field(cfg["field_name"], self.x1_eval, self.x2_eval)
        self.eval_truth_flat = self.eval_field.ravel()

        (self.x_cand, self.y_cand, self.x1_cand, self.x2_cand, self.candidate_points) = create_grid(
            cfg["candidate_grid_size"]
        )
        self.candidate_field = get_field(cfg["field_name"], self.x1_cand, self.x2_cand)
        clean = self.candidate_field.ravel()

        self.clean_lookup = {_key(p): float(v) for p, v in zip(self.candidate_points, clean)}
        rng_noise = np.random.default_rng(cfg["noise_map_seed"])
        noise = rng_noise.normal(0.0, cfg["noise_std"], size=len(self.candidate_points))
        self.noisy_lookup = {_key(p): float(v) for p, v in zip(self.candidate_points, clean + noise)}

    def measure(self, points):
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        return np.array([self.noisy_lookup[_key(p)] for p in pts], dtype=float)


def make_kernel(cfg, world, pilot_points):
    if cfg["outputscale_source"] == "pilot":
        pilot_meas = world.measure(pilot_points)
        outputscale = max(float(np.var(pilot_meas, ddof=1)), 1e-4)
    else:
        outputscale = float(np.var(world.eval_field))
    return FixedKernel(
        lengthscale=cfg["greedy_lengthscale"],
        outputscale=outputscale,
        noise_variance=cfg["noise_std"] ** 2,
    )


# ----------------------------------------------------------------------------- #
# Strategy execution
# ----------------------------------------------------------------------------- #
def build_pools(cfg, world, kernel, budgets, rng):
    max_budget = max(budgets)
    cand = world.candidate_points

    random_pool = random_sampler(cand, n_samples=max_budget, rng=rng)
    fine_pool = lawnmower_with_stride(
        world.x_cand, world.y_cand, max_budget, cfg["fine_row_stride"], cfg["fine_column_stride"]
    )
    coarse_pool = lawnmower_with_stride(
        world.x_cand, world.y_cand, max_budget, cfg["coarse_row_stride"], cfg["coarse_column_stride"]
    )

    pilot_indices = rng.choice(len(cand), size=cfg["initial_pilot"], replace=False)
    pilot_points = cand[pilot_indices]

    _, greedy_pool, greedy_gain, greedy_obj = greedy_information_sampler(
        cand, max_budget, kernel, initial_indices=pilot_indices
    )
    entropy_pool, _, _ = sequential_active_sampler(
        cand, max_budget, kernel, cfg["r_min"], world.measure,
        acquisition="entropy", initial_points=pilot_points,
    )

    pools = {
        "Random": random_pool,
        "Fine lawnmower": fine_pool,
        "Coarse lawnmower": coarse_pool,
        "Greedy information": greedy_pool,
        "Entropy": entropy_pool,
    }
    greedy_trace = {"gain": greedy_gain, "objective": greedy_obj, "pilot": cfg["initial_pilot"]}
    return pools, greedy_trace


def evaluate_pools(cfg, world, kernel, pools, budgets, greedy_trace, experiment):
    metrics_rows, maps_rows, point_rows = [], [], []
    max_budget = max(budgets)

    for name, pool in pools.items():
        for n in budgets:
            pts = pool[:n]
            noisy = world.measure(pts)
            out = fit_gp_predict(
                pts, noisy, world.eval_points, world.eval_truth_flat, kernel,
                mode=cfg["evaluation_gp_mode"], standardize=cfg["standardize_targets"],
                train_iters=cfg["gp_train_iters"], learning_rate=cfg["gp_learning_rate"],
            )
            ls = out["lengthscale_ard"]
            info_obj = np.nan
            if name == "Greedy information":
                idx = n - greedy_trace["pilot"] - 1
                if 0 <= idx < len(greedy_trace["objective"]):
                    info_obj = float(greedy_trace["objective"][idx])
            metrics_rows.append({
                "Experiment": experiment, "Strategy": name, "Samples": n,
                "PathLength": path_length(pts), "RMSE": out["rmse"],
                "MeanVariance": out["mean_var"], "MeanEntropy": out["mean_entropy"],
                "LengthScale": out["lengthscale"],
                "LengthScale_x": float(ls[0]) if len(ls) > 0 else np.nan,
                "LengthScale_y": float(ls[1]) if len(ls) > 1 else np.nan,
                "OutputScale": out["outputscale"], "NoiseVariance": out["noise_var"],
                "MeanConstant": out["mean_constant"], "ErrUncCorr": out["corr_err_unc"],
                "Coverage95": out["coverage_95"], "MeanZ": out["z_mean"],
                "InfoObjective": info_obj,
            })
            if n == max_budget:
                maps_rows.append(pd.DataFrame({
                    "Experiment": experiment, "Strategy": name,
                    "x": world.eval_points[:, 0], "y": world.eval_points[:, 1],
                    "mean": out["mean"], "var": out["var"], "abs_error": out["abs_error"],
                }))
                for order, p in enumerate(pts):
                    point_rows.append({
                        "Experiment": experiment, "Strategy": name,
                        "Order": order, "x": float(p[0]), "y": float(p[1]),
                    })
    return metrics_rows, maps_rows, point_rows


def run_two_sensor(cfg, world, kernel, budgets, rng):
    """Cold start: a few randomly placed sensors, then the GP samples as normal."""
    cand = world.candidate_points
    k = cfg["two_sensor_pilot"]
    pilot_idx = rng.choice(len(cand), size=k, replace=False)
    pilot_points = cand[pilot_idx]
    max_budget = max(budgets)

    _, greedy_pool, greedy_gain, greedy_obj = greedy_information_sampler(
        cand, max_budget, kernel, initial_indices=list(pilot_idx)
    )
    entropy_pool, _, _ = sequential_active_sampler(
        cand, max_budget, kernel, cfg["r_min"], world.measure,
        acquisition="entropy", initial_points=pilot_points,
    )
    pools = {
        f"Entropy ({k}-sensor)": entropy_pool,
        f"Greedy information ({k}-sensor)": greedy_pool,
    }
    greedy_trace = {"gain": greedy_gain, "objective": greedy_obj, "pilot": k}
    return pools, greedy_trace, pilot_points


def random_monte_carlo(cfg, world, kernel, budgets):
    rows = []
    for s in range(cfg["random_mc_seeds"]):
        rng_i = np.random.default_rng(cfg["random_mc_seed_base"] + s)
        pool_i = random_sampler(world.candidate_points, n_samples=max(budgets), rng=rng_i)
        for n in budgets:
            pts = pool_i[:n]
            out = fit_gp_predict(
                pts, world.measure(pts), world.eval_points, world.eval_truth_flat, kernel,
                mode=cfg["evaluation_gp_mode"], standardize=cfg["standardize_targets"],
                train_iters=cfg["gp_train_iters"], learning_rate=cfg["gp_learning_rate"],
            )
            rows.append({"Seed": s, "Samples": n, "RMSE": out["rmse"],
                         "MeanVariance": out["mean_var"], "PathLength": path_length(pts)})
    return rows


def outputscale_comparison(cfg, world, greedy_pool_pts, pilot_points, budgets):
    rows = []
    oracle = float(np.var(world.eval_field))
    pilot_meas = world.measure(pilot_points)
    pilot = max(float(np.var(pilot_meas, ddof=1)), 1e-4)
    pts = greedy_pool_pts[: max(budgets)]
    noisy = world.measure(pts)
    for label, scale in [("oracle-field-variance", oracle), ("pilot-estimated-variance", pilot)]:
        k = FixedKernel(cfg["greedy_lengthscale"], scale, cfg["noise_std"] ** 2)
        out = fit_gp_predict(
            pts, noisy, world.eval_points, world.eval_truth_flat, k,
            mode="fixed", standardize=cfg["standardize_targets"],
        )
        rows.append({"Mode": label, "OutputScaleUsed": scale, "RMSE": out["rmse"],
                     "MeanVariance": out["mean_var"], "Coverage95": out["coverage_95"],
                     "ErrUncCorr": out["corr_err_unc"]})
    return rows


# ----------------------------------------------------------------------------- #
# Main
# ----------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Run the greedy / active GP sampling experiment.")
    here = os.path.dirname(__file__)
    ap.add_argument("--config", default=os.path.join(here, "configs", "greedy_active_default.yaml"))
    ap.add_argument("--output-root", default=os.path.abspath(os.path.join(here, "..", "results")))
    ap.add_argument("--full", action="store_true", help="Use full_budgets instead of quick_budgets.")
    ap.add_argument("--tag", default="greedy_active", help="Label prefix for the run directory.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    budgets = cfg["full_budgets"] if args.full else cfg["quick_budgets"]
    # Drop budgets exceeding the coarse-lawnmower capacity so it never overflows.
    coarse_cap = (
        math.ceil(cfg["candidate_grid_size"] / cfg["coarse_row_stride"])
        * math.ceil(cfg["candidate_grid_size"] / cfg["coarse_column_stride"])
    )
    budgets = sorted(b for b in budgets if b <= coarse_cap)
    if not budgets:
        raise SystemExit("No budget fits within the coarse-lawnmower capacity.")

    world = World(cfg)
    rng = np.random.default_rng(cfg["seed"])

    # Kernel uses a pilot drawn AFTER the world is built but BEFORE pools, so the
    # output scale (if pilot-sourced) matches the same pilot the strategies use.
    pilot_for_scale = world.candidate_points[
        np.random.default_rng(cfg["seed"]).choice(len(world.candidate_points), size=cfg["initial_pilot"], replace=False)
    ]
    kernel = make_kernel(cfg, world, pilot_for_scale)

    pools, greedy_trace = build_pools(cfg, world, kernel, budgets, rng)
    metrics, maps, points = evaluate_pools(cfg, world, kernel, pools, budgets, greedy_trace, "main")

    ts_pools, ts_trace, ts_pilot = run_two_sensor(cfg, world, kernel, budgets, rng)
    ts_metrics, ts_maps, ts_points = evaluate_pools(cfg, world, kernel, ts_pools, budgets, ts_trace, "two_sensor")

    # Assemble frames
    metrics_df = pd.DataFrame(metrics + ts_metrics)
    maps_df = pd.concat(maps + ts_maps, ignore_index=True)
    points_df = pd.DataFrame(points + ts_points)

    greedy_steps = np.arange(1, len(greedy_trace["gain"]) + 1)
    greedy_df = pd.DataFrame({
        "Experiment": "main", "Step": greedy_steps,
        "MarginalGain": greedy_trace["gain"], "CumulativeObjective": greedy_trace["objective"],
    })
    ts_steps = np.arange(1, len(ts_trace["gain"]) + 1)
    greedy_df = pd.concat([greedy_df, pd.DataFrame({
        "Experiment": "two_sensor", "Step": ts_steps,
        "MarginalGain": ts_trace["gain"], "CumulativeObjective": ts_trace["objective"],
    })], ignore_index=True)

    border_df = pd.DataFrame([
        {"Experiment": "main", "Strategy": n, "BorderFraction": border_fraction(p[: max(budgets)])}
        for n, p in pools.items() if n in ("Random", "Greedy information", "Entropy")
    ] + [
        {"Experiment": "two_sensor", "Strategy": n, "BorderFraction": border_fraction(p[: max(budgets)])}
        for n, p in ts_pools.items()
    ])

    random_df = pd.DataFrame(random_monte_carlo(cfg, world, kernel, budgets))
    outscale_df = pd.DataFrame(
        outputscale_comparison(cfg, world, pools["Greedy information"], pilot_for_scale, budgets)
    )
    field_df = pd.DataFrame({
        "x": world.eval_points[:, 0], "y": world.eval_points[:, 1], "truth": world.eval_truth_flat,
    })

    # Write run directory
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_root, f"{args.tag}_{stamp}")
    os.makedirs(run_dir, exist_ok=True)

    resolved = dict(cfg)
    resolved.update({
        "timestamp": stamp, "budgets_used": budgets, "coarse_capacity": coarse_cap,
        "kernel_lengthscale": kernel.lengthscale, "kernel_outputscale": kernel.outputscale,
        "kernel_noise_variance": kernel.noise_variance,
        "eval_points": int(len(world.eval_points)), "candidate_points": int(len(world.candidate_points)),
        "two_sensor_pilot_points": ts_pilot.tolist(),
    })
    with open(os.path.join(run_dir, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(resolved, f, sort_keys=False)

    metrics_df.to_csv(os.path.join(run_dir, "metrics.csv"), index=False)
    maps_df.to_csv(os.path.join(run_dir, "final_maps.csv"), index=False)
    points_df.to_csv(os.path.join(run_dir, "selected_points.csv"), index=False)
    greedy_df.to_csv(os.path.join(run_dir, "greedy_gains.csv"), index=False)
    random_df.to_csv(os.path.join(run_dir, "random_seeds.csv"), index=False)
    outscale_df.to_csv(os.path.join(run_dir, "outputscale_comparison.csv"), index=False)
    border_df.to_csv(os.path.join(run_dir, "border_fraction.csv"), index=False)
    field_df.to_csv(os.path.join(run_dir, "field_eval.csv"), index=False)

    # Update a stable pointer to the latest run for convenience.
    with open(os.path.join(args.output_root, "LATEST.txt"), "w", encoding="utf-8") as f:
        f.write(os.path.basename(run_dir) + "\n")

    print(f"Wrote results to: {run_dir}")
    print(metrics_df[metrics_df["Samples"] == max(budgets)][["Experiment", "Strategy", "RMSE", "PathLength"]]
          .sort_values(["Experiment", "RMSE"]).to_string(index=False))


if __name__ == "__main__":
    main()

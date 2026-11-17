"""Regenerate all greedy / active sampling paper figures from a results directory.

DEPENDENCY-LIGHT BY DESIGN: imports only numpy / pandas / matplotlib / pyyaml.
It does NOT import torch, gpytorch, or the project's src package, so anyone with
just the CSVs produced by experiments/run_experiment.py can reproduce every figure
without installing the heavy GP stack or re-running the simulation.

Usage:
    python scripts/make_paper_figures.py [--run results/<dir>] [--results-root results]
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MAIN_ORDER = ["Random", "Fine lawnmower", "Coarse lawnmower", "Greedy information", "Entropy"]


# --------------------------------------------------------------------------- #
# Loading helpers
# --------------------------------------------------------------------------- #
def resolve_run(run, results_root):
    if run:
        return run
    latest = os.path.join(results_root, "LATEST.txt")
    if os.path.exists(latest):
        with open(latest, encoding="utf-8") as f:
            return os.path.join(results_root, f.read().strip())
    raise SystemExit("No --run given and results/LATEST.txt not found.")


def load_csv(run_dir, name):
    path = os.path.join(run_dir, name)
    return pd.read_csv(path) if os.path.exists(path) else None


def grid_axes(field_df):
    gx = np.sort(field_df["x"].unique())
    gy = np.sort(field_df["y"].unique())
    X, Y = np.meshgrid(gx, gy)
    return gx, gy, X, Y


def to_grid(values, gx, gy):
    return np.asarray(values).reshape(len(gy), len(gx))


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_comparison_curves(metrics, out):
    d = metrics[metrics["Experiment"] == "main"]
    fig, ax = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    for s in MAIN_ORDER:
        g = d[d["Strategy"] == s].sort_values("Samples")
        if g.empty:
            continue
        ax[0, 0].plot(g["Samples"], g["RMSE"], marker="o", label=s)
        ax[0, 1].plot(g["Samples"], g["MeanVariance"], marker="o", label=s)
        ax[1, 0].plot(g["Samples"], g["PathLength"], marker="o", label=s)
        ax[1, 1].plot(g["PathLength"], g["RMSE"], marker="o", label=s)
    ax[0, 0].set(title="RMSE vs Sample Count", xlabel="Samples", ylabel="RMSE")
    ax[0, 1].set(title="Mean Predictive Variance vs Samples", xlabel="Samples", ylabel="Mean variance")
    ax[1, 0].set(title="Selection-Order Path Length vs Samples", xlabel="Samples", ylabel="Path length")
    ax[1, 1].set(title="RMSE vs Selection-Order Path Length", xlabel="Path length", ylabel="RMSE")
    for a in ax.ravel():
        a.grid(alpha=0.25)
    h, l = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3)
    fig.savefig(out, dpi=130)
    plt.close(fig)


def _map_panels(maps, field_df, strategies, column, cmap, title, out, shared_scale=True):
    gx, gy, X, Y = grid_axes(field_df)
    present = [s for s in strategies if s in maps["Strategy"].unique()]
    if not present:
        return
    vals = maps[maps["Strategy"].isin(present)][column]
    vmin, vmax = (float(vals.min()), float(vals.max())) if shared_scale else (None, None)
    levels = np.linspace(vmin, vmax, 41) if shared_scale else 41
    ncols = 3
    nrows = int(np.ceil(len(present) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.3 * ncols, 4.2 * nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    c = None
    for ax, s in zip(axes, present):
        block = maps[maps["Strategy"] == s]
        Z = to_grid(block[column].values, gx, gy)
        c = ax.contourf(X, Y, Z, levels=levels, cmap=cmap, vmin=vmin, vmax=vmax, extend="both")
        ax.set_title(f"{s} - {title}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    for k in range(len(present), len(axes)):
        axes[k].axis("off")
    if c is not None:
        fig.colorbar(c, ax=axes.tolist(), shrink=0.8, label=title)
    fig.savefig(out, dpi=130)
    plt.close(fig)


def fig_selected_points(points, field_df, strategies, out, title="selected locations"):
    gx, gy, X, Y = grid_axes(field_df)
    Z = to_grid(field_df["truth"].values, gx, gy)
    present = [s for s in strategies if s in points["Strategy"].unique()]
    if not present:
        return
    ncols = min(3, len(present))
    nrows = int(np.ceil(len(present) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.3 * ncols, 4.2 * nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, s in zip(axes, present):
        ax.contourf(X, Y, Z, levels=40, cmap="viridis", alpha=0.85)
        b = points[points["Strategy"] == s].sort_values("Order")
        ax.plot(b["x"], b["y"], "-", lw=0.7, color="white", alpha=0.6)
        ax.scatter(b["x"], b["y"], s=20, c="white", edgecolor="black", linewidth=0.4, zorder=3)
        ax.set_title(f"{s} - {title}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    for k in range(len(present), len(axes)):
        axes[k].axis("off")
    fig.savefig(out, dpi=130)
    plt.close(fig)


def fig_greedy_gain(greedy, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for exp, grp in greedy.groupby("Experiment"):
        grp = grp.sort_values("Step")
        axes[0].plot(grp["Step"], grp["MarginalGain"], marker="o", ms=3, label=exp)
        axes[1].plot(grp["Step"], grp["CumulativeObjective"], marker="o", ms=3, label=exp)
    axes[0].set(title="Greedy Marginal Information Gain", xlabel="Greedy step", ylabel="Marginal gain")
    axes[1].set(title="Cumulative Information Objective F(A_t)", xlabel="Greedy step", ylabel="F(A_t)")
    for a in axes:
        a.grid(alpha=0.25)
        a.legend(fontsize=8)
    fig.savefig(out, dpi=130)
    plt.close(fig)


def fig_calibration(metrics, maps, field_df, out):
    d = metrics[(metrics["Experiment"] == "main") & (metrics["Samples"] == metrics["Samples"].max())]
    d = d.set_index("Strategy").reindex([s for s in MAIN_ORDER if s in d["Strategy"].values or s in d.index]).dropna(how="all")
    truth = field_df["truth"].values
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.6), constrained_layout=True)
    for s in MAIN_ORDER:
        block = maps[(maps["Experiment"] == "main") & (maps["Strategy"] == s)]
        if block.empty:
            continue
        std = np.sqrt(np.maximum(block["var"].values, 1e-12))
        axes[0].scatter(std, block["abs_error"].values, s=5, alpha=0.2, label=s)
    axes[0].set(title="Absolute Error vs Predictive Std (latent)", xlabel="Predictive std", ylabel="Absolute error")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    xpos = np.arange(len(d))
    axes[1].bar(xpos, d["Coverage95"].values)
    axes[1].axhline(0.95, ls="--", color="black", lw=1)
    axes[1].set_xticks(xpos)
    axes[1].set_xticklabels(d.index, rotation=20)
    axes[1].set(title="Empirical 95% Coverage", ylim=(0, 1.05))
    axes[1].grid(axis="y", alpha=0.25)

    axes[2].bar(xpos, d["MeanZ"].values)
    axes[2].axhline(0.0, ls="--", color="black", lw=1)
    axes[2].set_xticks(xpos)
    axes[2].set_xticklabels(d.index, rotation=20)
    axes[2].set(title="Mean Standardized Residual (z)")
    axes[2].grid(axis="y", alpha=0.25)
    fig.savefig(out, dpi=130)
    plt.close(fig)


def fig_random_seeds(random_df, metrics, out):
    summ = random_df.groupby("Samples").agg(
        RMSE_mean=("RMSE", "mean"), RMSE_std=("RMSE", "std")).reset_index()
    greedy = metrics[(metrics["Experiment"] == "main") & (metrics["Strategy"] == "Greedy information")].sort_values("Samples")
    fig, ax = plt.subplots(figsize=(7, 4.6), constrained_layout=True)
    ax.errorbar(summ["Samples"], summ["RMSE_mean"], yerr=summ["RMSE_std"], fmt="-o", label="Random mean +/- std")
    if not greedy.empty:
        ax.plot(greedy["Samples"], greedy["RMSE"], "-s", label="Greedy MI (single run)")
    ax.set(title="Random baseline (20 seeds) vs Greedy MI", xlabel="Samples", ylabel="RMSE")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def fig_outputscale(outscale, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].bar(outscale["Mode"], outscale["RMSE"])
    axes[0].set(title="RMSE by output-scale setting", ylabel="RMSE")
    axes[1].bar(outscale["Mode"], outscale["Coverage95"])
    axes[1].axhline(0.95, color="black", ls="--", lw=1)
    axes[1].set(title="Coverage95 by output-scale setting", ylabel="Coverage95", ylim=(0, 1.05))
    for a in axes:
        a.tick_params(axis="x", rotation=20)
        a.grid(axis="y", alpha=0.25)
    fig.savefig(out, dpi=130)
    plt.close(fig)


def fig_two_sensor_curves(metrics, out):
    ts = metrics[metrics["Experiment"] == "two_sensor"]
    if ts.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), constrained_layout=True)
    # RMSE: 2-sensor cold start vs the 5-pilot main counterparts.
    pairs = [("Entropy", [s for s in ts["Strategy"].unique() if s.startswith("Entropy")]),
             ("Greedy information", [s for s in ts["Strategy"].unique() if s.startswith("Greedy")])]
    for main_name, ts_names in pairs:
        m = metrics[(metrics["Experiment"] == "main") & (metrics["Strategy"] == main_name)].sort_values("Samples")
        if not m.empty:
            axes[0].plot(m["Samples"], m["RMSE"], "--o", label=f"{main_name} (main pilot)")
        for tn in ts_names:
            g = ts[ts["Strategy"] == tn].sort_values("Samples")
            axes[0].plot(g["Samples"], g["RMSE"], "-o", label=tn)
    axes[0].set(title="Cold-start (2 sensors) vs main-pilot RMSE", xlabel="Samples", ylabel="RMSE")
    for tn in ts["Strategy"].unique():
        g = ts[ts["Strategy"] == tn].sort_values("Samples")
        axes[1].plot(g["Samples"], g["MeanVariance"], "-o", label=tn)
    axes[1].set(title="Cold-start mean variance vs samples", xlabel="Samples", ylabel="Mean variance")
    for a in axes:
        a.grid(alpha=0.25)
        a.legend(fontsize=8)
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Regenerate figures from saved CSVs.")
    ap.add_argument("--run", default=None, help="Path to a results/<dir>. Defaults to LATEST.")
    ap.add_argument("--results-root", default="results")
    args = ap.parse_args()

    run_dir = resolve_run(args.run, args.results_root)
    fig_dir = os.path.join(run_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    metrics = load_csv(run_dir, "metrics.csv")
    maps = load_csv(run_dir, "final_maps.csv")
    points = load_csv(run_dir, "selected_points.csv")
    greedy = load_csv(run_dir, "greedy_gains.csv")
    random_df = load_csv(run_dir, "random_seeds.csv")
    outscale = load_csv(run_dir, "outputscale_comparison.csv")
    field_df = load_csv(run_dir, "field_eval.csv")

    def p(name):
        return os.path.join(fig_dir, name)

    if metrics is not None:
        fig_comparison_curves(metrics, p("comparison_curves.png"))
        fig_two_sensor_curves(metrics, p("two_sensor_curves.png"))
    if maps is not None and field_df is not None:
        main_maps = maps[maps["Experiment"] == "main"]
        ts_maps = maps[maps["Experiment"] == "two_sensor"]
        _map_panels(main_maps, field_df, MAIN_ORDER, "mean", "viridis", "GP mean", p("final_maps_mean.png"))
        _map_panels(main_maps, field_df, MAIN_ORDER, "var", "magma", "GP latent variance", p("final_maps_var.png"))
        _map_panels(main_maps, field_df, MAIN_ORDER, "abs_error", "cividis", "|mean - truth|", p("final_maps_error.png"))
        if not ts_maps.empty:
            _map_panels(ts_maps, field_df, sorted(ts_maps["Strategy"].unique()), "mean", "viridis",
                        "GP mean", p("two_sensor_maps.png"))
    if points is not None and field_df is not None:
        main_pts = points[points["Experiment"] == "main"]
        fig_selected_points(main_pts, field_df, MAIN_ORDER, p("selected_points.png"))
    if greedy is not None:
        fig_greedy_gain(greedy, p("greedy_gain.png"))
    if metrics is not None and maps is not None and field_df is not None:
        fig_calibration(metrics, maps, field_df, p("calibration.png"))
    if random_df is not None and metrics is not None:
        fig_random_seeds(random_df, metrics, p("random_seeds.png"))
    if outscale is not None:
        fig_outputscale(outscale, p("outputscale.png"))

    figs = sorted(f for f in os.listdir(fig_dir) if f.endswith(".png"))
    print(f"Wrote {len(figs)} figures to {fig_dir}:")
    for f in figs:
        print("  -", f)


if __name__ == "__main__":
    main()

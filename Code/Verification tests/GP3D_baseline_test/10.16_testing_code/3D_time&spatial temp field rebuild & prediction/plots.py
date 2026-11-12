"""Standalone scientific figures; all panels explicitly identify synthetic data."""

from pathlib import Path
import os

_CACHE_ROOT = Path(__file__).resolve().parent / ".cache"
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from synthetic_field import snapshot_grid, temperature_true
from evaluation import calibration_curve


def setup_style():
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 120})


def finish(fig, path):
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_ground_truth(config, path):
    fig, axes = plt.subplots(1, len(config["snapshot_hours"]), figsize=(12, 3.8),
                             layout="constrained")
    fields = [snapshot_grid(config["grid_size"], t, config["domain"])
              for t in config["snapshot_hours"]]
    truth = [temperature_true(q).reshape(xx.shape) for xx, yy, q in fields]
    vmin, vmax = min(z.min() for z in truth), max(z.max() for z in truth)
    for ax, t, (xx, yy, _), z in zip(axes, config["snapshot_hours"], fields, truth):
        im = ax.pcolormesh(xx, yy, z, cmap="viridis", vmin=vmin, vmax=vmax, shading="auto")
        ax.set(title=f"t = {t:g} h", xlabel="x (m)", ylabel="y (m)", aspect="equal")
    fig.colorbar(im, ax=axes.tolist(), label="True temperature (°C)", shrink=0.85)
    fig.suptitle("SYNTHETIC ground truth — moving local pattern and time variation")
    finish(fig, path)


def plot_reconstruction(config, model, train, out):
    snapshots = []
    for t in config["snapshot_hours"]:
        xx, yy, q = snapshot_grid(config["grid_size"], t, config["domain"])
        pred = model.predict(q)
        truth = temperature_true(q)
        snapshots.append((t, xx, yy, q, truth, pred))
    all_temp = np.concatenate([np.r_[truth, p["mean"]] for _, _, _, _, truth, p in snapshots])
    vmax_std = max(np.sqrt(p["latent_var"]).max() for *_, p in snapshots)
    vmax_err = max(np.abs(p["mean"] - truth).max() for _, _, _, _, truth, p in snapshots)
    fig, axes = plt.subplots(len(snapshots), 4, figsize=(14, 10.5), layout="constrained")
    arrays = {}
    for row, (t, xx, yy, q, truth, p) in enumerate(snapshots):
        std = np.sqrt(p["latent_var"])
        values = [truth, p["mean"], std, np.abs(p["mean"] - truth)]
        for col, (v, title, cmap, lims) in enumerate(zip(values,
                ["Ground truth", "Posterior mean", "Latent posterior std", "Absolute error"],
                ["viridis", "viridis", "magma", "inferno"],
                [(all_temp.min(), all_temp.max())]*2 + [(0, vmax_std), (0, vmax_err)])):
            ax = axes[row, col]
            im = ax.pcolormesh(xx, yy, v.reshape(xx.shape), shading="auto", cmap=cmap,
                               vmin=lims[0], vmax=lims[1])
            # Only nearby-in-time measurements are overlaid; these are not all
            # simultaneous field observations.
            near = np.abs(train.x[:, 2] - t) <= 1.0
            ax.scatter(train.x[near, 0], train.x[near, 1], s=10, c="white",
                       edgecolors="black", linewidths=0.3)
            ax.set(title=f"{title} | t={t:g} h", xlabel="x (m)", ylabel="y (m)", aspect="equal")
            fig.colorbar(im, ax=ax, label="°C", shrink=0.8)
        arrays[f"xyt_{t:g}h"] = q
        arrays[f"truth_{t:g}h"] = truth
        arrays[f"mean_{t:g}h"] = p["mean"]
        arrays[f"latent_var_{t:g}h"] = p["latent_var"]
    fig.suptitle("SYNTHETIC 3D GP reconstruction | dots: training samples within ±1 h\n"
                 "Each column uses a shared colour scale across timestamps")
    finish(fig, out / "reconstruction_maps.png")
    np.savez_compressed(out.parent / "snapshot_predictions.npz", **arrays)


def plot_sampling(train, validation, test, name, path):
    fig = plt.figure(figsize=(9, 5.5), layout="constrained")
    ax = fig.add_subplot(projection="3d")
    for data, label, color in [(train, "Train", "tab:blue"),
                               (validation, "Validation", "tab:orange"),
                               (test.prefix(200), "Test (up to 200 shown)", "tab:green")]:
        ax.scatter(*data.x.T, s=9, alpha=0.55, label=label, color=color)
    ax.set(xlabel="x (m)", ylabel="y (m)", zlabel="t (h)",
           title=f"SYNTHETIC measurement design — {name}")
    ax.legend(fontsize=8)
    finish(fig, path)


def plot_calibration(test, pred, path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), layout="constrained")
    for target, variance, label in [(test.truth, pred["latent_var"], "Noise-free / latent"),
                                    (test.observed, pred["observed_var"], "Noisy / observation")]:
        nominal, empirical = calibration_curve(target, pred["mean"], variance)
        axes[0].plot(nominal, empirical, "o-", label=label)
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set(xlabel="Nominal interval coverage", ylabel="Empirical coverage",
                title="Central interval calibration")
    axes[0].legend(fontsize=8)
    z = (test.observed - pred["mean"]) / np.sqrt(pred["observed_var"])
    axes[1].hist(z, bins=25, density=True, alpha=0.7)
    grid = np.linspace(-4, 4, 200)
    axes[1].plot(grid, norm.pdf(grid), "k--", label="N(0,1)")
    axes[1].set(xlabel="Standardized noisy residual", ylabel="Density", title="Residual calibration")
    axes[1].legend()
    axes[2].scatter(test.truth, pred["mean"], s=8, alpha=0.4)
    lo, hi = min(test.truth.min(), pred["mean"].min()), max(test.truth.max(), pred["mean"].max())
    axes[2].plot([lo, hi], [lo, hi], "k--")
    axes[2].set(xlabel="Ground truth (°C)", ylabel="Posterior mean (°C)", title="Held-out accuracy")
    fig.suptitle("SYNTHETIC main test — latent and observation distributions evaluated separately")
    finish(fig, path)


def plot_training(history, selected_iteration, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), layout="constrained")
    axes[0].plot([r["iteration"] for r in history], [r["negative_mll_scaled"] for r in history])
    validation = [r for r in history if "validation_noisy_nlpd_scaled" in r]
    axes[1].plot([r["iteration"] for r in validation],
                 [r["validation_noisy_nlpd_scaled"] for r in validation], "o-")
    for ax in axes:
        ax.axvline(selected_iteration, c="red", ls="--", label=f"Selected: {selected_iteration}")
        ax.set_xlabel("Iteration")
        ax.legend()
    axes[0].set(title="Training loss", ylabel="Negative MLL (standardized)")
    axes[1].set(title="Validation checkpoint selection", ylabel="Noisy NLPD (standardized)")
    fig.suptitle("SYNTHETIC main fit — test data never used for selection")
    finish(fig, path)


def plot_generalization(cases, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for ax, (name, data, prediction) in zip(axes, cases):
        coord = data.x[:, 0] if name == "spatial" else data.x[:, 2]
        ax.scatter(coord, prediction["mean"] - data.truth, s=8, alpha=0.4)
        std = np.sqrt(prediction["latent_var"])
        ax.scatter(coord, 1.96 * std, s=3, alpha=0.25, c="tab:orange", label="±1.96 latent std")
        ax.scatter(coord, -1.96 * std, s=3, alpha=0.25, c="tab:orange")
        ax.axhline(0, c="black", lw=1)
        ax.set(xlabel="Held-out x (m)" if name == "spatial" else "Unseen t (h)",
               ylabel="Prediction error (°C)", title=f"{name.capitalize()} generalization")
        ax.legend(fontsize=8)
    fig.suptitle("SYNTHETIC blocked holdouts — errors against noise-free ground truth")
    finish(fig, path)


def plot_robustness(summary, path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), layout="constrained")
    noises = sorted({r["noise_std_c"] for r in summary})
    for noise in noises:
        rows = sorted([r for r in summary if r["noise_std_c"] == noise], key=lambda r:r["train_count"])
        x = [r["train_count"] for r in rows]
        for ax, metric, title in zip(axes, ["rmse_c", "coverage_95", "nlpd"],
                                    ["Latent RMSE (°C)", "Latent 95% coverage", "Latent marginal NLPD"]):
            ax.errorbar(x, [r[metric + "_mean"] for r in rows],
                        yerr=[r[metric + "_std"] for r in rows], marker="o", capsize=3,
                        label=f"Noise σ={noise:g}°C")
            ax.set(xlabel="Training measurements", ylabel=title)
    axes[1].axhline(0.95, c="black", ls="--", lw=1)
    axes[0].legend(fontsize=8)
    fig.suptitle("SYNTHETIC robustness — mean ± sample SD across 3 seeds (not confidence intervals)")
    finish(fig, path)

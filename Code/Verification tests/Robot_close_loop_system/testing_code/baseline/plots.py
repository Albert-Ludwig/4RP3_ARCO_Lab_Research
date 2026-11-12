"""Figures for the executed mission and its fixed-time reconstruction."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_figures(folder, grid, points, nav, controller, evaluation, summary):
    folder.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 140})

    def finish(fig, name):
        fig.savefig(folder / name, dpi=160, bbox_inches="tight")
        plt.close(fig)

    def decorate(ax):
        for p in grid.obstacles:
            x, y = p.exterior.xy
            ax.fill(x, y, color="0.25", zorder=5)
        ax.set(xlim=(0, grid.size), ylim=(0, grid.size), xlabel="x (m)", ylabel="y (m)")
        ax.set_aspect("equal")

    fig, ax = plt.subplots(figsize=(8.4, 6))
    polygons = getattr(grid.inflated, "geoms", [grid.inflated])
    for polygon in polygons:
        ax.fill(*polygon.exterior.xy, color="0.8", label=None)
    ax.scatter(*points.T, s=14, color="0.55", label="Candidates")
    trajectory = np.array([[r["x_m"], r["y_m"]] for r in nav.trajectory])
    ax.plot(*trajectory.T, color="#2667a2", lw=1.3, label="Executed path")
    samples = np.array([[r.x_m, r.y_m] for r in controller.online]).reshape(-1, 2)
    if len(samples):
        ax.scatter(*samples.T, s=35, color="#c45b1f", label="Measurements", zorder=6)
        for i, xy in enumerate(samples):
            dx = -16 if 2 < xy[0] < 15 or xy[0] > 27 else 6
            dy = -14 if xy[1] > 27 else 6
            ax.annotate(str(i + 1), xy, xytext=(dx, dy), textcoords="offset points",
                        fontsize=8, zorder=10, bbox=dict(facecolor="white", edgecolor="none", pad=.3))
    ax.scatter(*grid.home, marker="*", s=160, color="#286b39", label="Home", zorder=7)
    decorate(ax)
    ax.set_title("Executed mission with footprint and safety buffers")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    finish(fig, "01_navigation.png")

    truth = evaluation.truth
    initial, final = evaluation.means[0], evaluation.means[-1]
    low, high = min(truth.min(), initial.min(), final.min()), max(truth.max(), initial.max(), final.max())
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    for ax, values, title in zip(axes, [truth, initial, final],
                                 ["Synthetic truth", "History only", "History + robot samples"]):
        im = ax.pcolormesh(evaluation.xx, evaluation.yy, evaluation.field(values),
                           shading="nearest", cmap="coolwarm", vmin=low, vmax=high)
        decorate(ax)
        ax.set_title(title)
    fig.colorbar(im, ax=axes, label="Temperature (°C)", shrink=.8)
    fig.suptitle(f"Temperature at t = {evaluation.query[0, 2]:.2f} h")
    finish(fig, "02_temperature.png")

    variances = [evaluation.variances[0], evaluation.variances[-1]]
    high = np.sqrt(variances[0]).max()
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout="constrained")
    for ax, variance, title in zip(axes, variances, ["Initial latent uncertainty", "Final latent uncertainty"]):
        im = ax.pcolormesh(evaluation.xx, evaluation.yy, evaluation.field(np.sqrt(variance)),
                           shading="nearest", cmap="viridis", vmin=0, vmax=high)
        decorate(ax)
        ax.set_title(title)
    fig.colorbar(im, ax=axes, label="Posterior standard deviation (°C)", shrink=.8)
    finish(fig, "03_uncertainty.png")

    times = [r["elapsed_s"] / 60 for r in evaluation.rows]
    for key, ylabel, name, color in [
        ("rmse_c", "RMSE (°C)", "04_rmse.png", "#2667a2"),
        ("mean_variance_c2", "Mean latent variance (°C²)", "05_variance.png", "#286b39")]:
        fig, ax = plt.subplots(figsize=(7, 3.6), layout="constrained")
        ax.plot(times, [r[key] for r in evaluation.rows], "o-", color=color, ms=4)
        ax.set(xlabel="Elapsed mission time (min)", ylabel=ylabel,
               title=f"Fixed evaluation target: {evaluation.query[0, 2]:.2f} h")
        ax.grid(alpha=.2)
        finish(fig, name)

    rows = [r for r in controller.log if r["gp_updated"]]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), layout="constrained")
    order = np.arange(1, len(rows) + 1)
    axes[0].plot(order, [(summary["budget_s"] - r["after_sample_s"]) / 60 for r in rows], "o-")
    axes[0].set(xlabel="Sampling order", ylabel="Remaining budget (min)")
    axes[1].plot(order, [r["total_gain_nats"] for r in rows], "o-", color="#c45b1f")
    axes[1].set(xlabel="Sampling order", ylabel="Accumulated information (nats)")
    fig.suptitle(f"{summary['measurements']} samples | {summary['travel_distance_m']:.1f} m | "
                 f"{summary['elapsed_s']:.1f} / {summary['budget_s']:.0f} s | "
                 f"Returned home: {summary['returned_home']}")
    finish(fig, "06_progress.png")

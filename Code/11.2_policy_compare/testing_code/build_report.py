"""Build English figures, README, and an executed artifact-verification Notebook."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import bootstrap as paths
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat as nbf
from nbclient import NotebookClient
from jupyter_client import KernelManager
from experiment import save_json

LABELS = {"mi": "GRG + MI", "emse": "GRG + EMSE", "fusion": "GRG + fusion (50/50)"}
COLORS = ["#3977ab", "#df8544", "#41917b"]


def read(path):
    return json.loads(Path(path).read_text())


def table(rows):
    lines = ["| Policy | Samples | Held-out RMSE (C) | MAE (C) | Target mean variance (C^2) | Joint MI (nats) | Planning wall time (s) |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {LABELS[r['policy']]} | {r['measurements']} | {r['final_rmse_c']:.6f} | {r['final_mae_c']:.6f} | {r['mean_target_variance_c2']:.6f} | {r['executed_joint_mi_nats']:.6f} | {r['measured_plan_wall_s']:.3f} |")
    return "\n".join(lines)


def plots(root, rows):
    folder = root / "figures"
    folder.mkdir(exist_ok=True)
    labels = [LABELS[r["policy"]] for r in rows]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    for ax, key, title, unit in zip(axes.ravel(),
            ["final_rmse_c", "mean_target_variance_c2", "executed_joint_mi_nats", "measured_plan_wall_s"],
            ["Held-out real-record reconstruction", "Fixed-deadline target uncertainty", "Executed joint information gain", "Measured planning computation"],
            ["RMSE (C)", "Mean latent variance (C^2)", "MI (nats)", "Wall time (s)"]):
        ax.bar(labels, [r[key] for r in rows], color=COLORS)
        ax.set(title=title, ylabel=unit)
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="y", alpha=.2)
        if key == "final_rmse_c":
            ax.axhline(rows[0]["initial_rmse_c"], color="black", ls="--", label="History-only GP")
            ax.legend(fontsize=8)
    fig.suptitle("One fixed Intel replay; no statistical significance claim")
    fig.savefig(folder / "comparison.png", dpi=150)
    plt.close(fig)
    config = read(root / "protocol.json")["baseline_config"]
    candidates = np.genfromtxt(root / "common/candidates.csv", delimiter=",", names=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    for ax, r, color in zip(axes, rows, COLORS):
        for polygon in config["map"]["obstacles"]:
            xy = np.asarray(polygon)
            ax.fill(xy[:, 0], xy[:, 1], color="#b8bdc2")
        trajectory = np.genfromtxt(root / r["policy"] / "robot_trajectory.csv", delimiter=",", names=True)
        ax.plot(trajectory["x_m"], trajectory["y_m"], color=color, lw=1.5, label="Executed simulated path")
        ax.scatter(candidates["x_m"], candidates["y_m"], s=13, c="#444444", label="Recorded sensor candidates")
        home = config["map"]["home_m"]
        ax.scatter(*home, marker="*", s=100, c="gold", edgecolor="black", zorder=5)
        ax.set(title=f"{LABELS[r['policy']]}\n{r['measurements']} acquired records", xlabel="x (m)", ylabel="y (m)",
               xlim=(0, 30), ylim=(0, 30), aspect="equal")
    axes[0].legend(fontsize=6, loc="upper left")
    fig.savefig(folder / "routes.png", dpi=150)
    plt.close(fig)
    with np.load(root / "common/objective_catalog.npz") as cache:
        xx, yy, mask = cache["xx"], cache["yy"], cache["mask"]
    predictions = []
    for r in rows:
        with np.load(root / r["policy"] / "target_predictions.npz") as p:
            predictions.append((p["final_mean"].copy(), np.sqrt(p["final_variance"])))
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), layout="constrained")
    for ri in range(2):
        low = min(p[ri].min() for p in predictions)
        high = max(p[ri].max() for p in predictions)
        for ci, (r, p) in enumerate(zip(rows, predictions)):
            values = np.full(mask.shape, np.nan)
            values[mask] = p[ri]
            artist = axes[ri, ci].pcolormesh(xx, yy, values, shading="auto", vmin=low, vmax=high,
                                            cmap="coolwarm" if ri == 0 else "viridis")
            axes[ri, ci].set(title=LABELS[r["policy"]] + (" predicted mean" if ri == 0 else " latent std"),
                             xlabel="x (m)", ylabel="y (m)", aspect="equal")
        fig.colorbar(artist, ax=list(axes[ri]), label="C")
    fig.suptitle("GP predictions at 72.25 h; dense measured ground truth is unavailable")
    fig.savefig(folder / "target_predictions.png", dpi=150)
    plt.close(fig)


def build(root):
    assert read(root / "verification.json")["passed"]
    for relative, digest in read(root / "artifacts_sha256.json").items():
        if "figures" not in Path(relative).parts and relative != "report_verification.json":
            assert paths.sha256(root / relative) == digest, relative
    manifest = read(root / "manifest.json")
    for relative, digest in manifest["protected_inputs_sha256"].items():
        assert paths.sha256(paths.WORKSPACE / relative) == digest, relative
    for relative, digest in manifest["source_sha256"].items():
        assert paths.sha256(paths.ROOT / relative) == digest, relative
    rows = read(root / "comparison.json")
    plots(root, rows)
    protocol = read(root / "protocol.json")
    rel = root.relative_to(paths.ROOT).as_posix()
    best = min(rows, key=lambda r: r["final_rmse_c"])
    var_best = min(rows, key=lambda r: r["mean_target_variance_c2"])
    conclusion = (f"On this single recorded replay, {LABELS[best['policy']]} has the lowest held-out RMSE "
                  f"({best['final_rmse_c']:.6f} C), while {LABELS[var_best['policy']]} has the lowest target mean variance "
                  f"({var_best['mean_target_variance_c2']:.6f} C^2). "
                  "These are different quantities. This is one fixed dataset and mission window, not evidence of statistical superiority.")
    improved = [LABELS[r["policy"]] for r in rows if r["final_rmse_c"] < r["initial_rmse_c"]]
    conclusion += " " + ("Lower RMSE than the history-only GP: " + ", ".join(improved) + "." if improved else "No policy lowers RMSE below the history-only GP in this run.")
    text = f"""# 11.2 - GRG Policy Objective Comparison

## Purpose

Compare MI, target-field EMSE reduction, and normalized 50/50 fusion while retaining the existing GRG search, GP checkpoint, Intel data split, navigation, and measurement rules. This is a Python recorded-data replay in the existing Docker container, not a ROS/Nav2 integration experiment.

## Controlled protocol

- Reuse the 10.18 three-day, 2,000-observation GP checkpoint without retraining. Kernel parameters remain fixed; only successful acquisitions update the posterior.
- Same 29 recorded-sensor candidates, 30 x 30 m simulated map, obstacles, Home at sensor 16, 0.35 m/s speed, 5 s dwell, 900 s budget, depth 2, two time splits, and spatial revisits.
- Preserve the existing travel + acquisition waiting + dwell clock. Online computation is charged as zero, as in the old baseline; measured wall time is reported separately. All arms compute both objective components.
- Fixed policy target: {protocol['catalog']['target_count']} reachable points on the original 0.75 m grid at 72.25 h. Target coordinates and timestamps are public; target temperatures are never supplied to the planner.
- Preserve the same 28 held-out real reference records. Their individual timestamps lie in the last five minutes of the mission. Evaluation is retrospective reconstruction, not a causal future forecast or a dense measured field.

## Objective definitions

For a scheduled route S and current observations D:

- **MI:** `0.5 * logdet(I + posterior_cov(S|D) / noise_variance)`.
- **EMSE:** maximize the average target latent-variance reduction, `trace(C_QS (C_SS + noise*I)^(-1) C_SQ) / n_Q`. Target uncertainty after sampling is minimized equivalently. This is a model-based expected-error objective, not access to unknown actual reconstruction error.
- **Fusion:** `0.5 * MI / s_MI + 0.5 * EMSE_reduction / s_EMSE`.

Both normalizers are fixed before any policy executes: the maximum initial single-event MI or EMSE gain over the same public acquisition-event catalog. `s_MI={protocol['catalog']['mi_scale_nats']:.9g}` nats and `s_EMSE={protocol['catalog']['emse_scale_c2']:.9g}` C^2. Route scores may exceed one and are not clipped. Equal weights mean equal relative weights under this declared normalization, not equal physical units or a proven optimal trade-off. No test temperatures or outcome metrics choose the scales or weights.

The subclass overrides only route scoring. All recursive subproblems use that same dispatch. No new GRG approximation guarantee is asserted for EMSE or fusion.

## Results

{table(rows)}

{conclusion}

![Objective comparison]({rel}/figures/comparison.png)
![Executed routes]({rel}/figures/routes.png)
![Predicted target field]({rel}/figures/target_predictions.png)

## Verification and provenance

Seven analytic objective checks, independent full-target GPyTorch/Schur-complement checks before and after numerical-fixture updates, and historical MI-control reproduction must pass. Every arm checks record uniqueness, held-out separation, frozen parameters, valid posterior updates, route timing, collision freedom, return and budget feasibility. Saved predictions independently reproduce reported metrics. Hashes verify protected old inputs and experiment source files.

The MI control reproduces the old recorded route, acquisition records, trajectory and metrics within 1e-7; machine-dependent wall times are excluded. Precomputed conditional covariance is exact for the unchanged fixed-hyperparameter GP. The initial shared catalog preparation time is reported in `protocol.json`; it is outside the original simulated mission clock.

Previous user experiments compared GRG depths and before/after sampling. They did not perform this three-objective comparison. The reference repo's Notebook 06 changed kernel and objective together and did not forward its custom objective to recursive calls; this new experiment does not reuse that implementation.

## Files

- [Executed Notebook](11.2_test_report.ipynb)
- [Comparison table]({rel}/comparison.csv)
- [Fixed protocol]({rel}/protocol.json), [input/source manifest]({rel}/manifest.json)
- [Overall verification]({rel}/verification.json), [MI baseline reproduction]({rel}/mi_baseline_verification.json), [independent numerical verification]({rel}/numerical_verification.json)
- `{rel}/mi`, `/emse`, `/fusion`: individual records, paths, scheduled routes, score components, predictions, final GP checkpoints and execution logs.
- `testing_code/`: scoring, runner, verification, reporting and analytic tests. Original 10.16/10.18/10.27 files are not modified.

## Run in Docker

Use the existing `bunker-jazzy` container with the project mounted at `/workspace` and the isolated GP environment at `/home/ubuntu/.venvs/policy-compare`. It resides on the container Linux filesystem, separate from ROS Python. The exact dependency set is in `testing_code/requirements.txt`; it matches the old numerical package versions. If the container is recreated, recreate this isolated environment from that file.

```powershell
powershell -ExecutionPolicy Bypass -File .\\testing_code\\Start-PolicyCompare.ps1
```

```bash
/home/ubuntu/.venvs/policy-compare/bin/python -B /workspace/Code/11.2_policy_compare/testing_code/run_compare.py
/home/ubuntu/.venvs/policy-compare/bin/python -B /workspace/Code/11.2_policy_compare/testing_code/build_report.py
```

Each run creates a new timestamped directory. Existing runs are never overwritten. Readiness checks and raw artifacts are preserved when an execution fails. This initial 50/50 test does not tune weights, compare kernels, add edge sampling, or change ROS integration.
"""
    (paths.ROOT / "README.md").write_text(text, encoding="utf-8")
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    nb = nbf.v4.new_notebook(cells=[
        md("# 11.2 - GRG Policy Objective Comparison\n\nExecuted analysis of saved experiments. The three full missions were run separately; this Notebook verifies and visualizes their artifacts without rerunning them."),
        code(f"from pathlib import Path\nimport json, hashlib\nimport numpy as np\nfrom IPython.display import display, Image, Markdown\nroot = Path.cwd()\nassert (root / 'testing_code/config.json').is_file()\nrun = root / {rel!r}\nread = lambda p: json.loads(p.read_text())\nprotocol = read(run / 'protocol.json')\nprint('Pinned run:', run.name)\nprint('Weights:', protocol['comparison_config']['fusion_weights'])\nprint('Target points:', protocol['catalog']['target_count'])\nprint('Target time (h):', protocol['catalog']['target_h'])"),
        md("## Hypothesis and fixed protocol\n\nOnly the route-scoring objective changes. GRG depth, timing, history, kernel parameters, navigation and record acquisition stay fixed. EMSE targets mean latent variance over a fixed-deadline reachable grid. The held-out temperature references are separate, nonsimultaneous real observations. Both components are computed in every arm."),
        md("## Normalized 50/50 fusion\n\n`J = 0.5 * MI / s_MI + 0.5 * EMSE_reduction / s_EMSE`. Each scale is a maximum initial single-event gain, frozen before acquisition. No clipping, outcome-based tuning or optimal-weight claim. All recursive scores use the configured objective."),
        code("for name in ['unit_tests.json','numerical_verification.json','mi_baseline_verification.json','verification.json']:\n    result = read(run / name)\n    assert result['passed'], name\n    print(name, 'PASSED')\nfor relative, digest in read(run / 'artifacts_sha256.json').items():\n    assert hashlib.sha256((run / relative).read_bytes()).hexdigest() == digest, relative\nprint('All original run artifact hashes match.')"),
        md("## Measured results\n\nRMSE and MAE use the same 28 held-out measured records; target variance is a model quantity over the fixed grid. These are not interchangeable scores."),
        code(f"display(Markdown({table(rows)!r}))\ndisplay(Image(filename=str(run / 'figures/comparison.png')))"),
        md("## Executed simulated paths\n\nOnly the objective changes the selected destinations; the reused grid navigator executes every leg. Temperature values come from distinct recorded sensor rows at their original times."),
        code("display(Image(filename=str(run / 'figures/routes.png')))"),
        md("## Fixed-deadline GP predictions\n\nThese are predictions and latent standard deviations, not measured dense-field ground truth. Color scales are shared between policy panels."),
        code("display(Image(filename=str(run / 'figures/target_predictions.png')))"),
        md("## Independent replay of saved metrics"),
        code("for policy in protocol['comparison_config']['policies']:\n    summary = read(run / policy / 'summary.json')\n    with np.load(run / policy / 'reference_predictions.npz') as saved:\n        error = saved['final_mean'] - saved['truth']\n        rmse = float(np.sqrt(np.mean(error**2)))\n        mae = float(np.mean(np.abs(error)))\n    assert np.isclose(rmse, summary['final_metrics']['rmse_c'], atol=1e-10)\n    assert np.isclose(mae, summary['final_metrics']['mae_c'], atol=1e-10)\n    with np.load(run / policy / 'target_predictions.npz') as saved:\n        variance = float(saved['final_variance'].mean())\n    assert np.isclose(variance, summary['final_mean_target_variance_c2'], atol=1e-10)\n    print(policy, 'RMSE:', round(rmse, 6), 'MAE:', round(mae, 6), 'target variance:', round(variance, 6))"),
        md("## Interpretation\n\n" + conclusion + "\n\nThe original mission clock excludes computation. Wall-time diagnostics are reported separately, so successful replay is not a real-time deployment claim. This comparison does not establish future-forecast accuracy, optimal fusion weights, or a new GRG approximation guarantee. Kernel changes, continuous acquisition and ROS/Nav2 integration are outside this experiment.")])
    nb.metadata.kernelspec = dict(name="python3", display_name="Python (policy comparison Docker)", language="python")
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    NotebookClient(nb, km=manager, timeout=180, resources={"metadata": {"path": str(paths.ROOT)}}).execute()
    nbf.validate(nb)
    notebook = paths.ROOT / "11.2_test_report.ipynb"
    nbf.write(nb, notebook)
    save_json(root / "report_verification.json", dict(passed=True,
        executed_code_cells=sum(c.cell_type == "code" for c in nb.cells),
        error_outputs=sum(o.output_type == "error" for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", [])),
        readme_sha256=paths.sha256(paths.ROOT / "README.md"), notebook_sha256=paths.sha256(notebook),
        report_builder_sha256=paths.sha256(Path(__file__)),
        figure_sha256={p.name: paths.sha256(p) for p in (root / "figures").glob("*.png")}))
    print("Wrote README and executed Notebook:", notebook, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or paths.ROOT / "11.2_result" / read(paths.ROOT / "11.2_result/latest.json")["run"]
    build(output)

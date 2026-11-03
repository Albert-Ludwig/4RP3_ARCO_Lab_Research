"""English report from verified five-policy reconstruction/forecast artifacts."""

import argparse
import csv
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

LABELS = {"history_only": "History only", "mi": "MI", "emse": "EMSE",
          "fusion": "MI:EMSE 50:50", "fusion_40_60": "MI:EMSE 40:60", "fusion_60_40": "MI:EMSE 60:40"}
TASKS = ["reconstruction", "forecast_0_15m", "forecast_15_30m", "forecast_30_60m"]
TASK_LABELS = ["Retrospective reconstruction", "Forecast (0, 15] min", "Forecast (15, 30] min", "Forecast (30, 60] min"]
COLORS = ["#555555", "#377eb8", "#e58b35", "#41917b", "#a05ab5", "#b74e62"]


def read(path):
    return json.loads(Path(path).read_text())


def aggregate_table(rows, policies):
    lines = ["| Policy (MI:EMSE) | Reconstruction | Forecast (0,15] min | Forecast (15,30] min | Forecast (30,60] min |",
             "|---|---:|---:|---:|---:|"]
    values = {(r["policy"], r["task"]): r for r in rows}
    for policy in policies:
        lines.append("| " + LABELS[policy] + " | " + " | ".join(f"{values[policy, task]['mean_window_rmse_c']:.6f}" for task in TASKS) + " |")
    return "\n".join(lines)


def window_table(rows, policies):
    values = {(r["window"], r["policy"], r["task"]): r for r in rows}
    lines = ["| Window | Policy | Reconstruction RMSE | Forecast (0,15] | Forecast (15,30] | Forecast (30,60] |",
             "|---|---|---:|---:|---:|---:|"]
    for window in sorted({r["window"] for r in rows}):
        for policy in policies:
            lines.append("| " + window + " | " + LABELS[policy] + " | " + " | ".join(f"{values[window, policy, task]['rmse_c']:.6f}" for task in TASKS) + " |")
    return "\n".join(lines)


def figures(root, aggregate, rows, policies):
    folder = root / "figures"
    folder.mkdir(exist_ok=True)
    values = {(r["policy"], r["task"]): r for r in aggregate}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for ax, task, title in zip(axes.ravel(), TASKS, TASK_LABELS):
        ax.bar(np.arange(len(policies)), [values[p, task]["mean_window_rmse_c"] for p in policies], color=COLORS)
        ax.set_xticks(np.arange(len(policies)), [LABELS[p].replace("MI:EMSE ", "") for p in policies], rotation=15, fontsize=8)
        ax.set(title=title, ylabel="Mean of three window RMSEs (C)")
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("Three preset windows of one dataset; descriptive comparison, no significance claim")
    fig.savefig(folder / "strict_comparison.png", dpi=150)
    plt.close(fig)
    values = {(r["window"], r["policy"], r["task"]): r for r in rows}
    windows = sorted({r["window"] for r in rows})
    fig, axes = plt.subplots(1, len(windows), figsize=(13, 4), layout="constrained")
    for ax, window in zip(axes, windows):
        for policy, color in zip(policies, COLORS):
            ax.plot(range(3), [values[window, policy, task]["rmse_c"] for task in TASKS[1:]],
                    marker="o", color=color, ls="--" if policy == "history_only" else "-", label=LABELS[policy])
        ax.set_xticks(range(3), ["(0,15]", "(15,30]", "(30,60]"])
        ax.set(title=window + " - frozen final GP", xlabel="Lead-time bin (min)", ylabel="RMSE (C)")
        ax.grid(alpha=.2)
    axes[-1].legend(fontsize=7)
    fig.suptitle("Forecasts use only observations available by each common cutoff")
    fig.savefig(folder / "forecast_by_window.png", dpi=150)
    plt.close(fig)
    active = policies[1:]
    fig, axes = plt.subplots(1, len(active), figsize=(16, 4.2), layout="constrained")
    window = root / "windows/w00"
    config = read(window / "protocol.json")["baseline_config"]
    candidates = np.genfromtxt(window / "common/candidates.csv", delimiter=",", names=True)
    for ax, policy, color in zip(axes, active, COLORS[1:]):
        for polygon in config["map"]["obstacles"]:
            xy = np.asarray(polygon)
            ax.fill(xy[:, 0], xy[:, 1], color="#b8bdc2")
        trajectory = np.genfromtxt(window / policy / "robot_trajectory.csv", delimiter=",", names=True)
        summary = read(window / policy / "summary.json")
        ax.plot(trajectory["x_m"], trajectory["y_m"], color=color, lw=1.3)
        ax.scatter(candidates["x_m"], candidates["y_m"], s=9, c="#333333")
        ax.scatter(*config["map"]["home_m"], marker="*", s=90, c="gold", edgecolor="black", zorder=5)
        ax.set(title=LABELS[policy] + f"\n{summary['measurements']} acquired records", xlabel="x (m)", ylabel="y (m)",
               xlim=(0, 30), ylim=(0, 30), aspect="equal")
    fig.suptitle("Window w00: executed simulated routes; other windows have separate saved trajectories")
    fig.savefig(folder / "five_routes_w00.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def build(root):
    assert read(root / "verification.json")["passed"]
    for relative, digest in read(root / "artifacts_sha256.json").items():
        assert paths.sha256(root / relative) == digest, relative
    manifest = read(root / "manifest.json")
    for relative, digest in manifest["protected_inputs_sha256"].items():
        assert paths.sha256(paths.WORKSPACE / relative) == digest, relative
    for relative, digest in manifest["source_sha256"].items():
        assert paths.sha256(paths.ROOT / relative) == digest, relative
    config, aggregate, rows = read(root / "config_frozen.json"), read(root / "aggregate.json"), read(root / "evaluation.json")
    policies = ["history_only"] + config["policies"]
    figures(root, aggregate, rows, policies)
    rel = root.relative_to(paths.ROOT).as_posix()
    outcomes = []
    for task, title in zip(TASKS, TASK_LABELS):
        group = [r for r in aggregate if r["task"] == task]
        best = min((r for r in group if r["policy"] != "history_only"), key=lambda r: r["mean_window_rmse_c"])
        history = next(r for r in group if r["policy"] == "history_only")
        outcomes.append(f"- {title}: lowest mean window RMSE among sampling arms is **{LABELS[best['policy']]}**, "
            f"{best['mean_window_rmse_c']:.6f} C; history only is {history['mean_window_rmse_c']:.6f} C.")
    conclusion = "\n".join(outcomes)
    counts = []
    missions = []
    equivalent_sequences = []
    for window in sorted((root / "windows").iterdir()):
        sequences = {}
        for metric in read(window / "history_only/task_metrics.json"):
            counts.append(f"| {window.name} | {metric['task']} | {metric['record_count']} | {metric['sensor_count']} | {metric['min_record_h']:.8f} - {metric['max_record_h']:.8f} |")
        for policy in config["policies"]:
            s = read(window / policy / "summary.json")
            missions.append(f"| {window.name} | {LABELS[policy]} | {s['measurements']} | {s['elapsed_s']:.3f} | {s['measured_plan_wall_s']:.3f} |")
            with (window / policy / "online_measurements.csv").open() as stream:
                sequence = tuple(int(r["record_id"]) for r in csv.DictReader(stream))
            sequences.setdefault(sequence, []).append(LABELS[policy])
        for group in sequences.values():
            if len(group) > 1:
                equivalent_sequences.append(f"- {window.name}: " + ", ".join(group) + " acquired the identical ordered record sequence.")
    counts_table = "| Window | Task | Held-out records | Sensors | Actual timestamps (h) |\n|---|---|---:|---:|---|\n" + "\n".join(counts)
    mission_table = "| Window | Policy | Acquisitions | Simulated mission (s) | Planning wall time (s) |\n|---|---|---:|---:|---:|\n" + "\n".join(missions)
    protocol_text = """Five preset arms: MI, EMSE, MI:EMSE 50:50, 40:60, and 60:40. The order is always MI then EMSE. Fusion uses `J = w_MI * MI/s_MI + w_EMSE * EMSE_reduction/s_EMSE`; raw MI and EMSE have different units and are not directly added. Normalizers are the unchanged maximum initial single-event gains, shared by all arms within each window and frozen before sampling. No test temperature selects weights or normalizers.

All arms reuse the same three-day/2,000-observation GP checkpoint, fixed kernel parameters, 29 candidate sensors, synthetic map and obstacles, Home, navigation, 0.35 m/s speed, 5 s dwell, 900 s budget, GRG depth 2 and two time splits. The only policy change is route scoring. Successful recorded acquisitions condition the posterior; there is no kernel refit. Every arm computes both score components.

Three preset mission starts are 72, 80 and 88 hours from the dataset origin (day-four 00:00, 08:00 and 16:00 in the original timestamp convention; timezone unspecified). Each is an independent replay reset to the same checkpoint, not a multi-window online trajectory. The last two therefore start with older information. The three windows are correlated parts of one dataset, not independent repeated trials. No confidence interval or statistical-superiority claim is justified by these three windows alone.

The original clock counts travel, acquisition waiting and dwell but charges online computation as zero. Planning wall time is saved separately. This is Python recorded-data replay, not a real-time or ROS/Nav2 experiment. Both the planning/execution acquisition schedule and return-to-Home budget are checked.
"""
    evaluation_text = """**Reconstruction:** after the 15-minute mission, evaluate the latest odd-index held-out observation per reachable sensor in the last five minutes before the common deadline. Predictions use each record's actual timestamp. This is retrospective window reconstruction; it is neither a simultaneous dense measured field nor a causal forecast for those pre-cutoff records.

**Forecasting:** the common forecast origin is mission start plus 15 minutes. Reload each saved final GP and freeze it. Evaluate only odd-index held-out records in `(0,15]`, `(15,30]` and `(30,60]` minutes after that origin. No future observations are appended and no parameters change. The history-only GP uses the same origins and test records. Acquisition pools are truncated at the common cutoff, and acquisition/test record IDs are disjoint. Forecast truths are used solely for evaluation.

**Objective versus evaluation:** EMSE still targets the same reachable grid at the mission deadline. The future forecast scores are additional outcomes; this run does not change EMSE into a future-horizon objective. Model-based expected squared error / latent variance is distinct from empirical error on held-out temperatures. Lower target variance does not guarantee lower measured RMSE.

**Metrics:** primary empirical RMSE and MAE use identical records for all policies within each task/window. NLPD and 95% predictive coverage use observation variance, including learned noise. Sensor-balanced RMSE gives sensors equal weight despite unequal logging rates. The summary below is the arithmetic mean of three window RMSEs, not pooled-record RMSE. Per-window results and record counts are retained to expose variation.

**Spatial scope:** the preserved split alternates records within each sensor; sensors themselves are not held out from acquisition. Nearby records are correlated, and dense measured ground truth is unavailable. Reconstruction therefore tests the recorded reachable locations, not spatial extrapolation to entirely unseen sensors or true error over the entire map. The future bins provide the separate temporal holdout. A later spatial-generalization claim would require a separate held-out-location protocol.
"""
    checks_text = """The five timestamp/split/weight tests and seven analytic objective tests pass. Each window also passes independent GPyTorch-versus-cached covariance/Schur-complement checks. Every mission checks frozen hyperparameters, unique and chronological acquisitions, valid posterior updates, held-out separation, feasible collision-free routes, matching schedules and budgeted return. Forecast verification checks that the final training timestamps are no later than the common origin and that evaluating future records leaves GP training data and parameters unchanged.

Saved prediction CSVs independently reproduce every task metric. The first window's MI, EMSE and 50:50 arms reproduce the earlier three-policy records, routes, trajectories and metrics within 1e-7; wall time is excluded. Input/source SHA-256 checks preserve the earlier successful run, original README/Notebook/latest pointer, and reused older experiment files.
"""
    text = f"""# 11.2 - Strict Reconstruction and Forecast Comparison

## Fixed protocol

{protocol_text}
## Evaluation and leakage boundaries

{evaluation_text}
## Results

Mean of three window RMSEs (C), lower is better:

{aggregate_table(aggregate, policies)}

{conclusion}

These describe this fixed dataset and these preset windows. They do not establish a universally best ratio or validate weight tuning on the same outcomes. A later confirmatory comparison should freeze the selected setting and use separate days or field data. Comparisons of this frozen-kernel model do not identify an optimal kernel.

Identical acquisitions can occur because the route choices are discrete; changing weights need not change the selected route. With the same fixed GP and identical acquisitions, predictions are identical:

{chr(10).join(equivalent_sequences) or 'No two sampling arms acquired identical sequences in these windows.'}

![Aggregate reconstruction and forecasts]({rel}/figures/strict_comparison.png)
![Forecast by window]({rel}/figures/forecast_by_window.png)

## Per-window results

All entries are empirical RMSE in C. Windows w00, w08 and w16 start at 72, 80 and 88 hours.

{window_table(rows, policies)}

## Actual held-out coverage

{counts_table}

## Mission accounting

{mission_table}

![Five routes at w00]({rel}/figures/five_routes_w00.png)

## Verification

{checks_text}
## Artifacts and reproduction

- [Executed Notebook](11.2_strict_test_report.ipynb)
- [Preset configuration](testing_code/strict_config.json)
- [Per-window/task metrics]({rel}/evaluation.csv), [aggregate metrics]({rel}/aggregate.csv)
- [Verification]({rel}/verification.json), [provenance and environment]({rel}/manifest.json)
- `{rel}/windows/w00`, `/w08`, `/w16`: common held-out record lists, catalogs and travel matrices; each policy contains acquisition records, trajectories, planned routes, score components, final GP, predictions, metrics and logs. `history_only` is the no-new-measurement control.
- The original three-policy report and run remain intact. New runs use `latest_strict.json`; `latest.json` still identifies the earlier run.

From this experiment folder, with Docker Desktop running and the existing `bunker-jazzy` container and `/home/ubuntu/.venvs/policy-compare` environment available:

```powershell
powershell -ExecutionPolicy Bypass -File .\\testing_code\\Start-StrictCompare.ps1
```

This creates a new timestamped run, verifies it, and builds this report and executed Notebook. The Notebook analyzes saved artifacts without rerunning missions. Dependencies are the pinned existing `requirements.txt` / `requirements-lock.txt`. No ROS launch, new kernel, map change, continuous along-path sampling or Nav2 integration is introduced.
"""
    readme = paths.ROOT / "README_strict.md"
    readme.write_text(text, encoding="utf-8")
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    nb = nbf.v4.new_notebook(cells=[
        md("# Strict GRG Policy Comparison: Reconstruction and Forecasting\n\nExecuted analysis of 15 saved missions and three shared history-only controls. The original three-policy experiment is preserved."),
        code(f"from pathlib import Path\nimport csv, hashlib, json\nimport numpy as np\nfrom IPython.display import Image, Markdown, display\nroot = Path.cwd()\nrun = root / {rel!r}\nread = lambda p: json.loads(p.read_text())\nconfig = read(run / 'config_frozen.json')\nprint('Pinned run:', run.name)\nprint('Preset MI:EMSE weights:', config['weights_mi_emse'])\nprint('Mission starts (h):', config['window_start_hours'])"),
        md("## Controlled protocol\n\n" + protocol_text),
        md("## Reconstruction versus causal future forecasting\n\n" + evaluation_text),
        code("assert read(run / 'verification.json')['passed']\nfor relative, digest in read(run / 'artifacts_sha256.json').items():\n    assert hashlib.sha256((run / relative).read_bytes()).hexdigest() == digest, relative\nprint('All original strict-run artifact hashes and verification checks pass.')"),
        md("## Mean of three window RMSEs\n\nEach window contributes equal weight to this descriptive mean; the windows are not independent trials."),
        code(f"display(Markdown({aggregate_table(aggregate, policies)!r}))\ndisplay(Image(filename=str(run / 'figures/strict_comparison.png')))"),
        md("## Per-window forecasting"),
        code(f"display(Markdown({window_table(rows, policies)!r}))\ndisplay(Image(filename=str(run / 'figures/forecast_by_window.png')))"),
        md("## Actual measured records and acquisition accounting\n\n" + counts_table + "\n\n" + mission_table),
        code("display(Image(filename=str(run / 'figures/five_routes_w00.png')))"),
        md("## Independently recompute RMSE and verify future boundaries"),
        code("verified = 0\nfor metric in read(run / 'evaluation.json'):\n    folder = run / 'windows' / metric['window'] / metric['policy']\n    with (folder / (metric['task'] + '_predictions.csv')).open() as stream:\n        predictions = list(csv.DictReader(stream))\n    error = np.array([float(r['predicted_temperature_c']) - float(r['temperature_c']) for r in predictions])\n    rmse = float(np.sqrt(np.mean(error**2)))\n    assert np.isclose(rmse, metric['rmse_c'], atol=1e-10)\n    if metric['task'].startswith('forecast'):\n        lo, hi = [int(v.rstrip('m')) for v in metric['task'].split('_')[1:]]\n        origin = metric['forecast_origin_h']\n        assert all(origin + lo/60 < float(r['t_h']) <= origin + hi/60 for r in predictions)\n    verified += 1\nprint('Verified', verified, 'task/model/window results from saved real-record predictions.')"),
        md("## Interpretation\n\n" + conclusion + "\n\nNo general winner is established. Freeze the chosen protocol before evaluating separate days or field data. The deadline EMSE objective and future forecasting remain distinct.\n\n" + checks_text)])
    nb.metadata.kernelspec = dict(name="python3", display_name="Python (policy comparison Docker)", language="python")
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    try:
        NotebookClient(nb, km=manager, timeout=240, resources={"metadata": {"path": str(paths.ROOT)}}).execute()
    finally:
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
    nbf.validate(nb)
    notebook = paths.ROOT / "11.2_strict_test_report.ipynb"
    nbf.write(nb, notebook)
    for relative, digest in manifest["protected_inputs_sha256"].items():
        assert paths.sha256(paths.WORKSPACE / relative) == digest, relative
    errors = sum(o.output_type == "error" for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", []))
    assert errors == 0
    save_json(root / "report_verification.json", dict(passed=True,
        executed_code_cells=sum(c.cell_type == "code" for c in nb.cells), error_outputs=errors,
        readme_sha256=paths.sha256(readme), notebook_sha256=paths.sha256(notebook),
        report_builder_sha256=paths.sha256(Path(__file__)),
        figure_sha256={p.name: paths.sha256(p) for p in (root / "figures").glob("*.png")}))
    print("Wrote strict README and executed Notebook:", notebook, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or paths.ROOT / "11.2_result" / read(paths.ROOT / "11.2_result/latest_strict.json")["run"]
    build(output)

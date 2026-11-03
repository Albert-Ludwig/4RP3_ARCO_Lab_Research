"""Build one concise README and executed Notebook from immutable saved runs."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import bootstrap as paths

LABELS = {"history_only": "History only", "mi": "MI", "emse": "EMSE", "fusion": "MI:EMSE 5:5",
          "fusion_40_60": "MI:EMSE 4:6", "fusion_60_40": "MI:EMSE 6:4", "fusion_70_30": "MI:EMSE 7:3"}
TASKS = ["reconstruction", "forecast_0_15m", "forecast_15_30m", "forecast_30_60m"]
TITLES = ["Reconstruction", "Forecast (0,15] min", "Forecast (15,30] min", "Forecast (30,60] min"]
COLORS = ["#555555", "#377eb8", "#e58b35", "#41917b", "#a05ab5", "#b74e62", "#82713b"]


def read(path):
    return json.loads(Path(path).read_text())


def historical_digest(path, expected, prefer_git=False):
    """Verify current bytes or the original Git blob, never substitute a digest."""
    path = Path(path)
    if not prefer_git and path.is_file() and paths.sha256(path) == expected:
        return "working_tree"
    relative = path.relative_to(paths.WORKSPACE).as_posix()
    git = ["git", "-c", f"safe.directory={paths.WORKSPACE}", "-C", str(paths.WORKSPACE)]
    revisions = subprocess.check_output(git + ["log", "--format=%H", "--", relative], text=True).splitlines()
    for revision in revisions:
        result = subprocess.run(git + ["show", f"{revision}:{relative}"], capture_output=True)
        if result.returncode == 0 and hashlib.sha256(result.stdout).hexdigest() == expected:
            return revision
    return None


def latest_run():
    for name in ["latest_70_30.json", "latest_strict.json"]:
        pointer = paths.ROOT / "11.2_result" / name
        if pointer.exists():
            return paths.ROOT / "11.2_result" / read(pointer)["run"]
    raise FileNotFoundError("Run Start-StrictCompare.ps1 before building this report.")


def verify_saved_results(root):
    """Recompute empirical errors and time boundaries without fitting or missions."""
    import numpy as np
    checks = read(root / "verification.json")
    assert checks["passed"]
    extension = root / "extension_verification.json"
    if extension.exists():
        assert read(extension)["passed"]
    for relative, expected in read(root / "artifacts_sha256.json").items():
        assert paths.sha256(root / relative) == expected, relative
    rows = read(root / "evaluation.json")
    for metric in rows:
        folder = root / "windows" / metric["window"] / metric["policy"]
        with (folder / f"{metric['task']}_predictions.csv").open() as stream:
            prediction = list(csv.DictReader(stream))
        error = np.array([float(r["predicted_temperature_c"]) - float(r["temperature_c"]) for r in prediction])
        assert np.isclose(np.sqrt(np.mean(error**2)), metric["rmse_c"], atol=1e-10)
        if metric["task"].startswith("forecast"):
            lo, hi = [int(v.rstrip("m")) for v in metric["task"].split("_")[1:]]
            origin = metric["forecast_origin_h"]
            assert all(origin + lo/60 < float(r["t_h"]) <= origin + hi/60 for r in prediction)
    print(f"Verified {len(rows)} metric cases; {len(checks['checks'])} saved execution checks passed.")
    return len(rows)


def summary_table(root):
    policies = ["history_only"] + read(root / "config_frozen.json")["policies"]
    values = {(r["policy"], r["task"]): r for r in read(root / "aggregate.json")}
    lines = ["| Policy | Reconstruction | Forecast (0,15] min | Forecast (15,30] min | Forecast (30,60] min |",
             "|---|---:|---:|---:|---:|"]
    for policy in policies:
        lines.append("| " + LABELS[policy] + " | " + " | ".join(
            f"{values[policy, task]['mean_window_rmse_c']:.3f}" for task in TASKS) + " |")
    return "\n".join(lines)


def make_figures(root):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    destination = paths.ROOT / "11.2_result/report_figures"
    destination.mkdir(exist_ok=True)
    policies = ["history_only"] + read(root / "config_frozen.json")["policies"]
    values = {(r["policy"], r["task"]): r for r in read(root / "aggregate.json")}
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    for ax, task, title in zip(axes.ravel(), TASKS, TITLES):
        ax.bar(range(len(policies)), [values[p, task]["mean_window_rmse_c"] for p in policies], color=COLORS)
        ax.set_xticks(range(len(policies)), [LABELS[p].replace("MI:EMSE ", "") for p in policies], rotation=15, fontsize=8)
        ax.set(title=title, ylabel="Mean window RMSE (C)")
        ax.grid(axis="y", alpha=.2)
    fig.savefig(destination / "comparison.png", dpi=140)
    plt.close(fig)
    values = {(r["window"], r["policy"], r["task"]): r for r in read(root / "evaluation.json")}
    windows = sorted({w for w, _, _ in values})
    fig, axes = plt.subplots(1, len(windows), figsize=(12, 3.8), layout="constrained")
    for ax, window in zip(np.atleast_1d(axes), windows):
        for policy, color in zip(policies, COLORS):
            ax.plot(range(3), [values[window, policy, task]["rmse_c"] for task in TASKS[1:]],
                color=color, marker="o", ls="--" if policy == "history_only" else "-", label=LABELS[policy])
        ax.set_xticks(range(3), ["(0,15]", "(15,30]", "(30,60]"])
        ax.set(title=window, xlabel="Forecast lead-time bin (min)", ylabel="RMSE (C)")
        ax.grid(alpha=.2)
    np.atleast_1d(axes)[-1].legend(fontsize=7)
    fig.savefig(destination / "forecast_windows.png", dpi=140)
    plt.close(fig)
    selected = [p for p in ["mi", "emse", "fusion_60_40", "fusion_70_30"] if p in policies]
    window = root / "windows/w16"
    config = read(window / "protocol.json")["baseline_config"]
    candidates = np.genfromtxt(window / "common/candidates.csv", delimiter=",", names=True)
    fig, axes = plt.subplots(1, len(selected), figsize=(12, 3.6), layout="constrained")
    for ax, policy in zip(np.atleast_1d(axes), selected):
        for polygon in config["map"]["obstacles"]:
            xy = np.asarray(polygon)
            ax.fill(xy[:, 0], xy[:, 1], color="#b8bdc2")
        trajectory = np.genfromtxt(window / policy / "robot_trajectory.csv", delimiter=",", names=True)
        summary = read(window / policy / "summary.json")
        ax.plot(trajectory["x_m"], trajectory["y_m"], color=COLORS[policies.index(policy)], lw=1.4)
        ax.scatter(candidates["x_m"], candidates["y_m"], s=9, c="#333333")
        ax.scatter(*config["map"]["home_m"], marker="*", s=85, c="gold", edgecolor="black", zorder=5)
        ax.set(title=f"{LABELS[policy]}\n{summary['measurements']} measurements", xlabel="x (m)", ylabel="y (m)",
            xlim=(0, 30), ylim=(0, 30), aspect="equal")
    fig.savefig(destination / "routes_w16.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    return destination


def build(root=None):
    import nbformat as nbf
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    root = Path(root or latest_run()).resolve()
    count = verify_saved_results(root)
    mutable = {paths.ROOT / name for name in ["README.md", "README_strict.md", "README_70_30.md",
        "11.2_test_report.ipynb", "11.2_strict_test_report.ipynb", "11.2_70_30_test_report.ipynb",
        "testing_code/run_strict_compare.py", "testing_code/run_70_30_compare.py"]}
    archived = {}
    manifest = read(root / "manifest.json")
    entries = [(paths.WORKSPACE / p, h) for p, h in manifest["protected_inputs_sha256"].items()]
    entries += [(paths.ROOT / p, h) for p, h in manifest["source_sha256"].items()]
    for path, expected in entries:
        if path in mutable:
            origin = historical_digest(path, expected, prefer_git=path.suffix in {".md", ".ipynb"})
            assert origin, str(path)
            if origin != "working_tree":
                archived[str(path.relative_to(paths.WORKSPACE))] = dict(sha256=expected, git_commit=origin)
        else:
            assert paths.sha256(path) == expected, str(path)
    figures = make_figures(root)
    relative = root.relative_to(paths.ROOT).as_posix()
    config = read(root / "config_frozen.json")
    policies = config["policies"]
    aggregate = read(root / "aggregate.json")
    conclusions = []
    for task, title in zip(TASKS, TITLES):
        best = min((r for r in aggregate if r["task"] == task and r["policy"] != "history_only"),
                   key=lambda r: r["mean_window_rmse_c"])
        conclusions.append(f"- **{title}:** {LABELS[best['policy']]} has the lowest mean RMSE among sampling arms ({best['mean_window_rmse_c']:.3f} C).")
    if "fusion_70_30" in policies:
        conclusions.append("- **7:3 versus 6:4:** the first two windows acquire identical records; w16 differs and 7:3 is worse on all four aggregate RMSEs.")
    conclusions += ["- **4:6 and 5:5:** identical acquisitions and results in all three windows.",
        "- At w00, history only beats all sampling arms; later windows benefit from new measurements. Improvement is not uniform across time.",
        "- Keep 6:4 as the current prediction-oriented candidate. Separate-day/field tests are needed before claiming general superiority. EMSE still targets the deadline field, not a future forecast horizon."]
    findings = "\n".join(conclusions)
    policy_text = ", ".join(LABELS[p] for p in policies)
    readme = f"""# 11.2 - GP/GRG Policy Comparison

## Purpose and comparison

Compare how the GRG route-scoring objective affects temperature-field reconstruction and future prediction while holding the GP, navigation, acquisition rules and budget fixed. This is a Python replay of Intel Lab measurements, separate from ROS 2/Nav2 simulation.

Sampling policies: **{policy_text}**. A history-only GP is the no-new-measurement control. MI maximizes conditional information gain; EMSE maximizes average latent-variance reduction over a fixed target grid. Fusion uses `w_MI * MI/s_MI + w_EMSE * EMSE_reduction/s_EMSE`, with shared initial single-event normalizers fixed before sampling.

| Item | Setting |
|---|---|
| Data/model | Intel Lab; 2,000 observations from days 1-3; saved GP with fixed kernel/noise parameters |
| Missions | Day-four starts at 72, 80 and 88 h; same initial checkpoint per policy/window |
| Navigation | 29 sensor candidates; synthetic 30 x 30 m map; grid/Dijkstra navigation; GRG depth 2 |
| Budget | 900 s; 0.35 m/s; 5 s dwell; travel and acquisition waiting included; computation charged zero, wall time saved separately |
| Reconstruction | Latest held-out record per reachable sensor in the final five minutes; retrospective, at actual record timestamps |
| Forecast | Freeze final GP at the common deadline; test strictly later records in (0,15], (15,30], (30,60] min; no future updates |
| Metrics | Mean of three window RMSEs; MAE, sensor-balanced RMSE, coverage and per-window NLPD also saved |

## Dependencies

Paths below are relative to the repository root. Keep these sibling experiments and data files in place.

| File/directory | Required role |
|---|---|
| `Code/11.2_policy_compare/testing_code/` | Runners, objective scoring, split rules, verification and the single report builder |
| `Code/10.18_Robot_close_loop_system/testing_code/baseline/` | GP wrapper, GRG recursion, mission controller, grid navigation and output helpers |
| `Code/10.18_Robot_close_loop_system/testing_code/intel_lab/` | Intel loading, candidate construction and recorded measurement acquisition |
| `Code/10.16_GP3D_baseline_test/10.16_testing_code/3D_time&spatial temp field rebuild & prediction/` | Shared GP/scaling/checkpoint loader and evaluation metrics |
| `Data/Intel_lab_data/processed/three_day_2000/` | `train_2000.csv`, `test_day4.csv`, `split.json` |
| `Code/10.18_Robot_close_loop_system/10.18_result/baseline/intel_lab/forecast/` | `initial_gp.pt`, `hyperparameters.json` |
| `Code/10.18_Robot_close_loop_system/10.18_result/baseline/intel_lab/replay/` | Historical MI control for regression verification |
| [requirements.txt](testing_code/requirements.txt), [requirements-lock.txt](testing_code/requirements-lock.txt) | Pinned numerical/reporting packages; saved runs use Python 3.12 and CPU PyTorch |

Use the existing Docker container `bunker-jazzy`, project mount `/workspace`, and Python `/home/ubuntu/.venvs/policy-compare/bin/python`. ROS Python is separate. Git history retains the superseded reports and verifies their original hashes. The local `GP-Planning` reference clone is not a runtime dependency.

## Run

From this folder, with Docker Desktop running, reproduce the five-policy comparison and then the 7:3 extension:

```powershell
powershell -ExecutionPolicy Bypass -File .\\testing_code\\Start-StrictCompare.ps1
powershell -ExecutionPolicy Bypass -File .\\testing_code\\Start-70-30Compare.ps1
```

Rebuild only the consolidated report from saved results:

```powershell
docker --context desktop-linux exec -u ubuntu bunker-jazzy /home/ubuntu/.venvs/policy-compare/bin/python -B /workspace/Code/11.2_policy_compare/testing_code/build_report.py
```

All report commands write only `README.md` and `11.2_test_report.ipynb`. The Notebook analyzes saved artifacts without running missions. The original three-policy runner remains available for regression reproduction.

## Results and records

- [Executed Notebook: results, figures and conclusions](11.2_test_report.ipynb)
- [Aggregate metrics]({relative}/aggregate.csv), [per-window metrics]({relative}/evaluation.csv)
- [Frozen settings]({relative}/config_frozen.json), [checks]({relative}/verification.json), [provenance]({relative}/manifest.json)
- `11.2_result/latest.json`, `latest_strict.json`, `latest_70_30.json` identify the initial, five-policy and six-policy records. This report uses `{root.name}`.
- Raw runs retain measurements, paths, score components, GP checkpoints, predictions, logs and hashes. A failed initial setup attempt is retained for diagnosis. Current figures and report verification are in `11.2_result/report_figures/` and `11.2_result/report_verification.json`.

**Current finding:** EMSE has the lowest mean reconstruction RMSE; 6:4 has the lowest mean RMSE in all three forecast bins. These are exploratory results from three correlated windows of one dataset; 7:3 was added after inspecting earlier results. Separate-day/field validation is needed. Reconstruction covers recorded sensor locations, not dense measured map truth. Historical manifests remain unchanged; superseded report/source hashes are verified against Git history.
"""
    (paths.ROOT / "README.md").write_text(readme, encoding="utf-8")
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    nb = nbf.v4.new_notebook(cells=[
        md(f"# 11.2 - GP/GRG Policy Comparison\n\n{len(policies)} sampling objectives, three fixed windows and a shared history-only control. The complete six-policy study includes 18 sampled missions; the 7:3 extension reuses the earlier 15. Same frozen GP and 15-minute budget. This Notebook reads saved results without rerunning missions. See README.md for dependencies and commands."),
        code(f"from pathlib import Path\nimport sys\nfrom IPython.display import Image, Markdown, display\nroot = Path.cwd()\nif not (root / 'testing_code').is_dir():\n    root = root / 'Code/11.2_policy_compare'\nsys.path.insert(0, str(root / 'testing_code'))\nfrom build_report import verify_saved_results, summary_table\nrun = root / {relative!r}\nprint('Saved run:', run.name)\nverify_saved_results(run)"),
        md("## Mean window RMSE (C)\n\nLower is better. Each window has equal weight; this is not pooled-record RMSE. Reconstruction uses actual held-out timestamps in the final five minutes. Forecast bins follow the common deadline with the final GP frozen. Other metrics and full per-window results remain in the raw records."),
        code("display(Markdown(summary_table(run)))\ndisplay(Image(filename=str(root / '11.2_result/report_figures/comparison.png')))"),
        md("## Forecast variation across time\n\nWindows w00/w08/w16 start at 72/80/88 h. Each resets to the same three-day checkpoint; later windows start with older information. Curves overlap when policies acquire identical records."),
        code("display(Image(filename=str(root / '11.2_result/report_figures/forecast_windows.png')))"),
        md("## Representative routes: w16\n\nThis is where 6:4 and 7:3 diverge. Paths use the Python grid navigator. Dots are candidate sensors; the star is Home. Measurement counts, complete trajectories, waiting time and budgeted return are saved per policy."),
        code("display(Image(filename=str(root / '11.2_result/report_figures/routes_w16.png')))"),
        md("## Conclusions\n\n" + findings + f"\n\n[Aggregate CSV]({relative}/aggregate.csv) | [Per-window CSV]({relative}/evaluation.csv) | [Checks]({relative}/verification.json)\n\nThese correlated, previously inspected Intel windows support recorded-location reconstruction and future observation prediction; they do not establish dense spatial truth, general optimality or hardware performance.")])
    nb.metadata.kernelspec = dict(name="python3", display_name="Python (policy comparison Docker)", language="python")
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    try:
        NotebookClient(nb, km=manager, timeout=240, resources={"metadata": {"path": str(paths.ROOT)}}).execute()
    finally:
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
    nbf.validate(nb)
    notebook = paths.ROOT / "11.2_test_report.ipynb"
    nbf.write(nb, notebook)
    errors = sum(o.output_type == "error" for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", []))
    assert errors == 0
    report = dict(passed=True, source_run=root.name, checked_results=count, executed_code_cells=4,
        error_outputs=errors, historical_files_verified_in_git=archived,
        readme_sha256=paths.sha256(paths.ROOT / "README.md"), notebook_sha256=paths.sha256(notebook),
        builder_sha256=paths.sha256(Path(__file__)), figure_sha256={p.name: paths.sha256(p) for p in figures.glob("*.png")})
    (paths.ROOT / "11.2_result/report_verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Updated the single README and executed Notebook:", notebook, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    build(parser.parse_args().output)

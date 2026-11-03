"""Report the incremental 70:30 experiment without replacing previous reports."""

import json
from pathlib import Path
import sys

import bootstrap as paths
import build_strict_report as base
import nbformat as nbf
from nbclient import NotebookClient
from jupyter_client import KernelManager
from experiment import save_json

base.LABELS["fusion_70_30"] = "MI:EMSE 70:30"
base.COLORS.append("#82713b")


def build():
    root = paths.ROOT / "11.2_result" / base.read(paths.ROOT / "11.2_result/latest_70_30.json")["run"]
    assert base.read(root / "verification.json")["passed"]
    assert base.read(root / "extension_verification.json")["passed"]
    for relative, digest in base.read(root / "artifacts_sha256.json").items():
        assert paths.sha256(root / relative) == digest, relative
    manifest = base.read(root / "manifest.json")
    for relative, digest in manifest["protected_inputs_sha256"].items():
        assert paths.sha256(paths.WORKSPACE / relative) == digest, relative
    for relative, digest in manifest["source_sha256"].items():
        assert paths.sha256(paths.ROOT / relative) == digest, relative
    config, rows, aggregate = (base.read(root / name) for name in ("config_frozen.json", "evaluation.json", "aggregate.json"))
    policies = ["history_only"] + config["policies"]
    base.figures(root, aggregate, rows, policies)
    summary = base.aggregate_table(aggregate, policies)
    per_window = base.window_table(rows, policies)
    lookup = {(r["policy"], r["task"]): r for r in aggregate}
    comparison = []
    for task, title in zip(base.TASKS, base.TASK_LABELS):
        previous = lookup["fusion_60_40", task]["mean_window_rmse_c"]
        current = lookup["fusion_70_30", task]["mean_window_rmse_c"]
        comparison.append(f"- {title}: 70:30 = {current:.6f} C, 60:40 = {previous:.6f} C; difference = {current - previous:+.6f} C (negative favors 70:30).")
    conclusion = "\n".join(comparison)
    equivalence = []
    import csv
    for window in sorted((root / "windows").iterdir()):
        sequences = {}
        for policy in config["policies"]:
            with (window / policy / "online_measurements.csv").open() as stream:
                sequences[policy] = tuple(int(r["record_id"]) for r in csv.DictReader(stream))
        same = [base.LABELS[p] for p, ids in sequences.items() if p != "fusion_70_30" and ids == sequences["fusion_70_30"]]
        equivalence.append(f"- {window.name}: 70:30 " + ("acquired the identical ordered record sequence as " + ", ".join(same) + "." if same else "acquired a different ordered record sequence from all earlier arms."))
    equivalence_text = "\n".join(equivalence)
    rel = root.relative_to(paths.ROOT).as_posix()
    protocol = """This is an exploratory follow-up requested after viewing the earlier five-policy comparison. Only three new 70:30 missions were executed, at the unchanged mission starts 72, 80 and 88 hours. The 15 earlier missions, three history-only controls, public acquisition catalogs, travel matrices, reference records and numerical checks are reused byte-for-byte and their hashes are verified. Existing run directories, reports and result pointers remain intact.

The new objective is `0.7 * MI / s_MI + 0.3 * EMSE_reduction / s_EMSE`. The unchanged normalizers were fixed from initial single-event gains within each window. Saved scores independently verify that every selected route uses MI weight 0.7 and EMSE weight 0.3. There is no new kernel training or change to the GP checkpoint, GRG depth 2, candidate nodes, synthetic map, navigation, 0.35 m/s speed, 5 s dwell or 900 s budget. Computation remains outside the original simulated mission clock; measured wall times are recorded separately.

Reconstruction uses the same latest held-out actual record per reachable sensor in the last five minutes before the mission deadline. This is retrospective reconstruction at recorded locations, not dense measured map truth or a held-out-sensor test. Forecasts freeze the final GP at the same common deadline and evaluate strictly later held-out records in (0,15], (15,30] and (30,60] minutes, with no future conditioning. EMSE still targets the deadline field; future forecasting is an additional evaluation, not a changed planning objective.

Primary summary values are arithmetic means of the three window RMSEs, in C; lower is better. MAE, sensor-balanced RMSE and predictive coverage are also saved in the aggregate CSV; per-window metric JSONs additionally include NLPD. The three windows are correlated portions of one dataset. Adding 70:30 after inspecting the earlier results is exploratory tuning on the same evaluation windows, not independent validation of a best ratio. Any selected ratio still needs a separate-day or separate-field test.
"""
    text = f"""# 11.2 - MI:EMSE 70:30 Follow-up

## Protocol and provenance

{protocol}
## Mean window RMSE comparison

{summary}

## 70:30 versus 60:40

{conclusion}

{equivalence_text}

![Seven models including the history-only control]({rel}/figures/strict_comparison.png)
![Forecast results by window]({rel}/figures/forecast_by_window.png)

## Per-window RMSE

{per_window}

## Executed routes

The plot shows all six sampling policies at w00. Every window has its own saved trajectory, acquired records, score components, final GP and predictions.

![Six sampling routes]({rel}/figures/five_routes_w00.png)

## Verification and files

- [Executed Notebook](11.2_70_30_test_report.ipynb)
- [Full metric table]({rel}/aggregate.csv), [per-window results]({rel}/evaluation.csv)
- [Frozen six-policy configuration]({rel}/config_frozen.json), [provenance manifest]({rel}/manifest.json)
- [Causal/metric/input verification]({rel}/verification.json), [copy and 70:30 score verification]({rel}/extension_verification.json)
- `{rel}/windows/w00/fusion_70_30`, `/w08/fusion_70_30`, `/w16/fusion_70_30`: the three newly executed missions. All other arm folders contain verified copies of the earlier results, including their original timing logs.
- Earlier reports are preserved: [five-policy report](README_strict.md), [original three-policy report](README.md).

The unchanged 12 analytic and timestamp tests were rerun successfully. Each new mission passes the original budget, return, trajectory, acquisition, fixed-parameter and posterior checks. Forecast records are common across policies, disjoint from acquired record IDs and strictly beyond the cutoff; saved predictions reproduce the metrics. The copied earlier inputs and results are hash-identical, and all older protected artifacts are unchanged.

From this experiment directory, in the existing Docker environment:

```powershell
powershell -ExecutionPolicy Bypass -File .\\testing_code\\Start-70-30Compare.ps1
```

The command reruns only the 70:30 arm for the same three windows and creates a new timestamped extension. It does not rerun ROS/Nav2 or modify the earlier five-policy configuration. `latest_70_30.json` identifies this extension; older pointers are preserved.
"""
    readme = paths.ROOT / "README_70_30.md"
    readme.write_text(text, encoding="utf-8")
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    nb = nbf.v4.new_notebook(cells=[
        md("# MI:EMSE 70:30 Follow-up\n\nThree new missions and 15 verified earlier missions.\n\n" + protocol),
        code(f"from pathlib import Path\nimport csv, hashlib, json\nimport numpy as np\nfrom IPython.display import display, Image, Markdown\nroot = Path.cwd()\nrun = root / {rel!r}\nread = lambda p: json.loads(p.read_text())\nassert read(run/'verification.json')['passed']\nassert read(run/'extension_verification.json')['passed']\nfor relative,digest in read(run/'artifacts_sha256.json').items():\n    assert hashlib.sha256((run/relative).read_bytes()).hexdigest()==digest,relative\nprint('Pinned extension:',run.name)\nprint('Newly executed missions:',read(run/'extension_verification.json')['newly_executed_missions'])\nprint('Requested weights:',read(run/'config_frozen.json')['weights_mi_emse']['fusion_70_30'])\nprint('Artifact and causal checks passed.')"),
        md("## Mean of the three window RMSEs"),
        code(f"display(Markdown({summary!r}))\ndisplay(Image(filename=str(run/'figures/strict_comparison.png')))"),
        md("## 70:30 versus 60:40\n\n" + conclusion + "\n\n" + equivalence_text),
        code(f"display(Markdown({per_window!r}))\ndisplay(Image(filename=str(run/'figures/forecast_by_window.png')))"),
        md("## All six sampling routes at w00"),
        code("display(Image(filename=str(run/'figures/five_routes_w00.png')))"),
        md("## Independently recompute all saved empirical RMSEs"),
        code("verified=0\nfor metric in read(run/'evaluation.json'):\n    folder=run/'windows'/metric['window']/metric['policy']\n    with (folder/(metric['task']+'_predictions.csv')).open() as stream:\n        rows=list(csv.DictReader(stream))\n    error=np.array([float(r['predicted_temperature_c'])-float(r['temperature_c']) for r in rows])\n    assert np.isclose(np.sqrt(np.mean(error**2)),metric['rmse_c'],atol=1e-10)\n    if metric['task'].startswith('forecast'):\n        lo,hi=[int(v.rstrip('m')) for v in metric['task'].split('_')[1:]]\n        origin=metric['forecast_origin_h']\n        assert all(origin+lo/60<float(r['t_h'])<=origin+hi/60 for r in rows)\n    verified+=1\nprint('Verified task/model/window results:',verified)"),
        md("## Interpretation limits\n\nThese are descriptive comparisons on the same three previously inspected windows. Do not present the added ratio as independently validated or universally best. The stored data support recorded-location reconstruction and strictly future forecasting, not dense spatial ground truth or real-robot execution.")])
    nb.metadata.kernelspec = dict(name="python3", display_name="Python (policy comparison Docker)", language="python")
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    try:
        NotebookClient(nb, km=manager, timeout=240, resources={"metadata": {"path": str(paths.ROOT)}}).execute()
    finally:
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
    nbf.validate(nb)
    notebook = paths.ROOT / "11.2_70_30_test_report.ipynb"
    nbf.write(nb, notebook)
    for relative, digest in manifest["protected_inputs_sha256"].items():
        assert paths.sha256(paths.WORKSPACE / relative) == digest, relative
    errors = sum(o.output_type == "error" for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", []))
    assert errors == 0
    save_json(root / "report_verification.json", dict(passed=True, error_outputs=errors,
        executed_code_cells=sum(c.cell_type == "code" for c in nb.cells),
        readme_sha256=paths.sha256(readme), notebook_sha256=paths.sha256(notebook),
        report_builder_sha256=paths.sha256(Path(__file__)),
        reused_plot_builder_sha256=paths.sha256(Path(base.__file__)),
        figure_sha256={p.name: paths.sha256(p) for p in (root / "figures").glob("*.png")}))
    print("Wrote 70:30 README and executed Notebook:", notebook, flush=True)


if __name__ == "__main__":
    build()

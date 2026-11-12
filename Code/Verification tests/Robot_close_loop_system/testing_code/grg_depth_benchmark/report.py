"""Aggregate the completed benchmark and append results to the existing Notebook."""

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import textwrap
import unittest
from urllib.parse import unquote

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = ROOT / "testing_code" / "baseline"
RESULT = ROOT / "10.18_result" / "grg_depth_benchmark"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "10.18_result" / ".cache" / "matplotlib"))
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(HERE))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat
import numpy as np
from nbclient import NotebookClient
from jupyter_client import KernelManager

from experiment import save_csv, save_json
from run_benchmark import read_events

METRICS = ["initial_rmse_c", "final_rmse_c", "rmse_reduction_c", "rmse_reduction_pct",
           "initial_variance_c2", "final_variance_c2", "variance_reduction_pct", "final_coverage95",
           "realized_information_gain_nats", "measurements", "planning_calls", "replanning_calls", "navigation_failures",
           "travel_distance_m", "travel_time_s", "sampling_time_s", "elapsed_s", "remaining_budget_s",
           "mean_planning_s", "median_planning_s", "max_planning_s", "p95_planning_s", "total_planning_s",
           "mission_wall_s", "peak_memory_mb"]


def read_json(path):
    return json.loads(path.read_text())


def read_csv(path):
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def aggregate():
    rows = [read_json(RESULT / "runs" / f"seed_{seed}" / f"depth_{depth}" / "summary.json")
            for seed in [18, 19, 20, 21, 22] for depth in [2, 3, 5]]
    for row in rows:
        if row["status"] == "SUCCESS":
            row["replanning_calls"] = row["planning_calls"] - 1
    save_csv(RESULT / "runs.csv", rows)
    screening = [read_json(RESULT / "screening" / f"depth_{d}" / "summary.json") for d in [2, 3, 5]]
    save_csv(RESULT / "screening.csv", screening)
    stats = []
    paired = []
    for depth in [2, 3, 5]:
        group = [r for r in rows if r["depth"] == depth]
        success = [r for r in group if r["status"] == "SUCCESS"]
        entry = {"depth": depth, "successful_runs": len(success),
                 "mission_timeouts": sum(r["status"] == "TIMEOUT" for r in group),
                 "failed_runs": sum(r["status"] in ["FAILED", "ERROR"] for r in group),
                 "not_run_after_screening": sum(r["status"].startswith("NOT_RUN") for r in group)}
        for metric in METRICS:
            values = [r[metric] for r in success]
            entry[metric + "_mean"] = float(np.mean(values)) if values else None
            entry[metric + "_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        stats.append(entry)
    for seed in [18, 19, 20, 21, 22]:
        base = next(r for r in rows if r["seed"] == seed and r["depth"] == 2)
        for depth in [3, 5]:
            other = next(r for r in rows if r["seed"] == seed and r["depth"] == depth)
            entry = {"seed": seed, "depth": depth, "status": other["status"]}
            if base["status"] == other["status"] == "SUCCESS":
                for metric in METRICS:
                    entry[metric + "_difference"] = other[metric] - base[metric]
                entry["mean_planning_ratio"] = other["mean_planning_s"] / base["mean_planning_s"]
                entry["total_planning_ratio"] = other["total_planning_s"] / base["total_planning_s"]
                entry["final_rmse_improvement_pct"] = 100 * (1 - other["final_rmse_c"] / base["final_rmse_c"])
                entry["final_variance_improvement_pct"] = 100 * (1 - other["final_variance_c2"] / base["final_variance_c2"])
                entry["same_executed_samples"] = (
                    RESULT / "runs" / f"seed_{seed}" / "depth_2/online_measurements.csv").read_bytes() == (
                    RESULT / "runs" / f"seed_{seed}" / f"depth_{depth}/online_measurements.csv").read_bytes()
            paired.append(entry)
    paired_stats = []
    for depth in [3, 5]:
        group = [p for p in paired if p["depth"] == depth and "final_rmse_c_difference" in p]
        entry = {"depth": depth, "complete_pairs": len(group)}
        keys = [m + "_difference" for m in METRICS] + ["mean_planning_ratio", "total_planning_ratio",
                                                      "final_rmse_improvement_pct", "final_variance_improvement_pct"]
        for key in keys:
            values = [p[key] for p in group]
            entry[key + "_mean"] = float(np.mean(values)) if values else None
            entry[key + "_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        paired_stats.append(entry)
    save_csv(RESULT / "aggregate.csv", stats)
    save_csv(RESULT / "paired_differences.csv", paired)
    save_csv(RESULT / "paired_summary.csv", paired_stats)
    calls = []
    for folder in [*(RESULT / "screening").glob("depth_*"), *(RESULT / "runs").glob("seed_*/depth_*")]:
        events = read_events(folder / "events.jsonl")
        starts, ends = {}, {}
        for event in events:
            if event["event"] == "plan_start":
                starts[event["call"]] = event
            elif event["event"] == "plan_end":
                ends[event["call"]] = event
        for index, start in starts.items():
            entry = {"run": str(folder.relative_to(RESULT)), "call": index,
                     "status": "COMPLETED" if index in ends else "INCOMPLETE",
                     "elapsed_s": start["elapsed_s"]}
            if index in ends:
                entry.update(ends[index])
            elif (folder / "execution.json").is_file():
                execution = read_json(folder / "execution.json")
                entry.update(status=execution["status"], right_censored=True,
                             observed_wall_s=execution.get("timed_out_call_s"))
            calls.append(entry)
    save_csv(RESULT / "planning_calls.csv", calls)
    return rows, screening, stats, paired, paired_stats


def figures(rows, screening):
    folder = RESULT / "figures"
    folder.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "figure.dpi": 130})
    colors = {2: "#2667a2", 3: "#bf671f", 5: "#35774a"}

    def save(fig, name):
        fig.savefig(folder / name, dpi=160, bbox_inches="tight")
        plt.close(fig)

    for metric, label, name in [
        ("final_rmse_c", "Final RMSE (°C)", "01_rmse_depth.png"),
        ("final_variance_c2", "Final mean latent variance (°C²)", "02_variance_depth.png"),
        ("realized_information_gain_nats", "Realized information gain (nats)", "03_information_depth.png"),
        ("measurements", "Successful measurements", "05_measurements_depth.png")]:
        fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
        for seed in [18, 19, 20, 21, 22]:
            group = [r for r in rows if r["seed"] == seed and r["status"] == "SUCCESS"]
            ax.plot([r["depth"] for r in group], [r[metric] for r in group], "o-",
                    alpha=.4, lw=1, ms=4, color="0.5")
        for depth in [2, 3, 5]:
            values = [r[metric] for r in rows if r["depth"] == depth and r["status"] == "SUCCESS"]
            if values:
                ax.errorbar(depth, np.mean(values), yerr=np.std(values, ddof=1) if len(values) > 1 else 0,
                            fmt="o", capsize=5, color=colors[depth], label=f"Depth {depth}: n={len(values)}")
            else:
                ax.text(depth, .5, "No completed\nmissions", transform=ax.get_xaxis_transform(),
                        ha="center", color="0.45")
        ax.set(xlabel="GRG recursive depth", ylabel=label, xticks=[2, 3, 5], xlim=(1.6, 5.4))
        ax.set_title("Grey: paired seeds; coloured: mean ± sample SD")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
        save(fig, name)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for key, marker, label in [("mean_planning_s", "o", "Mean per mission"),
                                ("max_planning_s", "s", "Maximum per mission")]:
        for depth in [2, 3, 5]:
            values = [r[key] for r in rows if r["depth"] == depth and r["status"] == "SUCCESS"]
            if values:
                axes[0].errorbar(depth + (-.07 if marker == "o" else .07), np.mean(values),
                                 yerr=np.std(values, ddof=1) if len(values) > 1 else 0,
                                 fmt=marker, color="C0" if marker == "o" else "C1", capsize=4,
                                 label=label if depth == 2 else None)
    axes[0].set(xlabel="Depth", ylabel="Planning wall time (s)", xticks=[2, 3, 5],
                xlim=(1.6, 5.4), title="Completed missions; mean ± sample SD", yscale="log")
    axes[0].legend(fontsize=8)
    for row in screening:
        depth = row["depth"]
        timed_out = row["status"] != "SUCCESS"
        duration = row.get("runtime_s", row.get("timed_out_call_s", 90))
        axes[1].bar(depth, duration, width=.6, color=colors[depth], hatch="//" if timed_out else None)
        axes[1].annotate(("≥ " if timed_out else "") + f"{duration:.2f} s",
                         (depth, duration), xytext=(0, 5), textcoords="offset points", ha="center")
    axes[1].set(xlabel="Depth", ylabel="Initial planning wall time (s)", xticks=[2, 3, 5],
                title="Seed 18 screening; hatched = timeout", yscale="log")
    axes[1].margins(y=.25)
    save(fig, "04_planning_runtime.png")

    config = read_json(BASE / "config.json")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    for ax, depth in zip(axes, [2, 3, 5]):
        run = RESULT / "runs/seed_18" / f"depth_{depth}"
        summary = read_json(run / "summary.json")
        for polygon in config["map"]["obstacles"]:
            xy = np.array(polygon)
            ax.fill(*xy.T, color=".25")
        if (run / "robot_trajectory.csv").is_file():
            trajectory = read_csv(run / "robot_trajectory.csv")
            ax.plot([float(r["x_m"]) for r in trajectory], [float(r["y_m"]) for r in trajectory],
                    color=colors[depth], lw=1)
        if (run / "online_measurements.csv").is_file():
            samples = read_csv(run / "online_measurements.csv")
            ax.scatter([float(r["x_m"]) for r in samples], [float(r["y_m"]) for r in samples],
                       color=colors[depth], s=15)
        if summary["status"] != "SUCCESS":
            ax.text(.5, .5, summary["status"].replace("_", "\n"), transform=ax.transAxes,
                    ha="center", fontsize=9)
        ax.scatter(2, 2, marker="*", s=90, color="green")
        ax.set(xlim=(0, 30), ylim=(0, 30), xlabel="x (m)", ylabel="y (m)", title=f"Depth {depth}")
        ax.set_aspect("equal")
    fig.suptitle("Seed 18 executed trajectories; unavailable runs are not imputed")
    save(fig, "06_trajectories.png")

    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    for depth in [2, 3, 5]:
        run = RESULT / "runs/seed_18" / f"depth_{depth}"
        if (run / "metrics.csv").is_file():
            metrics = read_csv(run / "metrics.csv")
            status = read_json(run / "summary.json")["status"]
            ax.plot([float(r["elapsed_s"]) / 60 for r in metrics], [float(r["rmse_c"]) for r in metrics],
                    "o-" if status == "SUCCESS" else "o--", color=colors[depth], ms=3,
                    label=f"Depth {depth}" + (" (partial)" if status != "SUCCESS" else ""))
    ax.set(xlabel="Elapsed simulated mission time (min)", ylabel="RMSE at 12.25 h (°C)",
           title="Seed 18; the reconstruction target is fixed")
    ax.legend()
    ax.grid(alpha=.2)
    save(fig, "07_rmse_time.png")


def patch_text(path, content):
    if path.exists():
        patch = f"*** Begin Patch\n*** Update File: {path}\n@@\n"
        patch += "".join("-" + line + "\n" for line in path.read_text().splitlines())
    else:
        patch = f"*** Begin Patch\n*** Add File: {path}\n"
    patch += "".join("+" + line + "\n" for line in content.splitlines())
    subprocess.run(["apply_patch"], input=patch + "*** End Patch\n", text=True, check=True, capture_output=True)


def technical_report(rows, screening, stats, paired_stats):
    lines = ["# GRG recursive depth benchmark", "", "## Objective and controls", "",
             "Test whether deeper GRG lookahead improves sampling and reconstruction enough to justify its planning latency.",
             "Only `grg.depth` changes. All runs for a seed load byte-identical copies of one initial GP checkpoint. "
             "The field, historical data, candidates, navigation, MI reward, target grid, and 900 s simulated budget are fixed.",
             "", "## Depth meaning", "",
             "Depth counts recursive left/right route-growth levels. At depth one, direct split duplicates are batched. "
             "The upper bound on sampling visits is N(d)=2 N(d-1)+1, N(0)=0: 3, 7, and 31 for depths 2, 3, and 5. "
             "This is an upper bound, not a promise of distinct or feasible stops. Budget and exclusions may reduce it. "
             "Depth 5 is accepted by the existing code; its computational feasibility is measured below.",
             "", "## Procedure", "",
             "1. Prepare one initial state for each seed. Existing checkpoints are loaded for seeds 18–20; seeds 21–22 are each trained once.",
             "2. Reproduce the seed-18 depth-2 baseline before comparisons.",
             "3. Screen one initial planning call at depths 2, 3, and 5 with a 90 s planning limit.",
             "4. Run independent missions for every screened-feasible depth across seeds 18–22. A worker is stopped after 90 s in a planning call or 900 s of total worker wall time.",
             "5. Compute realized MI from all executed samples conditioned on the original historical GP. Here realized refers to executed positions and timestamps; fixed-parameter covariance MI does not depend on their measured temperature values. Do not sum revised route rewards.",
             "6. Aggregate completed runs and paired differences. Keep timeouts and unattempted missions separate.",
             "", "## Baseline reproduction", "",
             "Seed 18, depth 2 reproduced all 14 checks, including identical routes, measurements, trajectories, and metrics. Wall times are excluded.",
             "", "## Initial planning screening", "",
             "| Depth | Status | Planning time (s) | Sampling visits | Planned MI (nats) |",
             "|---|---|---:|---:|---:|"]
    for r in screening:
        duration = r.get("runtime_s", r.get("timed_out_call_s"))
        t = f"{duration:.3f}" if r["status"] == "SUCCESS" else f"≥ {duration:.2f} (censored)"
        lines.append(f"| {r['depth']} | {r['status']} | {t} | {r.get('sampling_waypoints', '—')} | "
                     + (f"{r['gain']:.4f}" if "gain" in r else "—") + " |")
    lines += ["", "## Completed-mission comparison", "",
              "Values are mean ± sample standard deviation across successful seeds. No statistical significance is claimed.", "",
              "| Depth | Completed | Final RMSE (°C) | Final mean variance (°C²) | Realized MI (nats) | Mean planning time (s) |",
              "|---|---:|---:|---:|---:|---:|"]
    for r in stats:
        cells = []
        for metric in ["final_rmse_c", "final_variance_c2", "realized_information_gain_nats", "mean_planning_s"]:
            value, sd = r[metric + "_mean"], r[metric + "_sd"]
            cells.append("—" if value is None else f"{value:.6g} ± {sd:.3g}" if sd is not None else f"{value:.6g}")
        lines.append(f"| {r['depth']} | {r['successful_runs']}/5 | " + " | ".join(cells) + " |")
    lines += ["", "## Paired results and answers", ""]
    for r in paired_stats:
        if not r["complete_pairs"]:
            lines.append(f"- Depth {r['depth']}: no complete paired missions. Reconstruction, realized MI, route changes, and mission runtime cannot be compared from the screening result alone.")
            continue
        def val(key):
            return r[key + "_mean"]
        group = [p for p in rows if p["depth"] == r["depth"] and p["status"] == "SUCCESS"]
        diffs = list(csv.DictReader((RESULT / "paired_differences.csv").open()))
        changed = sum(p.get("same_executed_samples") == "False" for p in diffs if int(p["depth"]) == r["depth"])
        complete = [p for p in diffs if int(p["depth"]) == r["depth"] and p.get("final_rmse_c_difference")]
        wins = sum(float(p["final_rmse_c_difference"]) < 0 for p in complete)
        lines.append(f"- Depth {r['depth']} versus depth 2 ({r['complete_pairs']} pairs): mean final RMSE difference "
                     f"{val('final_rmse_c_difference'):+.6f}°C; mean final variance difference "
                     f"{val('final_variance_c2_difference'):+.6f}°C²; mean realized MI difference "
                     f"{val('realized_information_gain_nats_difference'):+.4f} nats. "
                     f"Mean planning latency ratio: {val('mean_planning_ratio'):.1f}×; total planning ratio: "
                     f"{val('total_planning_ratio'):.1f}×. Executed sample sequences changed in {changed}/{r['complete_pairs']} pairs. "
                     f"Final RMSE improved in {wins}/{r['complete_pairs']} pairs and worsened in {len(complete) - wins}/{r['complete_pairs']} pairs.")
    unavailable = [r["depth"] for r in screening if r["status"] != "SUCCESS"]
    if unavailable:
        lines.append(f"- Depth(s) {unavailable} exceeded or failed initial screening. Full missions were not attempted at those depths. They were not silently reduced or counted as successful runs.")
    lines += ["- Additional lookahead does not guarantee better realized information or RMSE under receding-horizon execution.",
              "- Depth 2 completes the validated mission with low latency. Retain it as the provisional ROS/Nav2 default unless the paired gains justify a larger planning delay. This is an engineering recommendation for this configuration, not proof of optimality.",
              "", "## Limits and interpretation", "",
              "The reward remains Sampling-Point MI, not target-field variance reduction. Five seeds are a sensitivity check, not a significance study. "
              "Compute time is excluded from simulated mission time, as in the baseline. Simulated return within 900 s does not establish a 900 s ROS wall-clock mission. Latency must be reconsidered for ROS execution. "
              "Timeout duration is right-censored: it is a lower bound, not a measured completion time. "
              "Unattempted full missions are not mission timeouts and have no final metrics. "
              "Peak memory for completed planning calls is the worker's lifetime peak RSS, not incremental GRG allocation. "
              "Planning statistics include the initial and terminal planning calls. Replanning calls equal total planning calls minus the initial call. Runs execute sequentially on one host; wall times are hardware- and load-dependent. "
              "The mean of per-seed percentage improvements differs from the percentage change between aggregate means; both use their stated denominators. "
              "The initial depth-2 export had a JSON boolean serialization issue in the benchmark wrapper; it was fixed before comparison. Its diagnostic files are excluded from all tables.",
              "", "## Run and files", "", "```bash",
              "python testing_code/grg_depth_benchmark/run_benchmark.py",
              "python testing_code/grg_depth_benchmark/report.py", "```", "",
              "- `shared/`: one immutable initial state per seed.",
              "- `screening/`: initial-call plans, events, runtimes, and timeout records.",
              "- `runs/`: each mission's configuration, checkpoints, trajectory, measurements, plans, metrics, and verification.",
              "- `runs.csv`, `aggregate.csv`, `paired_differences.csv`, `paired_summary.csv`: comparisons.",
              "- `planning_calls.csv`: completed calls and censored timeouts; planned rewards are not summed.",
              "- `figures/`: seven comparison plots.",
              "- `verification/`: baseline reproduction, original regression tests, benchmark tests, and source checks."]
    lines += ["", "## Research-question checklist", "",
              "| Question | Evidence / interpretation |", "|---|---|"]
    paired_three = next(r for r in paired_stats if r["depth"] == 3)
    if paired_three["complete_pairs"]:
        r = paired_three
        def delta(name):
            return r[name + "_difference_mean"]
        lines += [
            f"| 1. Higher realized information? | Depth 3 − 2: {delta('realized_information_gain_nats'):+.4f} nats on average over {r['complete_pairs']} complete pairs. |",
            f"| 2. Better final RMSE? | Depth 3 − 2: {delta('final_rmse_c'):+.6f}°C. Negative differences favour depth 3. |",
            f"| 3. Lower final variance? | Depth 3 − 2: {delta('final_variance_c2'):+.6f}°C². Negative differences favour depth 3. |",
            "| 4. Different routes? | Exact per-seed sample-sequence comparisons are in paired_differences.csv; see the paired results above. |",
            f"| 5. Depth-3 computation cost? | Mean paired latency ratio {r['mean_planning_ratio_mean']:.1f}×; mean paired total-planning ratio {r['total_planning_ratio_mean']:.1f}×. |"]
    else:
        lines += [f"| {i}. {question} | No complete depth-3 pairs; cannot infer mission performance from a planning screen. |"
                  for i, question in [(1, "Higher realized information?"), (2, "Better RMSE?"),
                                       (3, "Lower variance?"), (4, "Different routes?"),
                                       (5, "Depth-3 mission computation cost?")]]
    lines += ["| 6. Depth-5 cost? | Use its censored initial-call duration in the screening table. There is no measured completion or full-mission cost when screening times out. |",
              (f"| 7. Meaningful extra benefit? | Mean paired final-RMSE improvement is {paired_three['final_rmse_improvement_pct_mean']:.1f}% at {paired_three['total_planning_ratio_mean']:.1f}× total planning cost. The value of that tradeoff requires an agreed latency/accuracy criterion; no significance is claimed. |"
               if paired_three["complete_pairs"] else
               "| 7. Meaningful extra benefit? | No complete pairs; no reconstruction benefit can be established. |"),
              "| 8. Is depth 2 sufficient? | It satisfies the tested closed-loop correctness and mission goals. This does not establish global route optimality. |",
              "| 9. Best tradeoff? | No unique optimum is defined without a latency/accuracy preference. Depth 2 is the low-latency baseline; depth 3 is the higher-cost option. Depth 5 is unavailable after its failed screen. |",
              "| 10. ROS/Nav2 default? | Provisionally retain depth 2. The current experiment excludes computation from simulated time; ROS timing requires a separate validation. |"]
    return "\n".join(lines) + "\n"


def notebook_cells():
    tag = {"tags": ["grg_depth_benchmark"]}
    def md(text):
        return nbformat.v4.new_markdown_cell(textwrap.dedent(text).strip(), metadata=tag)
    def code(text):
        return nbformat.v4.new_code_cell(textwrap.dedent(text).strip(), metadata=tag)
    return [md("""
        ## 7. Benchmark: GRG recursive depth

        **Objective.** Test whether depths 3 and 5 improve realized sampling information and fixed-target reconstruction enough to justify their planning cost. Depth 2 is the frozen baseline. Only recursive depth changes.

        **Procedure.** Reproduce seed 18 at depth 2. Screen an initial call at depths 2, 3, and 5. Run independent missions for screened-feasible depths across seeds 18–22. Each seed shares one initial GP checkpoint, history, candidate set, map, and target grid. The Sampling-Point MI reward remains unchanged. Per-call and worker wall limits are 90 s and 900 s; simulated mission time is unchanged.

        Depth gives an upper bound of `2**depth - 1` sampling visits: 3, 7, and 31. Budget constraints may reduce this count. Runtime and feasibility are measured, not inferred from the bound.

        [Code](testing_code/grg_depth_benchmark/run_benchmark.py) · [Technical report](10.18_result/grg_depth_benchmark/report.md) · [All runs](10.18_result/grg_depth_benchmark/runs.csv)
        """), code("""
        from pathlib import Path
        import csv
        import json
        import numpy as np
        from IPython.display import Markdown, Image, display

        root = Path.cwd()
        bench = root / "10.18_result" / "grg_depth_benchmark"
        def load_csv(name):
            with (bench / name).open(newline="") as file:
                return list(csv.DictReader(file))
        def show_table(columns, rows):
            def fmt(value):
                return f"{value:.6g}" if isinstance(value, (float, np.floating)) else str(value)
            lines = [" | ".join(columns), " | ".join(["---"] * len(columns))]
            lines += [" | ".join(fmt(value) for value in row) for row in rows]
            display(Markdown("\\n".join(lines)))
        reproduction = json.loads((bench / "verification/baseline_reproduction.json").read_text())
        assert reproduction["passed"]
        screening = load_csv("screening.csv")
        show_table(["Depth", "Status", "Planning time (s)", "Planned samples", "Planned MI (nats)"], [
            [r["depth"], r["status"], float(r["runtime_s"]) if r.get("runtime_s") else
             ("≥ " + format(float(r.get("timed_out_call_s") or 90), ".3f")),
             r.get("sampling_waypoints") or "—", float(r["gain"]) if r.get("gain") else "—"] for r in screening
        ])
        print("Depth 2 reproduced all baseline checks. Timeout completion times are unknown.")
        """), md("""
        ### Mission results

        Realized MI is computed from the final executed samples conditioned on the shared initial GP. Revised route rewards are not summed. Missing final metrics represent uncompleted or unattempted missions, not zero performance.
        """), code("""
        runs = load_csv("runs.csv")
        def num(row, key):
            return float(row[key]) if row.get(key) else "—"
        show_table(["Seed", "Depth", "Status", "Initial RMSE (°C)", "Final RMSE (°C)",
                    "Realized MI (nats)", "Samples", "Elapsed (s)", "Home"], [
            [r["seed"], r["depth"], r["status"], num(r, "initial_rmse_c"), num(r, "final_rmse_c"),
             num(r, "realized_information_gain_nats"), r.get("measurements") or "—",
             num(r, "elapsed_s"), r.get("returned_home") or "—"] for r in runs
        ])
        stats = load_csv("aggregate.csv")
        show_table(["Depth", "Completed", "Failed", "Mission timeouts", "Not run after screening"], [
            [r["depth"], r["successful_runs"], r["failed_runs"], r["mission_timeouts"],
             r["not_run_after_screening"]] for r in stats
        ])
        def mean_sd(row, key):
            if not row.get(key + "_mean"):
                return "N/A"
            mean, sd = float(row[key + "_mean"]), row.get(key + "_sd")
            return f"{mean:.6g} ± {float(sd):.3g}" if sd else f"{mean:.6g}"
        show_table(["Depth", "Completed", "Final RMSE (°C)", "Final variance (°C²)",
                    "Realized MI (nats)", "Mean planning (s)", "Max planning (s)"], [
            [r["depth"], r["successful_runs"], mean_sd(r, "final_rmse_c"), mean_sd(r, "final_variance_c2"),
             mean_sd(r, "realized_information_gain_nats"), mean_sd(r, "mean_planning_s"),
             mean_sd(r, "max_planning_s")] for r in stats
        ])
        pairs = load_csv("paired_summary.csv")
        show_table(["Depth − 2", "Pairs", "RMSE difference (°C)", "Variance difference (°C²)",
                    "MI difference (nats)", "Mean planning ratio"], [
            [r["depth"], r["complete_pairs"], mean_sd(r, "final_rmse_c_difference"),
             mean_sd(r, "final_variance_c2_difference"), mean_sd(r, "realized_information_gain_nats_difference"),
             mean_sd(r, "mean_planning_ratio")] for r in pairs
        ])
        """), code("""
        for name in ["01_rmse_depth.png", "02_variance_depth.png", "03_information_depth.png",
                     "04_planning_runtime.png", "05_measurements_depth.png",
                     "06_trajectories.png", "07_rmse_time.png"]:
            display(Image(filename=str(bench / "figures" / name)))
        """), md("""
        ### Conclusions and verification

        The following conclusions use completed paired missions only. Five seeds do not establish statistical significance. The detailed report records timeouts and all ten research questions.
        """), code("""
        report = (bench / "report.md").read_text()
        conclusion = report.split("## Paired results and answers", 1)[1].split("## Limits", 1)[0]
        display(Markdown(conclusion.strip()))
        for name in ["regression/tests.json", "benchmark_tests.json", "baseline_sources_unchanged.json"]:
            check = json.loads((bench / "verification" / name).read_text())
            assert check["passed"], name
            print(name, "passed", check.get("count", ""))
        """)]


def run_verification():
    from run_suite import run_tests, TestResult
    verification = RESULT / "verification"
    if not (verification / "regression/tests.json").is_file():
        run_tests(verification / "regression")
    import test_benchmark
    suite = unittest.defaultTestLoader.loadTestsFromModule(test_benchmark)
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2, resultclass=TestResult).run(suite)
    (verification / "benchmark_tests.log").write_text(stream.getvalue())
    save_json(verification / "benchmark_tests.json", {"passed": result.wasSuccessful(),
                                                     "count": result.testsRun, "tests": result.records})
    print(stream.getvalue())
    assert result.wasSuccessful(), "Benchmark validation failed"


def main(build_only=False):
    rows, screening, stats, paired, paired_stats = aggregate()
    figures(rows, screening)
    run_verification()
    patch_text(RESULT / "report.md", technical_report(rows, screening, stats, paired_stats))
    path = ROOT / "10.18_test_report.ipynb"
    original = nbformat.read(path, as_version=4)
    preserved = [c for c in original.cells if "grg_depth_benchmark" not in c.metadata.get("tags", [])]
    cells = notebook_cells()
    if not build_only:
        temp = nbformat.v4.new_notebook(cells=cells, metadata=original.metadata)
        manager = KernelManager(kernel_name="python3")
        manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
        try:
            NotebookClient(temp, timeout=180, resources={"metadata": {"path": str(ROOT)}}, km=manager).execute()
        finally:
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)
        cells = temp.cells
    original.cells = preserved + cells
    nbformat.validate(original)
    patch_text(path, nbformat.writes(original) + "\n")
    save_json(RESULT / "verification/benchmark_source_hashes.json", {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(HERE.glob("*.py"))})
    links = re.findall(r"\]\(([^)]+)\)", (ROOT / "README.md").read_text() + "\n" + "\n".join(
        c.source for c in original.cells if c.cell_type == "markdown"))
    outputs = [o for c in cells if c.cell_type == "code" for o in c.outputs]
    checks = {"all_run_rows_recorded": len(rows) == 15,
              "baseline_reproduced": read_json(RESULT / "verification/baseline_reproduction.json")["passed"],
              "baseline_sources_unchanged": read_json(RESULT / "verification/baseline_sources_unchanged.json")["passed"],
              "regression_passed": read_json(RESULT / "verification/regression/tests.json")["passed"],
              "benchmark_tests_passed": read_json(RESULT / "verification/benchmark_tests.json")["passed"],
              "seven_saved_figures": len(list((RESULT / "figures").glob("*.png"))) == 7,
              "seven_embedded_figures": sum("image/png" in o.get("data", {}) for o in outputs) == 7,
              "new_cells_executed": all(c.execution_count is not None for c in cells if c.cell_type == "code"),
              "notebook_no_errors": not any(o.output_type == "error" for o in outputs),
              "local_links_exist": all((ROOT / unquote(link)).exists() for link in links
                                       if "://" not in link and not link.startswith("#"))}
    if not build_only:
        save_json(RESULT / "verification/deliverables.json", {"passed": all(checks.values()), "checks": checks})
        assert all(checks.values()), checks
    save_json(RESULT / "verification/environment.json", {"python": sys.version,
                                                         "executable": sys.executable,
                                                         "platform": platform.platform()})
    save_json(RESULT / "manifest.json", {str(p.relative_to(RESULT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                         for p in sorted(RESULT.rglob("*"))
                                         if p.is_file() and p != RESULT / "manifest.json"})
    print("Benchmark report, figures, and Notebook section saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()
    main(args.build_only)

"""Regenerate greedy / active sampling paper tables (Markdown + LaTeX) from a results directory.

DEPENDENCY-LIGHT: imports only pandas (+ stdlib). No torch/gpytorch/src. Reads the
CSVs from experiments/run_experiment.py and writes tables to <run>/tables/.

Usage:
    python scripts/make_paper_tables.py [--run results/<dir>] [--results-root results]
"""

import argparse
import os

import pandas as pd


def resolve_run(run, results_root):
    if run:
        return run
    latest = os.path.join(results_root, "LATEST.txt")
    if os.path.exists(latest):
        with open(latest, encoding="utf-8") as f:
            return os.path.join(results_root, f.read().strip())
    raise SystemExit("No --run given and results/LATEST.txt not found.")


def df_to_markdown(df, floatfmt="{:.4f}"):
    """Minimal Markdown table writer (avoids the optional 'tabulate' dependency)."""
    def fmt(v):
        if isinstance(v, float):
            return floatfmt.format(v)
        return str(v)
    cols = list(df.columns)
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *rows]) + "\n"


def write_table(df, run_dir, name, caption):
    tdir = os.path.join(run_dir, "tables")
    os.makedirs(tdir, exist_ok=True)
    with open(os.path.join(tdir, f"{name}.md"), "w", encoding="utf-8") as f:
        f.write(f"**{caption}**\n\n")
        f.write(df_to_markdown(df))
    with open(os.path.join(tdir, f"{name}.tex"), "w", encoding="utf-8") as f:
        f.write(df.to_latex(index=False, float_format="%.4f", caption=caption, label=f"tab:{name}"))
    return name


def main():
    ap = argparse.ArgumentParser(description="Regenerate tables from saved CSVs.")
    ap.add_argument("--run", default=None)
    ap.add_argument("--results-root", default="results")
    args = ap.parse_args()

    run_dir = resolve_run(args.run, args.results_root)
    metrics = pd.read_csv(os.path.join(run_dir, "metrics.csv"))
    main = metrics[metrics["Experiment"] == "main"]
    max_b = int(metrics["Samples"].max())
    written = []

    # 1) Final-budget ranking
    final = (main[main["Samples"] == max_b]
             .sort_values("RMSE")
             [["Strategy", "RMSE", "MeanVariance", "PathLength", "Coverage95", "MeanZ"]]
             .reset_index(drop=True))
    written.append(write_table(final, run_dir, "final_ranking", f"Final-budget (N={max_b}) ranking by RMSE"))

    # 2) RMSE by strategy x budget
    pivot = (main.pivot_table(index="Strategy", columns="Samples", values="RMSE")
             .reindex(["Greedy information", "Entropy", "Random", "Coarse lawnmower", "Fine lawnmower"])
             .reset_index())
    pivot.columns = ["Strategy"] + [f"N={c}" for c in pivot.columns[1:]]
    written.append(write_table(pivot, run_dir, "rmse_by_budget", "RMSE by strategy and sample budget"))

    # 3) Smart-method comparison (Greedy vs Entropy)
    smart = (main[(main["Samples"] == max_b) & (main["Strategy"].isin(["Greedy information", "Entropy"]))]
             [["Strategy", "RMSE", "MeanVariance", "PathLength", "Coverage95", "MeanZ"]]
             .reset_index(drop=True))
    written.append(write_table(smart, run_dir, "smart_comparison", f"Greedy MI vs Entropy at N={max_b}"))

    # 4) Two-sensor cold start
    ts = metrics[metrics["Experiment"] == "two_sensor"]
    if not ts.empty:
        ts_final = (ts[ts["Samples"] == max_b]
                    .sort_values("RMSE")
                    [["Strategy", "RMSE", "MeanVariance", "PathLength", "Coverage95"]]
                    .reset_index(drop=True))
        written.append(write_table(ts_final, run_dir, "two_sensor", f"2-sensor cold start at N={max_b}"))

    # 5) Border fraction
    border = os.path.join(run_dir, "border_fraction.csv")
    if os.path.exists(border):
        written.append(write_table(pd.read_csv(border), run_dir, "border_fraction", "Border fraction of selected points"))

    # 6) Output-scale comparison
    outscale = os.path.join(run_dir, "outputscale_comparison.csv")
    if os.path.exists(outscale):
        written.append(write_table(pd.read_csv(outscale), run_dir, "outputscale_comparison",
                                   "Oracle vs pilot output scale (fixed GP)"))

    print(f"Wrote {len(written)} tables (.md + .tex) to {os.path.join(run_dir, 'tables')}:")
    for n in written:
        print("  -", n)


if __name__ == "__main__":
    main()

"""Build and execute the concise results notebook from saved artifacts."""

import argparse
import json
from pathlib import Path
import subprocess
import textwrap

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "10.18_test_report.ipynb"


PREREQUISITES = r"""
## Given conditions and assumptions

The environment, temperature source, historical dataset, and candidate set are given before the mission. The robot selects sampling destinations; it does not discover the map or generate the temperature field.

**Temperature field.** Use the unchanged synthetic field from the 10.16 baseline, with inputs $(x,y,t)$ in metres and hours and output in °C:

$$
\begin{aligned}
T_{\mathrm{true}}(x,y,t)&=18+0.035x-0.025y+1.2\sin\!\left(\frac{2\pi(t-6)}{24}\right)+H-C,\\
H&=2.2\exp\!\left[-\frac12\left(\frac{x-(8+0.45t)}{5.5}\right)^2-\frac12\left(\frac{y-11}{7}\right)^2\right],\\
C&=\left[1+0.25\cos\!\left(\frac{2\pi t}{24}\right)\right]\exp\!\left[-\frac12\left(\frac{x-23}{6}\right)^2-\frac12\left(\frac{y-23}{5}\right)^2\right].
\end{aligned}
$$

This field combines a spatial background, a daily cycle, a moving warm region, and a cool region. It is synthetic, not Dashboard data. Only the measurement backend and evaluation can query truth. The GP and GRG receive observations and posterior information.

| Given item | Main-experiment setting |
|---|---|
| Known environment | 30 × 30 m; two fixed irregular obstacles, each 8.125 m²; home at (2, 2) m. |
| Navigation | 0.5 m grid; reachable free space; 0.6 × 0.4 m robot approximated by a circle plus a 0.1 m margin; speed 0.35 m/s. |
| Historical observations | 100 distinct reachable grid positions with timestamps sampled over 0–12 h. Temperature values come from the synthetic field. |
| Initial GP | Existing ConstantMean + 3D ARD RBF Exact GP. Train on history only for 300 steps. The initial field estimate is already a posterior conditioned on these 100 observations. |
| Candidate sampling nodes | 48 fixed reachable grid nodes, including home and 47 possible sampling destinations. Seeded farthest-point coverage selects them before planning, without GP uncertainty. They are possible destinations, not 48 collected measurements. |
| Mission | Start at 12 h; 900 s budget including travel, 5 s dwell per sample, and return home. Sample at actual arrival. Previously sampled nodes are excluded by default. |
| Planner | Time-aware GRG; depth 2; two split deadlines. Execute one waypoint, update the posterior, and replan. |
| Noise and model updates | Zero added historical/online measurement noise. GP likelihood noise remains positive and learned. Freeze all fitted parameters and normalization during the mission. |
| Evaluation reference | 1,393 independent accessible grid points, with 0.75 m spacing, at the fixed target time 12.25 h. Use the same truth for initial and final evaluation. |
| Reproducibility | Main seed 18, an identical repeat, and additional seeds 19 and 20. |

The RMSE comparison uses predictions at the evaluation grid: history-only posterior versus history-plus-online posterior. It is not the training error on the 100 historical observations. Motion assumes constant speed; turning, acceleration, localization error, and computation delay are excluded from mission time.
"""


def overall_conclusion(summary):
    return f"""
## 5. Overall conclusion

The implementation and correctness objectives were achieved under the stated simulation assumptions.

| Step | Objective | Outcome and evidence |
|---|---|---|
| 1. Environment | Provide a valid map and feasible sampling destinations. | Achieved. Two valid obstacles, footprint inflation, reachable candidates, and collision-free navigation passed verification. |
| 2. Historical initialization | Build the initial 3D GP without future data. | Achieved. Train on 100 observations from 0–12 h only. The initial posterior conditions on all historical data. |
| 3. Posterior queries | Supply correct mean and joint covariance to the system. | Achieved. Numerical tests agree with analytical conditioning and GPyTorch. GRG receives posterior covariance, not raw prior covariance. |
| 4. Time-aware GRG | Score ordered routes using arrival times and budget constraints. | Achieved. Controlled tests show that sampling time and route order change information gain. Planned routes include sampling and return costs. |
| 5. Navigation and sensing | Execute paths and measure at actual arrival poses and times. | Achieved. {summary['measurements']} valid samples were collected without collision. Failure tests reject unsuccessful navigation and invalid sensor data. |
| 6. Online feedback | Update after each sample and replan with the updated posterior. | Achieved. {summary['measurements']} posterior updates and {summary['planning_calls']} planning calls completed. All 112 observations remain included; parameters and normalization remain fixed. |
| 7. Mission completion | Collect at least three samples and return within 900 s. | Achieved. The robot returned at {summary['elapsed_s']:.2f} s after travelling {summary['travel_distance_m']:.2f} m. It stopped sampling when no feasible extra sample remained. |
| 8. Reconstruction | Evaluate initial and final estimates at the same target field. | Achieved. RMSE changed from {summary['initial_rmse_c']:.5f} to {summary['final_rmse_c']:.5f}°C over {summary['evaluation_count']:,} points at {summary['target_h']:.2f} h. Mean variance changed from {summary['initial_variance_c2']:.6f} to {summary['final_variance_c2']:.6f}°C². |
| 9. Verification | Check component correctness, reproducibility, and backend contracts. | Achieved. All 22 automated tests and 14 main-mission checks passed. The repeat matched all 14 reproduction checks. Seeds 19 and 20 also completed feasible missions and reduced final RMSE. Navigation and temperature contracts are defined and tested with Python backends. |

Online sampling improved final reconstruction in all three distinct seeds. Fixed-query variance did not increase during updates. Main-run 95% coverage was 100% before and after sampling, indicating conservative intervals on this field. Real-data uncertainty calibration, planner superiority, and ROS 2/Nav2 operation remain unvalidated by this experiment.
"""


def build():
    cells = []
    summary = json.loads((ROOT / "10.18_result/baseline/main/mission_summary.json").read_text())

    def md(text):
        cells.append(nbformat.v4.new_markdown_cell(textwrap.dedent(text).strip()))

    def code(text):
        cells.append(nbformat.v4.new_code_cell(textwrap.dedent(text).strip()))

    md("""
    # 10.18 — 3D GP + GRG closed-loop test results

    This is Johnson Ji 10.18 test, focus on the python integration and implementation of 3D GP + GRG, which is in order to achieve the closed loop navigation and temperature measurement, rebuild and prediction.

    Objective: validate historical GP initialization, time-aware GRG, simulated navigation,
    online conditioning, and fixed-time reconstruction. All temperatures are synthetic.
    [Method and rubric](README.md) · [Code audit](10.18_result/baseline/code_audit.md).
    """)
    md(PREREQUISITES)
    code("""
    from pathlib import Path
    import csv
    import json
    import os
    import sys
    import warnings

    root = Path.cwd()
    if root.name == "baseline" and root.parent.name == "testing_code":
        root = root.parent
    if root.name == "testing_code":
        root = root.parent
    assert (root / "10.18_result" / "baseline" / "main").is_dir()
    os.environ["MPLCONFIGDIR"] = str(root / "10.18_result" / ".cache" / "matplotlib")
    sys.path.insert(0, str(root / "testing_code" / "baseline"))
    warnings.filterwarnings("ignore", message=r"`torch.jit.script` is deprecated.*", category=FutureWarning)

    import numpy as np
    import torch
    from IPython.display import Image, Markdown, display
    from gp import GPModel
    from grg import GRGPlanner, information_gain
    torch.set_num_threads(1)

    result = root / "10.18_result" / "baseline"
    main = result / "main"
    read_json = lambda path: json.loads(path.read_text())

    def read_csv(path):
        with path.open(newline="") as file:
            return list(csv.DictReader(file))

    def table(columns, rows):
        def cell(value):
            return f"{value:.6g}" if isinstance(value, (float, np.floating)) else str(value)
        lines = [" | ".join(columns), " | ".join(["---"] * len(columns))]
        lines += [" | ".join(cell(value) for value in row) for row in rows]
        display(Markdown("\\n".join(lines)))

    def figure(name):
        display(Image(filename=str(main / "figures" / name)))

    config = read_json(main / "config.json")
    summary = read_json(main / "mission_summary.json")
    checks = read_json(main / "verification.json")
    tests = read_json(result / "verification" / "tests.json")
    """)
    md("""
    ## 1. Conditions and correctness tests

    Use historical measurements only for training. Freeze kernel, mean, noise, and scaling
    during the mission. Sample at arrival, then dwell for 5 s. GRG uses recursive depth 2
    and two split deadlines. It executes only the next waypoint before replanning.

    Source: [configuration](testing_code/baseline/config.json),
    [component tests](testing_code/baseline/test_components.py), [system tests](testing_code/baseline/test_system.py).
    Full rerun: `python testing_code/baseline/run_suite.py`.
    """)
    code("""
    hp = read_json(main / "gp_training_summary.json")
    map_info = read_json(main / "map_summary.json")
    table(["Condition", "Value"], [
        ["Seed", config["seed"]],
        ["History", "100 points; 0–12 h; no added noise"],
        ["Mission / target", "12 h start; 900 s budget; 12.25 h target"],
        ["Candidates / evaluation points", f"{summary['candidate_count']} / {summary['evaluation_count']}"],
        ["Obstacle areas (m²)", map_info["obstacle_areas_m2"]],
        ["Footprint + margin radius (m)", map_info["footprint_radius_with_margin_m"]],
        ["Learned lengthscales (m, m, h)", [round(x, 4) for x in hp["lengthscales_m_m_h"]]],
        ["Learned noise variance (°C²)", hp["noise_variance_c2"]],
        ["Training steps", hp["selected_iteration"]],
    ])
    groups = {}
    for item in tests["tests"]:
        group = item["test"].split(".")[1][4]
        groups.setdefault(group, []).append(item["status"])
    table(["Test", "Passed", "Total"], [
        [group, statuses.count("passed"), len(statuses)]
        for group, statuses in sorted(groups.items())
    ])
    assert tests["passed"] and checks["passed"]
    print(f"{tests['count']} automated tests passed; {len(checks['checks'])} main-mission checks passed.")
    """)
    md("""
    ### Posterior and temporal checks

    Objective: verify conditioned covariance and time-dependent information gain.
    The 12 h delay below is a controlled query test, not an extra mission.
    Joint covariance must agree between the GPyTorch model and the batched GRG evaluator.
    """)
    code("""
    initial_gp = GPModel.load(main / "initial_gp.pt")
    final_gp = GPModel.load(main / "final_gp.pt")
    points = np.array([[float(r["x_m"]), float(r["y_m"])] for r in read_csv(main / "candidates.csv")])
    travel = np.load(main / "travel_time_s.npy")
    planner = GRGPlanner(points, travel, 12, 5, depth=2, time_splits=2)
    route = [0, 1, 2, 0]
    query, arrivals, finish = planner.schedule(route, 0)
    _, cov = initial_gp.joint(query)
    error = np.max(np.abs(cov - initial_gp.covariance_many(query[None])[0]))
    early, late = [planner.score_routes([route], initial_gp, elapsed)[0] for elapsed in (0, 43200)]
    assert error < 1e-9
    assert np.linalg.eigvalsh(cov).min() >= -1e-10
    assert abs(late - early) > 1e-8
    slow = GRGPlanner(points, travel * 100, 12, 1000)
    forward, reverse = slow.score_routes([route, [0, 2, 1, 0]], initial_gp, 0)
    assert abs(forward - reverse) > 1e-8
    table(["Check", "Value"], [
        ["Covariance evaluator max error (°C²)", error],
        ["Prior marginal variance (°C²)", hp["output_variance_c2"]],
        ["Conditioned marginal variances (°C²)", np.diag(cov).round(6).tolist()],
        ["Route information at mission start (nats)", early],
        ["Same route after 12 h delay (nats)", late],
        ["Slow controlled route, order 1→2 (nats)", forward],
        ["Slow controlled route, order 2→1 (nats)", reverse],
    ])
    """)
    md("""
    Conclusion: covariance includes historical conditioning. Changing arrival times or
    route order changes the reward. Automated tests also verify analytical conditioning,
    fixed parameters, variance reduction, and temperature-independent covariance.
    """)
    md("""
    ## 2. Closed-loop mission

    Objective: complete multiple sampling and replanning cycles, then return within budget.
    The controller uses actual navigation feedback. The tests include delayed arrivals,
    navigation failures before and after motion, invalid measurements, empty paths, and insufficient budget.

    Source: [controller](testing_code/baseline/mission.py), [GRG](testing_code/baseline/grg.py),
    [navigation](testing_code/baseline/navigation.py), [interfaces](testing_code/baseline/interfaces.py).
    """)
    code("""
    table(["Metric", "Result"], [
        ["Successful measurements / GP updates", summary["measurements"]],
        ["GRG calls", summary["planning_calls"]],
        ["Travel distance (m)", summary["travel_distance_m"]],
        ["Travel time (s)", summary["travel_time_s"]],
        ["Total mission time (s)", summary["elapsed_s"]],
        ["Remaining budget at home (s)", summary["budget_s"] - summary["elapsed_s"]],
        ["Returned home", summary["returned_home"]],
        ["Information gain (nats)", summary["information_gain_nats"]],
    ])
    log = [r for r in read_csv(main / "planning_log.csv") if r["gp_updated"] == "True"]
    table(["Sample", "Node", "Arrival (s)", "GP timestamp (h)", "Temperature (°C)"], [
        [i + 1, r["next_node"], float(r["arrival_s"]), float(r["arrival_t_h"]), float(r["temperature_c"])]
        for i, r in enumerate(log)
    ])
    figure("01_navigation.png")
    figure("06_progress.png")
    """)
    md(f"""
    Conclusion: the default mission completed {summary['measurements']} measurements and {summary['planning_calls']} planning calls.
    The last call found no feasible extra sample. The robot returned without collision
    at {summary['elapsed_s']:.2f} s. All 14 saved mission checks passed. See the full
    [planning log](10.18_result/baseline/main/planning_log.csv) and
    [verification](10.18_result/baseline/main/verification.json).
    """)
    md(f"""
    ## 3. Fixed-target reconstruction

    Objective: reconstruct the same {summary['evaluation_count']:,} accessible points at {summary['target_h']:.2f} h before and after sampling.
    The evaluation grid is independent of the candidate set. Coverage uses latent intervals
    `mean ± 1.96 × latent standard deviation`. Sensor noise is not added to these intervals.

    Source: [GP updates](testing_code/baseline/gp.py), [evaluation](testing_code/baseline/evaluate.py),
    [figures](testing_code/baseline/plots.py).
    """)
    code("""
    saved = np.load(main / "reconstruction.npz")
    query, truth = saved["query"], saved["truth"]
    # Replay both saved models and verify the reported fields.
    for model, index in [(initial_gp, 0), (final_gp, -1)]:
        prediction = model.predict(query)
        np.testing.assert_allclose(prediction["mean"], saved["means"][index], atol=1e-10, rtol=0)
        np.testing.assert_allclose(prediction["latent_var"], saved["variances"][index], atol=1e-10, rtol=0)
    rows = []
    for label, index in [("Initial", 0), ("Final", -1)]:
        error = saved["means"][index] - truth
        variance = saved["variances"][index]
        rows.append([label, np.sqrt(np.mean(error**2)), np.mean(np.abs(error)),
                     variance.mean(), np.mean(np.abs(error) <= 1.959963984540054 * np.sqrt(variance)),
                     np.mean(2 * 1.959963984540054 * np.sqrt(variance))])
    table(["State", "RMSE (°C)", "MAE (°C)", "Mean variance (°C²)", "95% coverage", "Mean interval width (°C)"], rows)
    assert np.max(np.diff(saved["variances"], axis=0)) <= 1e-9
    for name in ["02_temperature.png", "03_uncertainty.png", "04_rmse.png", "05_variance.png"]:
        figure(name)
    """)
    md(f"""
    Conclusion: RMSE changed from {summary['initial_rmse_c']:.5f}°C to {summary['final_rmse_c']:.5f}°C
    ({100 * (1 - summary['final_rmse_c'] / summary['initial_rmse_c']):.1f}% reduction).
    Mean latent variance changed from {summary['initial_variance_c2']:.6f} to {summary['final_variance_c2']:.6f}°C²
    ({100 * (1 - summary['final_variance_c2'] / summary['initial_variance_c2']):.1f}% reduction). RMSE rose slightly after the first sample;
    it need not decrease after every update. Variance did not increase at any fixed query.
    Both initial and final coverage were 100%. This indicates conservative intervals on
    this noise-free field, not calibrated 95% coverage on real measurements.
    """)
    md("""
    ## 4. Reproducibility and runtime

    Objective: repeat seed 18 and check two additional seeds without changing the method.
    Report runtime separately from simulated mission time. GRG query time is included in GRG time.
    """)
    code("""
    runs = read_csv(result / "seed_summary.csv")
    table(["Run", "Samples", "Initial RMSE (°C)", "Final RMSE (°C)", "Elapsed (s)", "Home"], [
        [r["run"], r["measurements"], float(r["initial_rmse_c"]), float(r["final_rmse_c"]),
         float(r["elapsed_s"]), r["returned_home"]] for r in runs
    ])
    repeat = read_json(result / "verification" / "reproducibility.json")
    assert repeat["passed"]
    print(f"Fixed-seed reproduction: {sum(repeat['checks'].values())}/{len(repeat['checks'])} checks passed.")
    table(["Main-run runtime", "Seconds"], [
        ["Map and travel precomputation", summary["map_and_travel_precompute_s"]],
        ["GP training", summary["gp_training_s"]],
        ["GRG total", summary["grg_runtime_s"]],
        ["GRG posterior queries (subset)", summary["grg_covariance_query_s"]],
        ["Full run including plots and file output", summary["total_wall_s"]],
    ])
    """)
    md("""
    Conclusion: all three distinct seeds completed feasible missions and reduced final RMSE.
    The repeated run matched measurements, trajectories, routes, and predictions exactly.
    Three seeds are a smoke test, not evidence of planner superiority.
    """)
    md(overall_conclusion(summary))
    md("""
    ## 6. Next step?

    The Python GP–GRG–navigation–measurement loop is operational under the tested assumptions.
    The mission is short relative to the learned 7.41 h temporal lengthscale. Controlled tests
    establish time dependence; the default mission alone does not establish a temporal-planning benefit.
    Fixed-kernel covariance planning responds to sampling locations and times. Temperature values
    update the mean but do not directly change the route reward.

    The simulator omits turning, acceleration, localization error, and compute delay in mission time.
    GRG maximizes sampled-point information, not target-grid RMSE. Next, implement real sensor and
    Nav2 adapters and validate timing and return reserves. Planner comparisons remain separate.
    """)
    notebook = nbformat.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}})
    content = nbformat.writes(notebook) + "\n"
    old = PATH.read_text()
    patch = "*** Begin Patch\n*** Update File: " + str(PATH) + "\n@@\n"
    patch += "".join("-" + line + "\n" for line in old.splitlines())
    patch += "".join("+" + line + "\n" for line in content.splitlines())
    patch += "*** End Patch\n"
    subprocess.run(["apply_patch"], input=patch, text=True, check=True, capture_output=True)
    return notebook


def execute(notebook):
    import sys
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    client = NotebookClient(notebook, timeout=180, resources={"metadata": {"path": str(ROOT)}}, km=manager)
    try:
        client.execute()
    finally:
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
    nbformat.validate(notebook)
    nbformat.write(notebook, PATH)
    print(f"Saved executed report: {PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()
    notebook = build()
    if not args.build_only:
        execute(notebook)

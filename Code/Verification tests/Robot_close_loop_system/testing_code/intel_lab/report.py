"""Write current Intel reports and replace the main Notebook with measured-data results."""

import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import textwrap

from data import ROOT, PROCESSED, DATA
from experiment import save_json
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager


def read(path):
    return json.loads(path.read_text())


def table(headers, records):
    lines=["| "+" | ".join(headers)+" |", "|"+"---|"*len(headers)]
    for row in records:
        lines.append("| "+" | ".join(f"{v:.6g}" if isinstance(v,float) else str(v) for v in row)+" |")
    return "\n".join(lines)


def main():
    os.environ.setdefault("IPYTHONDIR",str(ROOT/"10.18_result/.cache/ipython"))
    os.environ.setdefault("JUPYTER_RUNTIME_DIR",str(ROOT/"10.18_result/.cache/jupyter"))
    path=ROOT/"10.18_test_report.ipynb"
    archive=ROOT/"10.18_result/baseline/archived_notebooks/10.18_synthetic_report.ipynb"
    if path.exists():
        previous=nbformat.read(path,as_version=4)
        current_tags={"intel_lab","grg_policy_verification"}
        historical=[c for c in previous.cells if not current_tags.intersection(c.metadata.get("tags",[]))]
        if historical:
            if archive.exists():
                assert nbformat.read(archive,as_version=4).cells==historical,"Historical archive differs; do not overwrite it"
            else:
                archive.parent.mkdir(parents=True,exist_ok=True)
                previous.cells=historical
                nbformat.write(previous,archive)
    result=ROOT/"10.18_result"
    forecast=result/"baseline/intel_lab/forecast"
    replay=result/"baseline/intel_lab/replay"
    depth=result/"grg_depth_benchmark/intel_lab"
    correctness=result/"grg_correcness/current"
    split=read(PROCESSED/"split.json")
    f=read(forecast/"summary.json");m=read(replay/"summary.json")
    d=read(depth/"depth_summary.json")
    tests=read(correctness/"new_tests.json");regressions=read(correctness/"regressions.json")
    numerical=read(correctness/"numerical_summary.json")
    replay_tests=read(correctness/"intel_replay_tests.json")
    source="[Intel Lab repository](https://github.com/linsea423/Intel_Lab_Data); [original dataset](https://db.csail.mit.edu/labdata/labdata.html). Pinned commit: `f46a8102e0ae93c98969e919c0b68897a202e0f3`."
    protocol="""Training: 2004-02-29 to 2004-03-02, inclusive. Test: 2004-03-03. Time zero is 2004-02-29 00:00; original timezone is unspecified. Training spans [0,72) h; test spans [72,96) h.

Select exactly 2,000 actual observations: 667/667/666 per day. Within each day, allocate near-equal quotas across available sensors and choose distinct records nearest equally spaced time targets. Seed 18 assigns remainder quotas. Training covers 53 sensors. The cleaned training pool has 280,944 rows; the complete cleaned test day has 90,690 rows from 52 sensors. Keep original record IDs and acquisition times.

Apply a fixed 0 to 50 C indoor plausibility screen, require known coordinates and finite temperature, and preserve empty fields when parsing. This excludes 151 historical and 205 test temperature rows; 40 additional window rows lack sensor/temperature values. This screen is a documented preprocessing assumption, not proof of sensor accuracy. No imputation or temperature interpolation is used. The fourth-day labels never select hyperparameters, normalization, or training checkpoints.

Reuse the existing ConstantMean + 3D ARD RBF Exact GP, CPU float64, 300 Adam steps, and existing initialization/settings. The fit learns mean, lengthscales, output variance and likelihood noise. No synthetic noise is added to real records. Observation intervals include learned likelihood noise. All depths share the identical saved initial GP and 2,000-row history.
"""
    forecast_text=f"""# Intel Lab: three-day history and fourth-day forecast

## Data and method

{source}

{protocol}

## Results

Predict every valid fourth-day record from the fixed origin at 72 h. No online observations are added in this forecast.

{table(['Metric','Result'],[['Training observations',2000],['Test observations',f['test_count']],['RMSE (C)',f['metrics']['rmse_c']],['MAE (C)',f['metrics']['mae_c']],['95% observation coverage',f['metrics']['coverage_95']],['Mean 95% interval width (C)',f['metrics']['interval_width_95_c']],['Training wall time (s)',f['training_s']],['Prediction wall time (s)',f['prediction_s']]])}

The current RBF has limited full-day forecasting accuracy. Coverage must be read with the wide intervals. These measured-data results are separate from the earlier synthetic results. This run does not establish that the kernel alone causes the errors.

## Files

`predictions.csv` preserves each test record, mean, latent variance and observation interval. `hourly_metrics.csv` reports forecast-horizon errors. `sensor_metrics.csv` reports sensor-level errors. `hyperparameters.json` records the fit. `figures/` contains the forecast and calibration plots. Selected training records and all cleaned test records are in `Data/Intel_lab_data/processed/three_day_2000/`.

![Forecast metrics](figures/forecast_metrics.png)
![Measured and predicted sensor temperatures](figures/sensor_forecasts.png)
"""
    (forecast/"report.md").write_text(forecast_text)
    replay_text=f"""# Intel Lab: GP-GRG baseline replay

## Conditions

{source}

{protocol}

The user authorized moving Home to the nearest reachable sensor: sensor 16 at (1.5,2) m. Keep the existing 30 x 30 m map, two synthetic obstacles, footprint, 0.5 m navigation grid, 0.35 m/s speed, 5 s dwell, depth-2 default and 900 s budget. This is a replay on a simulated navigation map, not a reconstruction of laboratory navigation geometry.

Use the 29 real sensor coordinates that are footprint-reachable on this map. The other 23 valid test sensors are outside or unreachable. Replacing the original 48 synthetic candidate locations is necessary for real-record replay and was explicitly agreed. Home sampling and spatial revisits remain enabled.

Sort each test sensor's records by time. Even-indexed records form the acquisition pool; odd-indexed records form a disjoint evaluation pool. This split uses time/order only. GRG can see acquisition timestamps to estimate availability but never future temperature values. On arrival, wait for the next available record, acquire it at its actual timestamp, then consume 5 s dwell. The same raw record cannot be consumed twice. Consecutive sampling is permitted; the record schedule determines acquisition delay.

Mission time = travel + acquisition waiting + dwell + online computation. Waiting is included in the logged sampling component; computation remains 0 s in the mission clock. Measured GRG wall time remains a separate diagnostic. No active wait action, revisit interval or retry limit was added.

## Reconstruction reference

Target grid: same 0.75 m spacing and accessible map domain, now at day-4 target 72.25 h. Dense maps show GP predictions and latent uncertainty only; no real dense-grid truth is available.

RMSE uses {m['evaluation_count']} fixed held-out real records, one per available sensor where a reference exists in the five minutes before 72.25 h. Their actual timestamps span {m['evaluation_time_range_h'][0]:.6f} to {m['evaluation_time_range_h'][1]:.6f} h. These are not simultaneous dense-field truth. The final posterior may use online samples later than some reference records, so this is retrospective reconstruction; the separate forecast experiment tests future prediction without online data.

## Results

{table(['Item','Result'],[['Measurements',m['measurements']],['Planning calls',m['planning_calls']],['Returned Home',m['returned_home']],['Mission time (s)',m['elapsed_s']],['Acquisition waiting (s)',m['measurement_wait_time_s']],['Dwell (s)',m['dwell_time_s']],['Initial held-out RMSE (C)',m['initial_metrics']['rmse_c']],['Final held-out RMSE (C)',m['final_metrics']['rmse_c']],['Executed joint MI (nats)',m['information_gain_nats']]])}

All saved replay acceptance checks are in `verification.json`. Lower RMSE is a performance outcome, not a correctness requirement. Sampling-point MI remains the planning objective; the held-out reconstruction targets do not influence route selection.

## Figures

![Navigation](figures/navigation.png)
![Predicted temperature](figures/temperature_maps.png)
![Latent uncertainty](figures/uncertainty_maps.png)
![Held-out reconstruction metrics](figures/reconstruction_metrics.png)
"""
    (replay/"report.md").write_text(replay_text)
    depth_text=f"""# Intel Lab: recursive-depth benchmark

Use the three-day 2,000-record GP and fourth-day replay described in [baseline report](../../baseline/intel_lab/replay/report.md). Test depths 1, 2 and 3 with identical history, checkpoint, candidates, Home, map, record availability, evaluation records and mission budget. Do not run depth 5.

{table(['Depth','Status','Samples','Final RMSE (C)','MI (nats)','GRG wall time (s)'],[[r['depth'],r['status'],r.get('measurements'),r.get('final_rmse_c'),r.get('mi_nats'),r.get('grg_runtime_s')] for r in d])}

An external watchdog limits each planning call to 90 s and each mission worker to 900 s wall time. These are runtime bounds, not simulated mission charges. Timeouts and unattempted missions are reported as such. The paired depth-2 repeat checks deterministic observations, routes, times and metrics. There is one fixed real dataset, not multiple independent stochastic trials; no statistical significance or global-optimality claim is made.

Setup failures, if present in `failed_attempts/`, preserve failed serialization attempts. The corrected runs are reported above. They are not planner correctness failures.

![Depth results](figures/depth_comparison.png)
"""
    (depth/"report.md").write_text(depth_text)
    correctness_text=f"""# Updated GRG correctness verification

## Policies and implementation

Allow all intentional repeated spatial events, including A-B-A and A-A. The start vertex denotes robot state; intermediate vertices denote measurements; the final Home vertex denotes return only. Every intentional event pays dwell and keeps its own time and joint-covariance entry. Pass the completed left prefix directly to the right recursive branch, so a shared junction is represented once. A new explicit insertion at the same node is a separate event.

Remove residual filters that excluded the current non-Home location or a matching recursive sampling endpoint. Retain the existing recursive search, budget splits, full-route MI, kernel and GP update semantics. Chunking score batches bounds temporary covariance memory without changing their values.

Temporary failure during cycle k excludes the node in k+1 and restores eligibility in k+2. Only top-level cycles count. Static collision/no-path evidence permits permanent exclusion. Invalid measurements do not update the GP; the existing one-cycle measurement-failure hook remains. If no feasible alternative exists during exclusion, stop and return safely. Remove the earlier empty-cycle deferral. Recovery tests now execute a genuine alternative measurement between failure and re-eligibility.

The existing partial-motion fallback returns from the actual pose and charges consumed travel. Replanning from arbitrary non-candidate poses remains unimplemented. The original 200-iteration guard remains a mission safety bound, not a new navigation retry limit. Home sampling stays enabled because the user explicitly authorized it before this task.

## Results

Eight updated categories: exhaustive bounded repeated-event routes; recursive deadlines/prefixes/junctions; consecutive and nonconsecutive revisits; cache consistency; time-aware posterior MI; dwell/budget/return boundaries; candidate eligibility/failures; obstacle paths and prior depth-3 reproduction.

- Updated recursive checks: {tests['count']} run, {tests['failures']} failures, {tests['errors']} errors.
- Current baseline regressions: see `baseline_regression.log` (52 tests).
- Historical benchmark artifact/watchdog regressions: see `benchmark_regression.log` (6 tests).
- Recorded-data checks: {replay_tests['count']} tests; passed = {replay_tests['passed']}.
- Maximum production versus independent joint-MI difference: {numerical['maximum_mi_difference_nats']:.3g} nats.
- Recursive subproblems checked: {numerical['recursive_subproblems_checked']}.
- Cache-enabled versus disabled states checked: {numerical['cache_states_checked']}.
- Bounded exhaustive configurations: {numerical['bounded_exhaustive_cases']}.

The GP independent reference uses its fitted kernel and a separate conditioning solve and Cholesky log determinant. Route references independently sum travel, event dwell and final return. Tests compare event times, finish times and MI. The exhaustive event bound is three with an 18 s budget and 5 s dwell; this bounds enumeration, not production revisits. Optimality gaps are diagnostics only.

Existing navigation, posterior, invalid-measurement and return tests are retained. Update old assumptions requiring [0,A,0] when consecutive A samples are now valid; verify feasibility and event costs instead. Update cooldown recovery fixtures to execute a real alternative action. Add a strict no-alternative termination check. The obsolete spatial-uniqueness test is replaced by event-consistency checks, while numerical and obstacle checks remain.

## Prior depth-3 case

Analyze [0,3,1,2,3,0] directly, independently of whether today's planner selects it. It contains four intended measurements, including two at node 3, each with its own time and 5 s dwell. Joint covariance includes all four. Independent timing, MI, obstacle paths and the 900 s budget pass. The old spatial-uniqueness failure is obsolete under the authorized policy. A duplicated recursive junction would still be a defect; prefix and deadline tracing checks this distinction. See `obstacle_map_reproduction.json`.

## Limits

These are bounded simulation tests, not proof of global optimality or verification for arbitrary depth. No depth-5 benchmark is executed. ROS/Nav2, dynamic obstacle recovery from arbitrary poses, real sensor timing adapters, and nonzero online computation charges remain future work. Recorded replay adds explicitly authorized acquisition waits; the synthetic timing convention is unchanged.
"""
    (correctness/"report.md").write_text(correctness_text)
    # The main report contains the current Intel experiment only. Historical
    # synthetic cells and embedded outputs are preserved in the separate archive.
    cells=[]
    def md(text,tag="intel_lab"):
        c=nbformat.v4.new_markdown_cell(inspect.cleandoc(text));c.metadata.tags=[tag];cells.append(c)
    def code(text,tag="intel_lab"):
        c=nbformat.v4.new_code_cell(textwrap.dedent(text).strip());c.metadata.tags=[tag];cells.append(c)
    md("""# 10.18 — Intel Lab temperature prediction and GP–GRG replay

    Objective: evaluate the existing 3D ARD RBF GP on recorded temperatures, then verify closed-loop sampling under the revised GRG policies.

    Source: [Intel Lab repository](https://github.com/linsea423/Intel_Lab_Data), [original dataset](https://db.csail.mit.edu/labdata/labdata.html).
    Train on 2004-02-29 to 2004-03-02; test on 2004-03-03. Select exactly 2,000 real records with day quotas 667/667/666, near-equal sensor quotas and equally spaced time targets. Fixed seed 18 assigns remainders. Preserve original coordinates, record IDs and times. Use a fixed 0–50°C plausibility screen; no imputation or interpolation.

    Reuse the existing 3D ARD RBF GP and 300-step fit. Fit normalization and all parameters on the selected history only. All temperature-performance tables and figures below come from the Intel experiment. Dense maps are GP estimates; the map and navigation remain simulated. Section 4 reports program correctness checks separately.

    ## Data and fitted model

    Historical interval: [2004-02-29 00:00, 2004-03-03 00:00). Test interval: [2004-03-03 00:00, 2004-03-04 00:00). Time is continuous hours from 2004-02-29 00:00; original timezone is unspecified. The cleaned history pool has 280,944 records. The test day has 90,690 valid records. The fixed screen excludes 151 history and 205 test temperature records; 40 additional four-day records lack sensor/temperature values. This screen does not certify all retained readings.
    """)
    code("""
    from pathlib import Path
    import csv
    import json
    from IPython.display import Image, Markdown, display
    root = next(p for p in [Path.cwd(), *Path.cwd().parents]
                if (p / 'testing_code/baseline/config.json').is_file())
    result = root / '10.18_result'
    intel = result / 'baseline/intel_lab'
    forecast = json.loads((intel / 'forecast/summary.json').read_text())
    mission = json.loads((intel / 'replay/summary.json').read_text())
    depth = json.loads((result / 'grg_depth_benchmark/intel_lab/depth_summary.json').read_text())
    workspace = root.parents[1]
    split = json.loads((workspace / 'Data/Intel_lab_data/processed/three_day_2000/split.json').read_text())
    def show(columns, rows):
        lines = [' | '.join(columns), ' | '.join(['---'] * len(columns))]
        lines += [' | '.join(f'{v:.6g}' if isinstance(v, float) else str(v) for v in row) for row in rows]
        display(Markdown('\\n'.join(lines)))
    show(['Condition', 'Value'], [[key, split[key]] for key in
         ['train_interval', 'test_interval', 'train_count', 'train_rows_per_day', 'train_sensor_count', 'test_count']])
    selected_path = workspace / 'Data/Intel_lab_data/processed/three_day_2000/train_2000.csv'
    with selected_path.open() as stream:
        selected = list(csv.DictReader(stream))
    assert len(selected) == 2000
    assert len({r['record_id'] for r in selected}) == 2000
    show(['Selected real record', 'Sensor', 'Timestamp', 'x (m)', 'y (m)', 'Temperature (C)'],
         [[r[k] for k in ['record_id','sensor_id','timestamp','x_m','y_m','temperature_c']] for r in selected[:5]])
    hp = json.loads((intel / 'forecast/hyperparameters.json').read_text())
    show(['Fitted parameter', 'Value'], [[k, hp[k]] for k in
         ['kernel','lengthscales_m_m_h','output_variance_c2','noise_variance_c2','mean_c','selected_iteration']])
    """)
    md("""## 1. Fourth-day temperature prediction

    Goal: forecast the complete test day from the fixed origin at 72 h. Predict at the actual sensor positions and recorded timestamps of all 90,690 valid measurements. Add no fourth-day data to the GP. Compare predictive mean with measured temperature. Report observation intervals, including learned likelihood noise. Source: [forecast.py](testing_code/intel_lab/forecast.py).
    """)
    code("""
    show(['Metric', 'Value'], list(forecast['metrics'].items()))
    show(['Timing', 'Seconds'], [[k, forecast[k]] for k in ['training_s', 'prediction_s']])
    for name in ['forecast_metrics.png', 'sensor_forecasts.png']:
        display(Image(filename=str(intel / 'forecast/figures' / name)))
    """)
    md(f"Conclusion: RMSE is {f['metrics']['rmse_c']:.3f}°C and MAE is {f['metrics']['mae_c']:.3f}°C. Coverage is {100*f['metrics']['coverage_95']:.2f}% with mean interval width {f['metrics']['interval_width_95_c']:.3f}°C. The current RBF has limited full-day forecast accuracy. The mean approaches its learned constant away from history, while uncertainty widens. No kernel comparison or performance tuning was performed.")
    md("""## 2. Intel GP–GRG baseline replay

    Goal: test recorded sampling and online reconstruction. Keep the 30×30 m simulated map. Home is sensor 16 at (1.5,2) m; use 29 footprint-reachable sensor candidates. Allow revisits and Home measurements. Wait for the next available record, charge waiting and 5 s dwell, and condition on its actual time. Only acquisition timestamps enter planning; future temperature values do not.

    The mission covers the first 900 s of the fourth day. Within each sensor, even-indexed time-ordered records form the acquisition pool; odd-indexed records form a disjoint evaluation pool. GRG uses the same posterior sampling-point joint MI, evaluated at actual estimated acquisition times.

    RMSE uses 28 fixed held-out records in the final five-minute reference window, with their actual recorded temperatures and timestamps. These timestamps differ. Final reconstruction can use online measurements later than some reference records. The 0.75 m dense maps at 72.25 h show GP predictions and latent uncertainty; dense real-field error is unavailable. This short retrospective reconstruction and the full-day forecast have different targets. Source: [replay.py](testing_code/intel_lab/replay.py).
    """)
    code("""
    show(['Mission', 'Value'], [[k, mission[k]] for k in ['measurements', 'planning_calls', 'elapsed_s', 'returned_home', 'measurement_wait_time_s', 'dwell_time_s', 'information_gain_nats']])
    show(['Stage', 'RMSE (C)', 'Mean latent variance (C2)'],
         [[name, mission[key]['rmse_c'], mission[key]['mean_variance_c2']] for name,key in
          [('History only','initial_metrics'),('After online samples','final_metrics')]])
    display(json.loads((intel / 'replay/verification.json').read_text()))
    for name in ['navigation.png', 'temperature_maps.png', 'uncertainty_maps.png', 'reconstruction_metrics.png']:
        display(Image(filename=str(intel / 'replay/figures' / name)))
    """)
    md(f"Conclusion: the robot collected {m['measurements']} distinct real records and returned at {m['elapsed_s']:.2f} s. Held-out RMSE increased from {m['initial_metrics']['rmse_c']:.5f} to {m['final_metrics']['rmse_c']:.5f}°C. Mean latent variance decreased from {m['initial_metrics']['mean_variance_c2']:.5f} to {m['final_metrics']['mean_variance_c2']:.5f}°C². This run reduces modeled uncertainty but does not improve temperature accuracy on the reference records. All recorded-data mission checks pass.")
    md("""## 3. Recursive-depth benchmark on Intel replay

    Use identical history, checkpoint, sensor candidates, availability and held-out references. Change depth only. Check a deterministic depth-2 repeat. Planning and worker watchdogs are 90 s and 900 s wall time; neither is charged to the idealized mission clock. Depth 5 is excluded. One fixed dataset does not provide independent stochastic trials.
    """)
    code("""
    show(['Depth','Status','Samples','Final RMSE (C)','MI (nats)','GRG time (s)'],
         [[r['depth'],r['status'],r.get('measurements'),r.get('final_rmse_c'),r.get('mi_nats'),r.get('grg_runtime_s')] for r in depth])
    display(Image(filename=str(result / 'grg_depth_benchmark/intel_lab/figures/depth_comparison.png')))
    display(json.loads((result / 'grg_depth_benchmark/intel_lab/repeat_verification.json').read_text()))
    """)
    md("Conclusion: all three depths return within budget. Depth 3 collects more observations and MI, with lower posterior variance and much higher planning time. Its final RMSE is slightly above the initial 0.48958°C; none of these runs improves on the history-only RMSE. The depth-2 repeat reproduces observations, trajectory, metrics, routes and event times. Depth 5 is excluded.")
    md("""## 4. Correctness and data-provenance checks

    These are current program-verification results. Recursive, navigation and failure tests use bounded controlled fixtures, including a historical synthetic-trained checkpoint. They do not measure Intel temperature-field performance. Intel-specific checks use the current 2,000-record GP and real-record acquisition rules.

    Allow A–B–A and A–A, with separate event times and dwells. Merge shared recursive junctions once. Preserve posterior joint MI, deadlines, return reserves and static reachability. Temporary navigation failure excludes exactly the next top-level cycle. Execute a real alternative action to reach a later cycle; otherwise terminate and return safely. Do not spin empty planning cycles.

    Retain numerical and navigation tests. Replace spatial-uniqueness assertions with explicit-event checks. Replay additionally verifies record-time matching, acquisition waits, no record reuse and executed-only GP updates. Source: [updated tests](testing_code/grg_correctness/test_recursive.py), [failure tests](testing_code/baseline/test_failures.py), [replay tests](testing_code/intel_lab/test_replay.py).
    """, "grg_policy_verification")
    code("""
    correctness = result / 'grg_correcness/current'
    for name in ['new_tests.json', 'numerical_summary.json', 'regressions.json', 'intel_replay_tests.json']:
        display(Markdown('**' + name + '**'))
        display(json.loads((correctness / name).read_text()))
    display(json.loads((correctness / 'obstacle_map_reproduction.json').read_text()))
    """, "grg_policy_verification")
    md("""The earlier route `[0,3,1,2,3,0]` is valid under the new event semantics: four measurements, two distinct node-3 timestamps, four dwells, joint covariance of all events, feasible obstacle-aware return. Recursive prefix tracing checks that a shared junction is not duplicated. These bounded tests do not prove global optimality or correctness for arbitrary depth. Partial-motion recovery retains safe return from the actual pose; arbitrary-pose replanning and ROS/Nav2 remain future work.
    """, "grg_policy_verification")
    nb=nbformat.v4.new_notebook(cells=cells,metadata={"kernelspec":{"name":"python3","display_name":"Python 3","language":"python"}})
    manager=KernelManager(kernel_name="python3")
    manager.kernel_spec.argv=[sys.executable,"-m","ipykernel_launcher","-f","{connection_file}"]
    try:
        NotebookClient(nb,km=manager,timeout=180,resources={"metadata":{"path":str(ROOT)}}).execute()
    finally:
        if manager.has_kernel:manager.shutdown_kernel(now=True)
    nbformat.write(nb,path)
    import copy
    standalone=copy.deepcopy(nb)
    for cell in standalone.cells:
        if cell.cell_type=="markdown":
            cell.source=cell.source.replace("](testing_code/", "](../../../testing_code/")
    nbformat.write(standalone,replay.parent/"intel_lab_results.ipynb")
    save_json(replay.parent/"source_hashes.json",{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((ROOT/"testing_code").rglob("*.py")) if "__pycache__" not in p.parts})
    print("Updated reports and executed Notebook sections.")


if __name__=="__main__":main()

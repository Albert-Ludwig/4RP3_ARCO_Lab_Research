"""Generate exactly one English README and one executed Comparison.ipynb."""
import argparse
import json
from pathlib import Path
import textwrap
import nbformat as nbf
from nbclient import NotebookClient

ROOT=Path(__file__).resolve().parents[1]


def build(comparison):
    comparison=Path(comparison).resolve();relative=comparison.relative_to(ROOT).as_posix()
    evaluation=json.loads((comparison/'evaluation.json').read_text())
    execution=json.loads((comparison/'execution.json').read_text())
    bundle_text=execution['bundle'].replace('\\','/')
    bundle_relative=bundle_text.split('/Code/ROS_Nav2_policy_compare/',1)[-1]
    fixture=evaluation['synthetic_fixture']
    title='Synthetic ROS 2 / Nav2 Validation' if fixture else 'Recorded-Data ROS 2 / Nav2 Comparison'
    note=evaluation['note']
    readme=f'''# ROS 2 / GP / GRG Reconstruction Experiment - Preliminary Feasibility Validation

## Purpose and current evidence

This preliminary experiment assesses whether the integrated **ROS 2 / Gazebo - Nav2 - GP - GRG simulation** provides a feasible, traceable experimental workflow for evaluating the temperature-field reconstruction effects of different GRG scoring policies. It provides initial feasibility evidence for the final research experiment described in [WELL-E 项目核心逻辑](../../Notes/WELL-E项目核心逻辑.md), under the simplified conditions actually executed here.

The executed pilot used **MI, full-domain EMSE, and fixed nonuniform WIVR** as interchangeable GRG scoring functions, with the same RBF spatiotemporal GP, Bunker Mini physics, Nav2 stack, candidate locations, 900-second mission budget, 1-second sensing dwell, and return-home constraint. Reconstruction RMSE was computed on identical hidden measurement records. The historical run did calculate numerical differences; these remain pilot outputs rather than the final policy ranking. Forecasting and fusion policies are outside this experiment.

**Latest executed experiment: {title}.** {note} See [Comparison.ipynb](Comparison.ipynb) for the saved pilot outputs and figures. The completed result supports feasibility in the simplified scene; formal policy conclusions remain subject to the revised protocol and data/map readiness. The [legacy recorded-data readiness gate](result/real_data_preflight/preflight.json) documents inputs for the old acquisition-pool design, not completion requirements for the revised continuous-SRF experiment.

Completed validation: mathematical objective/GRG tests; posterior equivalence to the original trained GPyTorch model; actual Gazebo motion through Nav2; measurement lookup, posterior updates, repeated planning and physical return; MCAP telemetry readback. Actual logs: [{relative}]({relative}/). Failed startup attempts are retained separately as diagnostics, not experiment results. Runtime versions: [environment.json](result/environment.json).

Comparison parameters: [{bundle_relative}/online/parameters.json]({bundle_relative}/online/parameters.json). A separate [300-iteration historical-training preparation check](result/history_training_validation.json) used only synthetic Datasets 0–3; its checkpoint is stored under `result/validation/history_fit_bundle/online/initial_training.pt`.

## Feasibility evidence and limits

The saved run demonstrates the complete workflow under a 10 x 8 m known-static-obstacle scene with three sensing sites:

```text
Initial GP -> selected scoring objective -> GRG route -> Nav2 / Gazebo execution
    -> valid synthetic recorded observation -> GP posterior update -> replanning
    -> return Home -> common held-out reconstruction evaluation
```

- **Executable workflow:** MI, EMSE and fixed-weight WIVR each completed navigation, multiple posterior updates and replanning, and return within the common mission budget.
- **Evaluation feasibility:** a common initial model, declared configuration, identity-disjoint observations and hidden records, and saved GP states support a reproducible reconstruction evaluation. Hidden temperatures and RMSE do not steer online planning.
- **Traceability:** saved trajectories, event logs, GP snapshots, mathematical checks, common-input audits and rosbag readback provide evidence for both execution and evaluation.

**Supported conclusion:** the integrated simulation is a feasible preliminary platform for studying how GRG scoring policies affect reconstruction, under this pilot's declared assumptions. Its saved results support progressing to the final experiment; they do not establish that the revised final protocol has been implemented or validated, or that one policy is generally superior.

The pilot uses an analytic synthetic recorded-data source, node-arrival sensing, fixed WIVR weights and ideal simulator localization. The revised final experiment still needs continuous edge sensing and path-event scoring, GSTools SRF, segment-end batch updates, Adaptive WIVR, and registered real-site data/map and target-scale checks. This feasibility interpretation preserves the original methods, measurements, numerical tables and figures; no new comparison or simulation is performed by this documentation revision.

## Experimental definition

| Item | Definition |
|---|---|
| Initial model | Exact ConstantMean + ScaleKernel(RBF_ARD_xy × RBF_t); fixed parameters during a mission. Real preparation fits Datasets 0–3 only. Synthetic fixtures use declared fixed parameters to isolate integration. |
| Input units | `(x_m, y_m, t_s)` in registered ROS `map` metres and a common seconds origin; original training uses hours, converted explicitly. |
| MI | `0.5 log det(I + C_SS / noise_variance)`, conditional on current measurements; joint-event redundancy retained. This is the existing event-set MI baseline. |
| EMSE | Mean of `diag(C_QS (C_SS + noise_variance I)^-1 C_SQ)` over **all** fixed space-time queries Q. |
| WIVR | Same reduction, averaged with fixed positive nonuniform weights. Fixture: upper-field ROI y ∈ [1,3] m has weight 3; elsewhere 1. This is a declared monitoring priority, not a fitted temperature hotspot. |
| Query domain | Fixture: 13 predefined free-space query positions in the central map region × times 0,60,…,900 s (208 queries). Exactly the same Q for EMSE/WIVR and every policy. Real query positions/time resolution must be fixed before evaluation. |
| GRG | Unchanged recursive search from `Robot_close_loop_system`; scoring adapter is used at every recursion/cache level. Ordered node IDs and estimated sampling times, not a robot trajectory. |
| Nav2 cost | Directed `ComputePathToPose` requests. Estimate = `1.5 × (path_length/0.2 + accumulated_turn/0.45) + 3 s`. These conservative estimates are not measured travel times. Unreachable = infinity; no symmetry assumption. |
| Execution | Real `NavigateToPose` action and Gazebo tracked-body motion. No teleportation. Known static occupancy map and ideal simulator map→odom correction; no LiDAR/SLAM/AMCL validation. |
| Budget | ROS clock runs in a background executor: path requests, GRG computation, navigation, record waiting, dwell and return consume the same 900 s. 15 s return and 5 s planning reserves; feasibility is rechecked after computation. |
| Acquisition | Public recorded pool only. Match actual map pose within 0.25 m and actual mission time within 0.1 s; wait for a valid timestamp if necessary. Preserve original recorded `(x,y,t,T)` and log actual pose/time separately. No temperature interpolation, relabeling, or measurement after failed navigation. |
| Time mapping | Recorded relative time = common `mission_start_s` + (ROS clock − per-run mission clock origin); no time scaling. Simulator startup/initial historical model loading precedes mission start. |
| Hidden split | Alternate eligible chronological records within candidate neighborhoods; other in-domain off-candidate records remain hidden. IDs are disjoint. Only the offline evaluator opens hidden temperatures. |
| Evaluation | History Only / MI / EMSE / WIVR at identical hidden inputs. Curves retrospectively evaluate the fixed target interval; they are not a dedicated forecasting benchmark. Short capped smoke runs can include extrapolation beyond their final observation and cannot establish a full-budget ranking. |
| Fair reset | Fresh Gazebo/Nav2 process, physical pose, clock, GP and unused-record state per policy. Same historical observations, parameters, scene, query/weights and hidden records. RMSE never controls online decisions or stopping. |

Scores are set functions of the scheduled `(x,y,t)` observations. Route order changes their arrival times and therefore changes those score inputs. In the fixture trace, node `0` is Home, nodes `1–3` are sensing sites, and `4` is the virtual current-robot vertex appended at each decision; it is not a fourth sensing site. Repeated site IDs are intentional temporal revisits.

## Files and computational trace

All paths below are relative to this folder; bare module names in the table reside in `testing_code/nav2_policy/`. Test/report scripts reside directly in `testing_code/`. Shared dependencies are relative to repository root. Exact installed simulation-source hashes are saved in [runtime_dependencies.json](result/runtime_dependencies.json).

| File / function | Input → calculation → output | Caller / downstream use |
|---|---|---|
| `config/real.json` | Five canonical dataset paths, map/frame/time setup and fixed experimental rules | `prepare.prepare` → separated online/offline bundle |
| `testing_code/nav2_policy/prepare.py:prepare` | Datasets 0–3 → unchanged `fit_gp` MLL training → fixed physical-unit parameters; Dataset 4 → identity-disjoint pool/hidden split | Offline setup; hashes and compatibility gate prevent accidental reuse of different data |
| `gp.py:GaussianProcess.posterior/covariance/append` | Historical + acquired tuples → exact RBF Cholesky conditioning → means, latent covariance and updated GP | Objective, mission updates, offline reconstruction |
| `objectives.py:Objective.score` | Conditional covariance, candidate space-time event set, Q and weights → MI / EMSE / WIVR | `planner.make_planner` at every recursion; cache key includes policy, GP version and event times |
| `planner.py:make_planner` | Directed Nav2 travel matrix, clock, available record times and GP → reused GRG → ordered node route and score | Mission executes the next node only; then replans |
| `navigation.py:Navigation.estimate/navigate` | Actual TF pose + candidate goal → Nav2 global path estimate / actual navigation action → costs and success/failure | Mission feasibility, execution, failure invalidation and return |
| `acquisition.py:Acquisition.next_time/acquire` | Actual pose/time + public recorded pool → strict tolerances → unused original measurement or no measurement | Mission only updates the GP on valid observations |
| `mission.py:run` | Public bundle + selected policy → navigation → dwell → valid measurement → GP update → GRG replan → Home | Writes events, snapshots, trajectories, costs, plans and mission status |
| `runner.py:run/verify_bag` | Independent missions → MCAP recording and independent physical telemetry checks | Offline evaluator after online process ends |
| `evaluation.py:evaluate` | Saved GP snapshots/final GP + the common hidden labels → RMSE/MAE curves and summary | `Comparison.ipynb`; no feedback to planning |
| `fixtures.py:generate` | Explicit offline analytic synthetic fixture → five recorded CSVs → bundle | Integration validation only; no analytic temperature function in online mission |
| `test_math.py`, `test_training.py` | Direct conditioning identities / original GPyTorch posterior → numerical validation | Saved JSON validation results, not field accuracy claims |
| `build_report.py:build` | Saved experiment logs and GP outputs → this README and one executed notebook | Tables/figures contain measured run outputs only |

Shared sources retained unchanged:

- `Code/GP3D_baseline_test/10.16_testing_code/3D_time&spatial temp field rebuild & prediction/gp_model.py`
- `Code/Robot_close_loop_system/testing_code/baseline/grg.py`
- `Code/ROS_simulation/BunkerMini_drive_test/testing_code/src/well_e_bunker_sim/`
- `Code/ROS_simulation/Nav2_test/testing_code/src/well_e_nav2_test/` and `testing_code/tools/rosbag_capture.py`
- Existing aligned real input: `Data/Real_Data_Integration/result/gp_samples.csv`. Its ENU coordinates still require registration to the navigation map; do not treat GPS trajectory points as an occupancy map.

The old `Code/Policy_compare/` Intel replay and its results are preserved and marked legacy.

## Reproduce

Prerequisites: Docker Desktop, existing `bunker-jazzy` image/container, ROS 2 Jazzy/Nav2/Gazebo packages, existing built Bunker/Nav2 test packages under `/ros-data/build/10.27_ROS_simulation/Nav2_test/install`, and `/home/ubuntu/.venvs/policy-compare`. The container GP venv supplies NumPy/Torch/GPyTorch; ROS apt dependencies are appended after venv packages. Reporting uses `nbformat`, `nbclient`, `ipykernel` and Matplotlib.

Windows launcher (from the repository root): `powershell -ExecutionPolicy Bypass -File Code/ROS_Nav2_policy_compare/testing_code/Run-Comparison.ps1 -Mode full`. Use `validate` for numerical checks, `smoke` for three acquisitions per policy, or `real` after configuring real inputs. It records new results and regenerates the single report; it never overwrites previous run directories.

Alternatively run inside the container from the repository root, using **new** output names:

```bash
R=Code/ROS_Nav2_policy_compare
bash "$R/testing_code/run.sh" "$R/testing_code/test_math.py"
bash "$R/testing_code/run.sh" "$R/testing_code/test_training.py"
bash "$R/testing_code/run.sh" -m nav2_policy.fixtures --output "$R/result/new_fixture" --full-budget
bash "$R/testing_code/run.sh" -m nav2_policy.runner --bundle "$R/result/new_fixture/bundle" --output "$R/result/new_comparison" --policies mi emse wivr
```

**Legacy recorded-data reproduction only:** the instructions below describe the existing acquisition-pool implementation. The final continuous-SRF experiment follows the revised logic document and does not use Dataset 4 as a simulated acquisition pool.

For the legacy mode, first provide **five** CSVs with globally unique `record_id,run_id,x_m,y_m,t_s,temperature_c`; all coordinates must already share the ROS `map` frame, and all times must share the specified UTC epoch. Datasets 0–3 must precede Dataset 4's mission start. Use repository-relative POSIX paths in `config/real.json` for Docker, including candidate sites, ROI, target grid, map YAML/image and scene bounds. Then run `-m nav2_policy.prepare --config ... --output ...` and use the resulting bundle. The preflight checks recorded availability; actual Nav2 travel costs provide a second compatibility gate. Missing records are not filled with model-generated temperatures. The exact CPU implementation is currently validated at the fixture's 312 historical observations; planning latency must be rechecked at the full real-data size.

Generate the report using a Python environment with the reporting packages:

```bash
python Code/ROS_Nav2_policy_compare/testing_code/build_report.py --comparison Code/ROS_Nav2_policy_compare/{relative}
```

Logs include `events.jsonl`, `plans.json`, `measurements.csv`, `trajectory.csv`, `navigation.csv`, `travel_estimates.csv`, GP snapshot `.npz` files and MCAP readback. Large `.mcap` recordings remain local and are ignored by Git; metadata, checks, numerical tables and this notebook remain reviewable. No commit or push is performed by this workflow.
'''
    (ROOT/'README.md').write_text(readme,encoding='utf-8')
    cells=[]
    def md(text):cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(text).strip()))
    def code(text):cells.append(nbf.v4.new_code_cell(textwrap.dedent(text).strip()))
    md('''# Preliminary Feasibility Validation: ROS 2 / GP / GRG Reconstruction Experiment

**Purpose:** assess whether an integrated ROS 2 / Gazebo - Nav2 - GP - GRG simulation can support the final study of how MI, EMSE and WIVR scoring affect temperature-field reconstruction. This is preliminary feasibility evidence under simplified, explicitly recorded conditions.

The saved synthetic pilot used a 10 x 8 m scene, three sensing sites, node-arrival observations, a fixed-parameter RBF GP, and fixed WIVR spatial weights. Each policy executed navigation, measurement acquisition, posterior updates, replanning and return, followed by reconstruction evaluation at the same hidden inputs.

**Evidence status:** the original numerical differences and figures are preserved as pilot outputs. They do not establish the final ranking or validate continuous edge sensing, GSTools SRF, Adaptive WIVR, or real-site-scale operation. The revised experiment is specified in [WELL-E 项目核心逻辑](../../Notes/WELL-E项目核心逻辑.md). See README for the executed methods and dependencies.

This documentation revision does not rerun the simulation, retrain the GP, or change saved observations or outputs.''')
    code(f'''
    from pathlib import Path
    import json, csv
    import numpy as np
    import matplotlib.pyplot as plt
    from IPython.display import display, Markdown
    ROOT = Path.cwd()
    if not (ROOT / 'result').is_dir():
        raise RuntimeError('Run this notebook from Code/ROS_Nav2_policy_compare')
    RUN = ROOT / {relative!r}
    def read_json(path): return json.loads(path.read_text())
    def rows(path):
        with path.open(newline='', encoding='utf-8-sig') as f: return list(csv.DictReader(f))
    evaluation = read_json(RUN / 'evaluation.json')
    execution = read_json(RUN / 'execution.json')
    bundle_text = execution['bundle'].replace('\\\\','/')
    marker = '/Code/ROS_Nav2_policy_compare/'
    BUNDLE = ROOT / bundle_text.split(marker,1)[1] if marker in bundle_text else Path(bundle_text)
    summary = rows(RUN / 'summary.csv')
    curves = rows(RUN / 'reconstruction_curves.csv')
    policies = ['mi','emse','wivr']
    colors = {{'History Only':'gray','MI':'#3574b9','EMSE':'#d47b22','WIVR':'#2b9975'}}
    print('Mode:', evaluation['mode'], '| Identical hidden records:', evaluation['hidden_count'])
    print('Final real-data comparison completed:', evaluation['full_real_comparison_complete'])
    print('Initial historical observations:', len(np.load(RUN/'mi/gp_000.npz')['x']))
    print('Physical-unit RBF parameters:', read_json(BUNDLE/'online/parameters.json'))
    print('Objective tests:', read_json(ROOT/'result/math_tests.json'))
    print('Original-model equivalence:', read_json(ROOT/'result/training_equivalence.json'))
    if (ROOT/'result/history_training_validation.json').is_file():
        print('Separate historical-fit preparation check:',read_json(ROOT/'result/history_training_validation.json'))
    ''')
    md('''## Pilot numerical outputs
RMSE and MAE are in °C. These saved numerical outputs document the preliminary experiment, not a final real-field ranking. Distance and mission duration come from physical telemetry and ROS time. Measurement counts include temporal revisits and are not counts of distinct spatial sites. History Only performs no navigation.''')
    code('''
    columns = [('method','Method'),('rmse_c','RMSE (°C)'),('mae_c','MAE (°C)'),('measurements','Measurements'),
               ('distance_m','Distance (m)'),('duration_s','Duration (s)'),('return_home_success','Home'),('status','Status')]
    table = '| ' + ' | '.join(label for key,label in columns) + ' |\\n'
    table += '| ' + ' | '.join('---' for _ in columns) + ' |\\n'
    for row in summary:
        values=[]
        for key,label in columns:
            value=row.get(key,'')
            if key in ('rmse_c','mae_c','distance_m','duration_s'): value=f'{float(value):.4f}'
            values.append(str(value) or 'N/A')
        table += '| ' + ' | '.join(values) + ' |\\n'
    display(Markdown(table))
    ''')
    md('## Reconstruction and posterior uncertainty\nCurves re-query the same full hidden space-time set after each posterior update. They are retrospective full-interval evaluations, not instantaneous sensor errors or a separately designed forecasting test.')
    code('''
    fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    axes[0,0].bar([r['method'] for r in summary],[float(r['rmse_c']) for r in summary],color=[colors[r['method']] for r in summary])
    axes[0,0].set(title='Final hidden-set reconstruction',ylabel='RMSE (°C)')
    for method in ['History Only','MI','EMSE','WIVR']:
        selected=[r for r in curves if r['method']==method]
        if not selected:continue
        t=[float(r['time_s']) for r in selected];n=[int(r['measurements']) for r in selected];rmse=[float(r['rmse_c']) for r in selected]
        axes[0,1].plot(t,rmse,'o-',label=method,color=colors[method])
        axes[1,0].plot(n,rmse,'o-',label=method,color=colors[method])
        variance=[r for r in selected if r.get('mean_query_variance_c2')]
        if variance:axes[1,1].plot([float(r['time_s']) for r in variance],[float(r['mean_query_variance_c2']) for r in variance],'o-',label=method,color=colors[method])
    axes[0,1].set(title='Hidden RMSE vs mission time',xlabel='ROS mission time (s)',ylabel='RMSE (°C)')
    axes[1,0].set(title='Hidden RMSE vs successful measurements',xlabel='Successful measurements',ylabel='RMSE (°C)')
    axes[1,1].set(title='Mean latent variance over fixed Q',xlabel='ROS mission time (s)',ylabel='Variance (°C²)')
    for ax in axes.flat:ax.grid(alpha=.2)
    axes[0,1].legend();axes[1,1].legend();plt.show()
    ''')
    md('## Actual Nav2 trajectories, travel estimates and computation\nPaths below are physical Gazebo odometry, not GRG straight-line routes. Travel-time estimates include conservative speed/turn/reserve assumptions and should not be interpreted as calibrated predictions.')
    code('''
    from matplotlib.patches import Rectangle
    fig,axes=plt.subplots(1,3,figsize=(15,4),constrained_layout=True)
    # Geometry comes from the saved physical world configuration, not a drawn result.
    meta=read_json(RUN/'run_configuration.json')
    scene=meta['scene'];xmin,xmax,ymin,ymax=scene['map_bounds_m']
    if meta['mode']=='synthetic_fixture':
        obstacle=scene['obstacle'];x,y=obstacle['center_m'];w,h=obstacle['size_m']
        axes[0].add_patch(Rectangle((x-w/2,y-h/2),w,h,color='black',alpha=.3))
    else:
        import matplotlib.image as mpimg
        import yaml
        mapmeta=yaml.safe_load((RUN/'mi/generated/map.yaml').read_text())
        image=mpimg.imread(RUN/'mi/generated'/mapmeta['image']);ox,oy,_=mapmeta['origin'];res=mapmeta['resolution']
        axes[0].imshow(image,cmap='gray',extent=[ox,ox+image.shape[1]*res,oy,oy+image.shape[0]*res])
    all_est=[];all_actual=[]
    for policy in policies:
        trajectory=rows(RUN/policy/'trajectory.csv');navigation=rows(RUN/policy/'navigation.csv');plans=read_json(RUN/policy/'plans.json')
        axes[0].plot([float(r['x_m']) for r in trajectory],[float(r['y_m']) for r in trajectory],label=policy.upper(),color=colors[policy.upper()])
        valid=[r for r in navigation if r.get('estimated_s') and r.get('actual_travel_s')]
        est=[float(r['estimated_s']) for r in valid];actual=[float(r['actual_travel_s']) for r in valid]
        axes[1].scatter(est,actual,label=policy.upper(),color=colors[policy.upper()]);all_est+=est;all_actual+=actual
        axes[2].plot([p['time_s'] for p in plans],[p['ros_compute_s'] for p in plans],'o-',label=policy.upper(),color=colors[policy.upper()])
    points=np.array(meta['candidate_xy_m']);axes[0].scatter(points[:,0],points[:,1],marker='x',c='black');axes[0].scatter(*points[0],marker='*',s=100,c='red',label='Home')
    axes[0].set(xlim=(xmin,xmax),ylim=(ymin,ymax),aspect='equal',title='Actual physical trajectories',xlabel='map x (m)',ylabel='map y (m)')
    bound=max(all_est+all_actual+[1]);axes[1].plot([0,bound],[0,bound],'k--',alpha=.4)
    axes[1].set(title='Estimated vs actual travel',xlabel='Estimated (s)',ylabel='Actual ROS travel (s)')
    axes[2].set(title='GRG online computation',xlabel='ROS mission time (s)',ylabel='Charged ROS computation (s)')
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8)
    plt.show()
    ''')
    md('## Trace, acceptance and conclusions\nAn example traces the first MI route, actual Nav2 arrival, original recorded measurement and posterior update. All assertions use saved outputs.')
    code('''
    for policy in policies:
        mission=read_json(RUN/policy/'mission.json');bag=read_json(RUN/policy/'rosbag_summary.json')
        acquired=rows(RUN/policy/'measurements.csv');sites=sorted({int(r['node']) for r in acquired})
        print(policy.upper(),':',mission['successful_measurements'],'updates at',len(sites),'distinct sensing sites,',round(mission['duration_s'],3),'s, Home=',mission['return_home_success'],
              '| rosbag:',bag['status'],'| clearance:',bag.get('minimum_footprint_clearance_m'))
        assert mission['status']=='passed' and mission['return_home_success'] and mission['within_budget']
        assert mission['successful_measurements']>=2 and mission['planning_decisions']>=2
        assert bag['status']=='passed'
    trace=[json.loads(line) for line in (RUN/'mi/events.jsonl').read_text().splitlines()]
    for event in ['grg_plan','navigation_result','gp_update']:
        print(event,':',next(r for r in trace if r['event']==event))
    print('Replan after first posterior update:',next(r for r in trace if r['event']=='grg_plan' and r['gp_version']==1))
    # Recompute the logged first-route MI from saved initial observations and
    # scheduled event inputs, without future temperature values.
    import sys
    sys.path.insert(0,str(ROOT/'testing_code'))
    from nav2_policy.gp import GaussianProcess
    initial=np.load(RUN/'mi/gp_000.npz');hp=read_json(BUNDLE/'online/parameters.json')
    initial_gp=GaussianProcess(initial['x'],initial['y'],hp)
    first=read_json(RUN/'mi/plans.json')[0];points=read_json(RUN/'run_configuration.json')['candidate_xy_m']
    events=np.array([[*points[node],t] for node,t in zip(first['route'][1:-1],first['arrivals_s'])])
    covariance=initial_gp.covariance(events)
    reproduced_mi=.5*np.linalg.slogdet(np.eye(len(events))+covariance/initial_gp.noise)[1]
    print('First planned sensing events (m,m,s):',events)
    print('Conditional event covariance (°C²):',covariance)
    print('Recomputed MI / logged MI (nats):',reproduced_mi,first['score'])
    assert np.isclose(reproduced_mi,first['score'],atol=1e-10,rtol=1e-10)
    print('Hidden records and common inputs audit:',read_json(RUN/'fairness.json'))
    baseline=float(next(r['rmse_c'] for r in summary if r['method']=='History Only'))
    print('Observed fixture RMSE differences from History Only (positive = lower RMSE):')
    for row in summary[1:]:print(row['method'],round(baseline-float(row['rmse_c']),6),'°C')
    display(Markdown('**Conclusion:** '+evaluation['note']+' Numerical differences describe this recorded fixture only. '
        'They do not establish the best policy for the real field. The formal comparison requires five registered real datasets and the matching 2D navigation map.'))
    ''')
    md('''## Interpretation for the final experiment

The saved results provide initial evidence that the complete ROS - GP - GRG simulation can execute the intended experiment workflow and support a traceable, common-input reconstruction evaluation for three scoring policies. The pilot genuinely evaluated reconstruction differences; those values remain limited to the executed fixture.

Use this result as a preliminary feasibility study for the final experiment. Subsequent readiness checks should verify the added SRF source, continuous sampling/path events, batch updates and Adaptive WIVR, then demonstrate each policy separately. Formal performance conclusions require the revised protocol, registered field map/data and target-scale validation.

The preceding saved console conclusion describes the recorded-data pilot and its legacy five-dataset preparation mode. The current final design uses SRF observations and reserves Dataset 4 for independent interpolation-valid real-data validation when suitable support exists. Historical outputs are retained unchanged.''')
    nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}})
    path=ROOT/'Comparison.ipynb';nbf.write(nb,path)
    NotebookClient(nb,timeout=240,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}}).execute()
    nbf.write(nb,path)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--comparison',required=True);a=p.parse_args();build(a.comparison)

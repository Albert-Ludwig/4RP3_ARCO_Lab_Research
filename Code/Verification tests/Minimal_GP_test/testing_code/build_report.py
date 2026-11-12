"""Verify saved results, then create one English README and executed Notebook."""
import csv
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nbformat
from nbclient import NotebookClient

ROOT=Path(__file__).resolve().parents[1]
LABELS={'blocked_reconstruction':'Held-out route segment','forward_prediction':'Forward prediction',
        'train_mean':'Training mean','last_observation':'Last observation',
        'temporal_gp':'Temporal GP','spatiotemporal_gp':'Spatiotemporal GP'}
COLORS={'train_mean':'#777777','last_observation':'#b57b32','temporal_gp':'#315eab','spatiotemporal_gp':'#138677'}


def read_csv(path):
    with path.open(newline='',encoding='utf-8') as f: return list(csv.DictReader(f))


def read_json(path): return json.loads(path.read_text(encoding='utf-8'))
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(run):
    manifest=read_json(run/'manifest.json')
    for p,h in manifest['outputs'].items(): assert sha(run/p)==h,p
    assert sha(run/'input_samples.csv')==manifest['input_sha256']
    source=read_csv(run/'input_samples.csv'); byid={r['sample_id']:r for r in source}
    splits=read_csv(run/'splits.csv'); predictions=read_csv(run/'predictions.csv'); metrics=read_csv(run/'metrics.csv')
    hp=read_json(run/'hyperparameters.json'); candidates=read_csv(run/'initialization_comparison.csv')
    for task in hp:
        train=[r for r in splits if r['task']==task and r['role']=='train']
        test={r['sample_id'] for r in splits if r['task']==task and r['role']=='test'}
        assert not test & {r['sample_id'] for r in train}
        for model,h in hp[task].items():
            ys=[float(byid[r['sample_id']]['temperature_c']) for r in train]
            assert abs(np.mean(ys)-h['scaler']['y_mean'])<1e-12
            cs=[r for r in candidates if r['task']==task and r['model']==model]
            selected=next(r for r in cs if r['selected']=='True')
            assert float(selected['final_negative_training_mll'])==min(float(r['final_negative_training_mll']) for r in cs)
        for met in [m for m in metrics if m['task']==task]:
            rr=[r for r in predictions if r['task']==task and r['model']==met['model']]
            assert len(rr)==len(test) and {r['sample_id'] for r in rr}==test
            for r in rr:
                assert float(r['observed_temperature_c'])==float(byid[r['sample_id']]['temperature_c'])
            errors=np.array([float(r['predicted_temperature_c'])-float(r['observed_temperature_c']) for r in rr])
            assert abs(np.sqrt(np.mean(errors**2))-float(met['rmse_c']))<1e-12
            assert abs(np.mean(np.abs(errors))-float(met['mae_c']))<1e-12
            if met['coverage_95']:
                var=np.array([float(r['observed_variance_c2']) for r in rr])
                assert abs(np.mean(np.abs(errors)<=1.959963984540054*np.sqrt(var))-float(met['coverage_95']))<1e-12
                assert abs(np.mean(.5*(np.log(2*np.pi*var)+errors**2/var))-float(met['marginal_nlpd']))<1e-12
    v=read_json(run/'verification.json'); assert v['passed']
    result=dict(passed=True,run=run.name,hashed_outputs=len(manifest['outputs']),
                verified_metric_rows=len(metrics),worker_checks=len(v['checks']),
                checks=['Saved artifacts match hashes','Input snapshot matches prepared data digest',
                        'Train/test IDs are disjoint','GP scaling uses training labels only',
                        'Initialization selection uses training likelihood only',
                        'Every prediction has the correct held-out ID and observed label',
                        'RMSE, MAE, predictive coverage and marginal NLPD recomputed'])
    (ROOT/'result'/'report_verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    return result


def main():
    run=ROOT/'result'/read_json(ROOT/'result'/'latest.json')['run']
    verification=verify(run)
    source=read_csv(run/'input_samples.csv'); splits=read_csv(run/'splits.csv'); info=read_json(run/'splits.json')
    predictions=read_csv(run/'predictions.csv'); metrics=read_csv(run/'metrics.csv'); hp=read_json(run/'hyperparameters.json')
    hist=read_csv(run/'training_history.csv')
    figdir=ROOT/'result'/'figures'; figdir.mkdir(exist_ok=True)
    byid={r['sample_id']:r for r in source}
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    for ax,task in zip(axes,info):
        for role,color in [('train','#777777'),('buffer','#d5a63c'),('test','#ba4040')]:
            rows=[byid[r['sample_id']] for r in splits if r['task']==task and r['role']==role]
            if rows: ax.scatter([float(r['x_east_m']) for r in rows],[float(r['y_north_m']) for r in rows],s=7,c=color,label=role)
        ax.set(xlabel='East (m)',ylabel='North (m)',title=LABELS[task]); ax.set_aspect('equal'); ax.legend(fontsize=8)
    fig.suptitle('Fixed data partitions on the measured route'); fig.savefig(figdir/'splits.png',dpi=155); plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(13,7.6),layout='constrained')
    for row,task in enumerate(info):
        for col,model in enumerate(['temporal_gp','spatiotemporal_gp']):
            ax=axes[row,col]
            rr=[r for r in predictions if r['task']==task and r['model']==model]
            t=np.array([float(r['t_s'])/60 for r in rr]); mu=np.array([float(r['predicted_temperature_c']) for r in rr])
            ax.fill_between(t,[float(r['lower_95_c']) for r in rr],[float(r['upper_95_c']) for r in rr],color=COLORS[model],alpha=.18,label='95% observation interval')
            ax.plot(t,[float(r['observed_temperature_c']) for r in rr],color='black',lw=1,label='Held-out observation')
            ax.plot(t,mu,color=COLORS[model],lw=1.4,label=LABELS[model])
            for baseline in ['train_mean']+(['last_observation'] if task=='forward_prediction' else []):
                base=next(r for r in predictions if r['task']==task and r['model']==baseline)
                ax.axhline(float(base['predicted_temperature_c']),ls='--',color=COLORS[baseline],lw=.9,label=LABELS[baseline])
            met=next(m for m in metrics if m['task']==task and m['model']==model)
            ax.set(title=f"{LABELS[task]} — {LABELS[model]}\nRMSE {float(met['rmse_c']):.3f} °C; coverage {float(met['coverage_95']):.1%}",
                   xlabel='Minutes since first GPS fix',ylabel='Temperature (°C)')
            ax.legend(fontsize=7,loc='best')
    fig.savefig(figdir/'predictions.png',dpi=155); plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,6),layout='constrained')
    for row,task in enumerate(info):
        for col,model in enumerate(['temporal_gp','spatiotemporal_gp']):
            ax=axes[row,col]
            for init in [.3,1.,3.]:
                rr=[r for r in hist if r['task']==task and r['model']==model and float(r['initial_lengthscale_scaled'])==init]
                ax.plot([int(r['iteration']) for r in rr],[float(r['negative_mll_scaled']) for r in rr],label=f'initial lengthscale {init}')
            ax.set(title=LABELS[task]+' — '+LABELS[model],xlabel='Adam iteration',ylabel='Negative training MLL'); ax.legend(fontsize=7)
    fig.savefig(figdir/'training.png',dpi=155); plt.close(fig)
    conclusions=[]
    for task in info:
        ms={m['model']:float(m['rmse_c']) for m in metrics if m['task']==task}
        st=ms['spatiotemporal_gp']; tm=ms['temporal_gp']; best=min(ms,key=ms.get)
        conclusions.append(f"**{LABELS[task]}:** {LABELS[best]} has the lowest RMSE ({ms[best]:.4f} °C). "
            f"Spatiotemporal GP RMSE is {st:.4f} °C versus {tm:.4f} °C for temporal GP and {ms['train_mean']:.4f} °C for the training mean.")
    observations='\n'.join('- '+x for x in conclusions)
    forward_hp=hp['forward_prediction']['spatiotemporal_gp']
    st_rows={r['task']:r for r in metrics if r['model']=='spatiotemporal_gp'}
    diagnosis=(f"The plotted GP means largely return to their learned constant levels away from training support and miss the local peaks. "
        f"The selected forward spatiotemporal GP has a temporal lengthscale of {forward_hp['lengthscales_m_m_s'][2]:.1f} s, "
        f"while the maximum prediction horizon is {info['forward_prediction']['forecast_horizon_seconds']:.0f} s and the median nearest-training "
        f"spatial distance is {info['forward_prediction']['nearest_train_distance_m']['median']:.1f} m. "
        f"This is consistent with weak correlation to the training data over much of the forecast. "
        f"Spatiotemporal GP interval coverage is 100% in these two splits, with mean widths "
        f"{float(st_rows['blocked_reconstruction']['mean_interval_width_c']):.3f} °C and "
        f"{float(st_rows['forward_prediction']['mean_interval_width_c']):.3f} °C; broad intervals do not establish precise predictions or calibrated 95% coverage. "
        f"The data-to-GP pipeline is working, but this experiment does not support a forecasting advantage over persistence.")
    table='\n'.join('| '+LABELS[m['task']]+' | '+LABELS[m['model']]+' | '+f"{float(m['rmse_c']):.4f} | {float(m['mae_c']):.4f} | "+(f"{float(m['coverage_95']):.1%}" if m['coverage_95'] else 'n/a')+' |' for m in metrics)
    # From this experiment, Code/GP3D_baseline_test is two levels up.
    model_link='../../GP3D_baseline_test/10.16_testing_code/3D_time%26spatial%20temp%20field%20rebuild%20%26%20prediction/gp_model.py'
    readme=f'''# Minimal GP Test

## Purpose and scope

Check whether the aligned **2026-11-03 WELL-E field data** support useful GP estimates of a missing route segment and later measurements. Compare spatiotemporal GP, temporal-only GP and simple baselines. This experiment contains no policy ranking, GRG planning or ROS navigation.

Input: **{len(source)} observations** with local ENU position, timestamp and unmodified HOBO temperature. Accurate HOBO/GPS synchronization with zero extra offset is an **accepted experimental assumption**, as requested. The source's `clock_alignment_verified=False` records the absence of independent verification; it is not a training exclusion or a blocker here.

## Fixed protocol

| Item | Setting |
|---|---|
| Route-segment holdout | Middle 20% of chronological records; {info['blocked_reconstruction']['test_count']} test, {info['blocked_reconstruction']['train_count']} train, {info['blocked_reconstruction']['buffer_count']} buffer records |
| Boundary buffer | Exclude 10 seconds on each side of the held-out interval; no random row split |
| Forward prediction | First 70% ({info['forward_prediction']['train_count']}) train; final 30% ({info['forward_prediction']['test_count']}) test; {info['forward_prediction']['forecast_horizon_seconds']:.0f} s maximum horizon; no test-time updates |
| Spatiotemporal GP | Existing ConstantMean + ScaleKernel(RBF_ARD_xy × RBF_t), Gaussian likelihood |
| Temporal GP | Same training procedure, using time only; measurements still come from a moving sensor |
| Baselines | Training-set mean in both tasks; last training temperature held constant for forward prediction |
| Fitting | CPU float64, four threads, Adam 300 steps, learning rate 0.05; standardized lengthscale starts 0.3, 1, 3 |
| Selection | Lowest final negative **training** marginal likelihood across starts; no test-label checkpoint/initialization selection |
| Scaling / units | Each task/model fits scalers on its training set only; spatial units metres, GP time hours (`t_s/3600`), temperature °C |
| Uncertainty | 95% marginal intervals for observations, including learned likelihood noise |

The route-segment task uses observations before **and after** the missing interval and is retrospective completion. It does not measure reconstruction of an entire simultaneous field. The forward task predicts temperature at recorded future query positions, so temporal extrapolation and spatial generalization occur together. Query positions/times are known for evaluation, but all held-out temperature labels are excluded from fitting. There is no multi-day test or independent replication. Three initializations are optimization checks, not three independent experiments.

## Results

| Task | Model | RMSE (°C) | MAE (°C) | 95% observation coverage |
|---|---|---:|---:|---:|
{table}

{observations}

{diagnosis}

The table reports error against held-out **measured temperature**. Noisy observation intervals are used for coverage; point-only baselines have no assigned intervals. Marginal NLPD and interval width are saved in `metrics.csv`. Numerical checks passing does not imply good accuracy or calibrated uncertainty. Interpret coverage against the nominal 95%, together with interval width and NLPD. One moving pass cannot establish unique spatial/temporal kernel identification or transfer to another day.

## Files and dependencies

| File | Purpose |
|---|---|
| [test_report.ipynb](test_report.ipynb) | One executed report with metrics, split/forecast/training figures, fitted parameters and conclusions |
| [testing_code/protocol.json](testing_code/protocol.json) | Protocol fixed before model fitting |
| [testing_code/run_test.py](testing_code/run_test.py) | Copies only the small input/model/scripts into the existing Docker environment, runs fits and retrieves results |
| [testing_code/worker.py](testing_code/worker.py) | Reuses GP fitting, creates splits, evaluates models and checks numerical correctness |
| [testing_code/build_report.py](testing_code/build_report.py) | Verifies saved data and rebuilds the single README/Notebook |
| [Prepared data](../Real_Data_Integration/result/gp_samples.csv) | Input from the completed temperature/GPS alignment |
| [Existing GP model]({model_link}) | Reused unchanged model, scaler, fit and persistence functions; source SHA-256 recorded |
| [result/{run.name}](result/{run.name}) | Input snapshot, frozen protocol, splits, all predictions, metrics, training histories, selected checkpoints and provenance |

The GP implementation is reused from `Code/GP3D_baseline_test/10.16_testing_code/3D_time&spatial temp field rebuild & prediction/gp_model.py`. No Intel-trained weights are loaded. Docker container `bunker-jazzy` uses `/home/ubuntu/.venvs/policy-compare/bin/python`; exact numerical library versions are in the run manifest. Host processing uses standard Python; reports use the existing `C:\\Ubuntu\\ROS\\reporting-venv` with NumPy, Matplotlib, nbformat, nbclient and a Python Jupyter kernel.

## Reproduce and verification

From this folder, with the existing Docker container running:

```powershell
powershell -ExecutionPolicy Bypass -File .\\testing_code\\Run-MinimalGP.ps1
```

Each invocation creates a new run directory and updates `result/latest.json`. `build_report.py` can rebuild the report without refitting. The current run is `{run.name}`.

**{verification['worker_checks']} worker checks passed**, including split boundaries, training-only scaling, positive variances, independently computed RBF/Cholesky predictions and checkpoint reload. Report verification independently recomputed all {verification['verified_metric_rows']} metric rows, checked source labels/IDs, selection criteria and saved-file hashes. Records are in the run's `verification.json` and `result/report_verification.json`. Data and reused GP source were unchanged; no commit or upload was performed.

These are initial field-data diagnostics. Further changes to kernels or protocols would need a fresh, separately reported experiment rather than replacing these results.
'''
    readme += future_comparison_plan()
    (ROOT/'README.md').write_text(readme,encoding='utf-8')
    nb=nbformat.v4.new_notebook()
    nb.metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}}
    nb.cells=[nbformat.v4.new_markdown_cell('''# Minimal GP Test: Real Field Data

Two fixed tasks: missing route-segment estimation and forward prediction. Accurate clock synchronization is accepted as an experimental assumption. All parameters and preprocessing use training observations only. Full methods and dependencies are in the [README](README.md).

The route holdout uses both past and later observations; the forward test uses only earlier observations. There are no online updates, policy comparisons or navigation runs.'''),
    nbformat.v4.new_code_cell('''from pathlib import Path
import csv, json, hashlib, html
from IPython.display import display, HTML, Image
root=Path.cwd()
run=root/'result'/json.loads((root/'result'/'latest.json').read_text())['run']
def read_csv(p):
    with p.open(newline='',encoding='utf-8') as f: return list(csv.DictReader(f))
def table(headers,rows):
    def line(values,tag): return '<tr>'+''.join(f'<{tag}>{html.escape(str(v))}</{tag}>' for v in values)+'</tr>'
    display(HTML('<table>'+line(headers,'th')+''.join(line(r,'td') for r in rows)+'</table>'))
metrics=read_csv(run/'metrics.csv')
table(['Task','Model','RMSE °C','MAE °C','95% coverage','Interval width °C','Marginal NLPD'],
      [[r['task'],r['model']]+[f'{float(r[k]):.4f}' if r[k] else 'n/a' for k in
        ['rmse_c','mae_c','coverage_95','mean_interval_width_c','marginal_nlpd']] for r in metrics])
print('Run:',run.name)
print('Coverage is for noisy observations. Baselines have no assigned uncertainty intervals.')'''),
    nbformat.v4.new_markdown_cell('''## Splits and held-out predictions

Partitions were fixed before fitting. Position and time are correlated along this moving route, so the forward task is not a pure fixed-location temporal forecast. Shaded bands include learned observation noise.'''),
    nbformat.v4.new_code_cell("for name in ['splits.png','predictions.png','training.png']:\n    display(Image(filename=str(root/'result'/'figures'/name),embed=True))"),
    nbformat.v4.new_markdown_cell('''## Parameters and checks

These fitted parameters describe this run under the chosen model. Different initializations may give different training optima; selection uses training marginal likelihood only. They are not validated universal field parameters. The three starts are not independent data replicates.'''),
    nbformat.v4.new_code_cell('''hp=json.loads((run/'hyperparameters.json').read_text())
rows=[]
for task,models in hp.items():
    for model,h in models.items():
        length=h.get('lengthscales_m_m_s',[h.get('lengthscale_seconds')])
        rows.append([task,model,', '.join(f'{v:.3f}' for v in length),f"{h['noise_variance_c2']**.5:.4f}",h['selected_initial_lengthscale_scaled']])
table(['Task','Model','Lengthscales (x m, y m, t s; or t s)','Noise SD °C','Selected initial scale'],rows)
m=json.loads((run/'manifest.json').read_text())
for name,expected in m['outputs'].items():
    assert hashlib.sha256((run/name).read_bytes()).hexdigest()==expected,name
v=json.loads((run/'verification.json').read_text()); assert v['passed']
rv=json.loads((root/'result'/'report_verification.json').read_text()); assert rv['passed']
print(len(v['checks']),'worker checks passed;',rv['verified_metric_rows'],'metric rows independently verified.')
print('Source clock assumption accepted; no independent synchronization certification is claimed.')'''),
    nbformat.v4.new_markdown_cell('## Findings\n\n'+observations+'\n\n'+diagnosis+'''

Performance and predictive coverage must be judged separately from numerical correctness. The comparison covers one mobile run and two fixed partitions. It does not validate dense field reconstruction, multi-day prediction, unique spatial/temporal parameter identification or policy superiority. Future observations were used for scoring only in the forward test.

These results provide a diagnostic starting point for subsequent GP/ROS work. They do not establish a deployment-ready kernel.''')]
    NotebookClient(nb,timeout=120,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}}).execute()
    nbformat.validate(nb); nbformat.write(nb,ROOT/'test_report.ipynb')
    print(table)
    print('\n'.join(conclusions))
    print('One README and one executed Notebook saved.')


def future_comparison_plan():
    return '''
## Planned next experiment: real-data reconstruction policy comparison

**Status: planned, not implemented or run.** Start after the next multiple real spatiotemporal collections and the 2D field map are available. Four additional collections are expected. Preserve the present single-run results as the initial baseline.

This semester's objective is temperature-field **reconstruction with the existing RBF spatiotemporal kernel**. Time represents changes during traversal. Comparisons of different kernel families for prediction are deferred to next semester.

The next experiment will cover **all comparison methods implemented in the original GP-Planning repository**, together with the user's **MI:EMSE = 5:5 and 4:6 fusion policies**. Freeze an explicit source-backed method inventory before execution, distinguishing GRG scoring objectives from other sampling/planning baselines:

| Group | Planned coverage |
|---|---|
| GRG scoring objectives | MI, EMSE, WIVR (weighted integrated variance reduction) |
| User-defined fusion policies | MI:EMSE = 5:5 and 4:6, with explicit fixed normalization and weights |
| Original sampling/planning comparisons | Random, fine/coarse lawnmower, nearest neighbor, greedy MI, sequential entropy/variance sampling, myopic, open-loop and receding-horizon GRG, and continuous EMSE/edge-sampling variants where implemented |

Original method definitions and dependencies must be checked in `GP-Planning/src/solvers.py`, `src/planners.py`, and notebooks 02-07. Some methods are equivalent under particular assumptions (for example, sequential entropy and maximum-variance selection); record these equivalences rather than counting them as independent evidence. Comparisons that change the planner or sampling mode must be reported separately from the fixed-GRG objective comparison. No previously mentioned MI, EMSE or fusion result establishes a winner on the new field data.

Use one shared RBF model and identical initial observations, candidate locations, navigation costs, time budget, sampling/noise rules and reconstruction reference times for the fixed-GRG comparison. Define WIVR's region weights from the task/map before inspecting evaluation temperatures; do not copy the original pool-edge weighting into this field without a matching reason. Fusion normalizers are set from the common initial model/candidate catalog and remain fixed across policies and replans.

Fit kernel parameters using the designated training runs, and reserve at least one complete collection from kernel fitting. Separate acquired observations from a shared held-out evaluation set, retain actual positions and timestamps, and evaluate reconstruction using RMSE, MAE, uncertainty coverage/width, sampling count, travel cost and runtime. Apply consistent acquisition rules and uncertainty definitions across methods.

Recorded measurements exist only at their observed positions/times. Freeze a feasible replay/measurement-source protocol before comparing arbitrary routes. Interpolated temperatures or GP predictions must not be labelled measured ground truth. Dense map accuracy can be evaluated only where independent reference data support it. Run the new comparison as a separate experiment and retain its full provenance. This plan records future work only; it does not start training, policy evaluation or ROS integration now.
'''


if __name__=='__main__': main()

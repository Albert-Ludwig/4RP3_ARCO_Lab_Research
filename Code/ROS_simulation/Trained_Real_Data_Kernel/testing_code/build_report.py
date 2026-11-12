"""Verify full-data kernel artifacts and build one README and Notebook."""
import csv
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nbformat
from nbclient import NotebookClient

ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parents[2]


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def csv_rows(p):
    with p.open(newline='',encoding='utf-8') as f: return list(csv.DictReader(f))


def main():
    run=ROOT/'result'/read(ROOT/'result'/'latest.json')['run']
    manifest=read(run/'manifest.json'); hp=read(run/'hyperparameters.json')
    summary=read(run/'training_summary.json'); coords=read(run/'coordinate_system.json')
    verification=read(run/'verification.json')
    for path,h in manifest['outputs'].items(): assert sha(run/path)==h,path
    assert verification['passed'] and hp['training_observations']==980
    assert sha(run/'input_samples.csv')==manifest['input_sha256']
    assert sha(ROOT/'testing_code'/'gp_model.py')==manifest['gp_model_sha256']
    candidates=csv_rows(run/'initialization_comparison.csv')
    selected=next(r for r in candidates if r['selected']=='True')
    assert float(selected['final_negative_training_mll'])==min(float(r['final_negative_training_mll']) for r in candidates)
    # Verify the earlier held-out artifacts, without writing into that experiment.
    test_root=ROOT.parent/'Minimal_GP_test'/'result'
    earlier=test_root/read(test_root/'latest.json')['run']
    previous=read(earlier/'manifest.json')
    for path,h in previous['outputs'].items(): assert sha(earlier/path)==h,path
    hist=csv_rows(run/'training_history.csv')
    fig,ax=plt.subplots(figsize=(8,4.2),layout='constrained')
    for initial in [.3,1.,3.]:
        rows=[r for r in hist if float(r['initial_lengthscale_scaled'])==initial]
        ax.plot([int(r['iteration']) for r in rows],[float(r['negative_mll_scaled']) for r in rows],label=f'Initial lengthscale {initial}')
    ax.set(xlabel='Adam iteration',ylabel='Negative training marginal log likelihood',
           title='Kernel fit using all 980 real observations')
    ax.legend(); ax.spines[['top','right']].set_visible(False)
    figures=ROOT/'result'/'figures'; figures.mkdir(exist_ok=True)
    fig.savefig(figures/'full_data_training.png',dpi=160); plt.close(fig)
    lx,ly,lt=hp['lengthscales_m_m_s']
    model=f'result/{run.name}/spatiotemporal_gp.pt'
    params=f'result/{run.name}/hyperparameters.json'
    readme=f'''# Trained Real-Data Kernel

## Purpose and model

Fit a reusable **spatiotemporal GP on all 980 aligned real observations** from the 2026-11-03 WELL-E lawnmower collection. This directory holds the full-data model separately from `Minimal_GP_test` and its held-out test checkpoints.

Model: `ConstantMean + ScaleKernel(RBF_ARD_xy * RBF_t)`, with a learned Gaussian observation-noise variance. The fit learns spatial and temporal lengthscales, signal variance, constant mean and likelihood noise. No Dashboard or Intel observations or pretrained weights are used. Accurate HOBO/GPS synchronization with zero additional offset is the accepted experimental assumption.

## Saved model and parameters

- **[Loadable model]({model})**: learned parameters, all standardized training inputs/targets and input/temperature scalers.
- **[Human-readable parameters]({params})**.
- [Coordinate/time reference](result/{run.name}/coordinate_system.json): original ENU origin and time origin.
- `result/latest.json` identifies the full-data model run. The earlier tests retain their own separate pointer.

| Parameter | Fitted value |
|---|---:|
| East spatial lengthscale | {lx:.6f} m |
| North spatial lengthscale | {ly:.6f} m |
| Temporal lengthscale | {lt:.6f} s |
| Signal variance | {hp['output_variance_c2']:.8f} °C² |
| Observation-noise variance | {hp['noise_variance_c2']:.8f} °C² |
| Observation-noise standard deviation | {hp['noise_std_c']:.6f} °C |
| Learned constant mean | {hp['mean_c']:.6f} °C |
| Training observations | {hp['training_observations']} |

The learned noise is a model parameter, not independent sensor calibration. These fitted lengthscales describe this one mobile run under the chosen kernel. All 980 measurements were used for fitting, so this model has **no independent held-out score**. The earlier split-test metrics describe their own separately trained models, not this checkpoint.

## Fit and records

CPU float64, four threads, 300 Adam steps per initialization, learning rate 0.05. Standardized initial lengthscales: 0.3, 1, 3. Select the lowest final negative **training** marginal log likelihood; selected initialization: {hp['selected_initial_lengthscale_scaled']}. Scalers are fitted on all 980 training observations. Model-family code is reused unchanged from `Code/GP3D_baseline_test/10.16_testing_code/3D_time&spatial temp field rebuild & prediction/gp_model.py`.

The [run directory](result/{run.name}) contains the input snapshot, protocol, all optimization curves, initialization comparison, parameters, checkpoint, training sample IDs, hashes, execution log and verification records. Source observations remain in `../Real_Data_Integration/result/gp_samples.csv` and the raw files remain under `Data/Real_spatialtemporal_data`. The current full-data run is `{run.name}`.

**{len(verification['checks'])} checks passed**: expected sample count, chronological finite inputs, all observations in the checkpoint, full-data scalers, finite positive parameters and predictive variances, noise handling, independent RBF/Cholesky calculation, and save/load equivalence. The previous held-out run's saved artifact hashes were also checked and remain unchanged. Numerical checks establish computational consistency, not future forecasting accuracy.

## Load and query

From this folder in a Python environment with PyTorch, GPyTorch and NumPy:

```python
from pathlib import Path
import json
import sys
import numpy as np

sys.path.insert(0, str(Path("testing_code").resolve()))
from gp_model import load_gp

run = json.loads(Path("result/latest.json").read_text())["run"]
gp = load_gp(Path("result") / run / "spatiotemporal_gp.pt")

# Input order: east metres, north metres, hours since the saved time origin.
# Example query only. Replace these values with your own ENU/time values.
query = np.array([[10.0, 5.0, 100.0 / 3600.0]])
prediction = gp.predict(query)
print(prediction["mean"])          # degrees Celsius
print(prediction["latent_var"])    # field variance, degrees Celsius squared
print(prediction["observed_var"])  # includes learned observation noise
```

Use the original ENU frame (x east, y north). The time origin is `{coords['time_origin_utc']}`; convert seconds since that origin to **hours** before calling `predict`. Pass physical coordinates and Celsius-scale data to the wrapper; its saved scaler applies standardization. No ROS `map` transform has been established yet. A query outside the original spatial or temporal support can have weak correlation to the training observations.

## Reproduce

With the existing `bunker-jazzy` container running, use the shared training runner:

```powershell
powershell -ExecutionPolicy Bypass -File .\\testing_code\\Train-Kernel.ps1
```

This calls `../Minimal_GP_test/testing_code/run_test.py --full-fit`, writes a **new run only into this directory**, and refreshes this README and the [executed Notebook](test_report.ipynb). The frozen full-fit protocol and worker are in `../Minimal_GP_test/testing_code/full_fit_protocol.json` and `full_fit_worker.py`; the shared numerical check helper is `worker.py`. The model-loading copy in this folder is byte-identical to the fitted model source, verified by SHA-256.

Training uses `/home/ubuntu/.venvs/policy-compare/bin/python` in Docker. Reports use the existing host reporting environment (Matplotlib, nbformat, nbclient and a Python Jupyter kernel); exact numerical versions are saved in the run manifest. New measurements used for a future independent evaluation must not already be among these 980 training observations. No commit or upload was performed.
'''
    (ROOT/'README.md').write_text(readme,encoding='utf-8')
    nb=nbformat.v4.new_notebook()
    nb.metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}}
    nb.cells=[nbformat.v4.new_markdown_cell('''# Full Real-Data Kernel Fit

Fit one reusable spatiotemporal GP to **all 980 aligned observations**. Clock synchronization with zero additional offset is accepted as an experimental assumption. This full-data model is stored separately from the minimal-test models. There is no independent test set for this fit.

The [README](README.md) describes the source data, exact units, dependencies and model-loading example.'''),
    nbformat.v4.new_code_cell('''from pathlib import Path
import json, hashlib, html
from IPython.display import display, HTML, Image
root=Path.cwd()
run=root/'result'/json.loads((root/'result'/'latest.json').read_text())['run']
hp=json.loads((run/'hyperparameters.json').read_text())
rows=[['East lengthscale (m)',hp['lengthscales_m_m_s'][0]],
      ['North lengthscale (m)',hp['lengthscales_m_m_s'][1]],
      ['Time lengthscale (s)',hp['lengthscales_m_m_s'][2]],
      ['Signal variance (°C²)',hp['output_variance_c2']],
      ['Observation-noise variance (°C²)',hp['noise_variance_c2']],
      ['Observation-noise SD (°C)',hp['noise_std_c']],
      ['Learned mean (°C)',hp['mean_c']],['Training observations',hp['training_observations']]]
display(HTML('<table><tr><th>Parameter</th><th>Value</th></tr>'+''.join(
    '<tr><td>'+html.escape(k)+'</td><td>'+str(round(v,6))+'</td></tr>' for k,v in rows)+'</table>'))
print('Model:',run/'spatiotemporal_gp.pt')
print('Kernel: ConstantMean + ScaleKernel(RBF_ARD_xy × RBF_t), Gaussian likelihood')'''),
    nbformat.v4.new_code_cell("display(Image(filename=str(root/'result'/'figures'/'full_data_training.png'),embed=True))"),
    nbformat.v4.new_markdown_cell('''## Verification and interpretation

The initialization is selected using training marginal likelihood. These curves measure training optimization, not held-out accuracy. The checkpoint includes the fitted kernel, all training observations and the scaler. Original coordinate and time references must be retained when querying it.'''),
    nbformat.v4.new_code_cell('''manifest=json.loads((run/'manifest.json').read_text())
for name,expected in manifest['outputs'].items():
    assert hashlib.sha256((run/name).read_bytes()).hexdigest()==expected,name
assert hashlib.sha256((run/'input_samples.csv').read_bytes()).hexdigest()==manifest['input_sha256']
assert hashlib.sha256((root/'testing_code'/'gp_model.py').read_bytes()).hexdigest()==manifest['gp_model_sha256']
v=json.loads((run/'verification.json').read_text()); assert v['passed']
print(len(v['checks']),'numerical/data checks passed. Saved model and loader hashes match.')
print('The worker checked reload predictions and independent NumPy/SciPy RBF/Cholesky calculations.')
print('This full-data fit has no independent accuracy estimate; previous test scores remain separate.')'''),
    nbformat.v4.new_markdown_cell('''## Result

The full-data kernel and conditioned GP are saved and loadable. They can be used for the next stage under the recorded experimental assumptions. Fit completion does not establish long-horizon accuracy or unique separation of spatial and temporal structure from one mobile run. Keep any new validation observations separate from these training data.''')]
    NotebookClient(nb,timeout=120,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}}).execute()
    nbformat.validate(nb); nbformat.write(nb,ROOT/'test_report.ipynb')
    print('Full-data model report saved separately:',ROOT)


if __name__=='__main__': main()

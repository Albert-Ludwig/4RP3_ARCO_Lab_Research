"""Fit all real observations; preserve a reusable full-data GP checkpoint."""
import csv
import hashlib
import json
import math
import platform
from pathlib import Path
import numpy as np
import scipy
import torch
import gpytorch
from gp_model import fit_gp, load_gp
from worker import independent_posterior, save_csv, dump, sha

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'result'


def main():
    config=json.loads((ROOT/'protocol.json').read_text(encoding='utf-8'))
    aligned=json.loads((ROOT/'alignment_summary.json').read_text(encoding='utf-8'))
    with (ROOT/'gp_samples.csv').open(encoding='utf-8',newline='') as f:
        records=list(csv.DictReader(f))
    n=len(records)
    x=np.array([[float(r[k]) for k in config['input_columns']] for r in records]); x[:,2]/=3600.
    y=np.array([float(r['temperature_c']) for r in records])
    checks=[]
    def check(name,condition,detail=None):
        checks.append(dict(name=name,passed=bool(condition),detail=detail))
        if not condition: raise AssertionError(name)
    check('all_expected_real_observations',n==config['expected_observations']==aligned['matched_count'])
    check('unique_chronological_finite_data',len({r['sample_id'] for r in records})==n
          and np.all(np.diff(x[:,2])>0) and np.isfinite(x).all() and np.isfinite(y).all())
    OUT.mkdir(exist_ok=True)
    dump(OUT/'protocol.json',config)
    dump(OUT/'coordinate_system.json',dict(**aligned['coordinate_system'],
         checkpoint_input_order=['x_east_m','y_north_m','t_h'],
         checkpoint_time_conversion='t_h = t_s / 3600; preserve the recorded time origin',
         accepted_clock_assumption=config['clock_assumption']))
    best=None; best_loss=math.inf; histories=[]; candidates=[]
    for initial in config['initial_lengthscales_scaled']:
        settings=dict(config['fit'],initial_lengthscale_scaled=initial)
        fitted=fit_gp(x,y,None,None,settings,seed=config['seed'],select_by_validation=False)
        fitted.model.train(); fitted.likelihood.train()
        with torch.no_grad(),gpytorch.settings.max_cholesky_size(10000):
            loss=float(-gpytorch.mlls.ExactMarginalLogLikelihood(fitted.likelihood,fitted.model)(
                fitted.model(fitted.model.train_inputs[0]),fitted.model.train_targets))
        check(f'initial_{initial}:finite_training',math.isfinite(loss)
              and all(math.isfinite(h['negative_mll_scaled']) for h in fitted.history))
        histories.extend(dict(initial_lengthscale_scaled=initial,**h) for h in fitted.history)
        candidates.append(dict(initial_lengthscale_scaled=initial,final_negative_training_mll=loss,
                               fit_seconds=fitted.fit_seconds))
        if loss<best_loss:
            best_loss=loss; best=fitted; chosen=initial
        print(f'Full-data GP: {n} rows, init={initial}, negative training MLL={loss:.6f}, {fitted.fit_seconds:.1f}s',flush=True)
    best.selection_method='All real observations; fixed 300-step fits; initialization chosen by training marginal likelihood only'
    for r in candidates: r['selected']=r['initial_lengthscale_scaled']==chosen
    hp=best.hyperparameters()
    hp.update(lengthscales_m_m_s=[*hp['lengthscales_m_m_h'][:2],hp['lengthscales_m_m_h'][2]*3600],
              noise_std_c=math.sqrt(hp['noise_variance_c2']),
              signal_std_c=math.sqrt(hp['output_variance_c2']),
              training_observations=n,selected_initial_lengthscale_scaled=chosen,
              final_negative_training_mll=best_loss,independent_test_performed=False)
    check('all_samples_in_checkpoint',len(best.model.train_targets)==n and np.allclose(
        best.model.train_inputs[0].detach().numpy(),best.scaler.transform_x(x),rtol=0,atol=1e-12))
    check('all_targets_in_checkpoint',np.allclose(best.model.train_targets.detach().numpy(),best.scaler.transform_y(y),rtol=0,atol=1e-12))
    check('full_data_scaler',np.allclose(best.scaler.x_mean,x.mean(0),atol=1e-12,rtol=0)
          and abs(best.scaler.y_mean-y.mean())<1e-12)
    check('finite_positive_hyperparameters',np.isfinite(hp['lengthscales_m_m_s']).all()
          and min(hp['lengthscales_m_m_s'])>0 and hp['noise_variance_c2']>0 and hp['output_variance_c2']>0)
    # Queries test numerical implementation only; no test accuracy is claimed.
    q=x[np.linspace(0,n-1,20,dtype=int)].copy()
    q[:,0]+=.25; q[:,1]-=.25; q[:,2]+=1./3600
    pred=best.predict(q)
    check('finite_positive_prediction',np.isfinite(pred['mean']).all()
          and np.isfinite(pred['observed_var']).all() and np.all(pred['observed_var']>0))
    check('noise_added_once',np.allclose(pred['observed_var']-pred['latent_var'],hp['noise_variance_c2'],rtol=1e-7,atol=1e-10))
    mu,var=independent_posterior(best,q)
    error=max(float(np.max(np.abs(mu-pred['mean']))),float(np.max(np.abs(var-pred['observed_var']))))
    check('independent_rbf_cholesky_prediction',error<1e-6,dict(max_abs_error=error))
    best.save(OUT/'spatiotemporal_gp.pt')
    reloaded=load_gp(OUT/'spatiotemporal_gp.pt').predict(q)
    check('checkpoint_reload',np.allclose(reloaded['mean'],pred['mean'],atol=1e-9,rtol=0)
          and np.allclose(reloaded['observed_var'],pred['observed_var'],atol=1e-9,rtol=0))
    save_csv(OUT/'training_history.csv',histories)
    save_csv(OUT/'initialization_comparison.csv',candidates)
    dump(OUT/'hyperparameters.json',hp)
    dump(OUT/'training_summary.json',dict(training_observations=n,run_id=records[0]['run_id'],
        training_time_range_utc=[records[0]['timestamp_utc'],records[-1]['timestamp_utc']],
        source_sample_ids=[r['sample_id'] for r in records],
        purpose='Full-data fit for reuse; no independent held-out accuracy estimate',
        prior_test_models_unchanged=True,clock_assumption=config['clock_assumption']))
    dump(OUT/'verification.json',dict(passed=all(c['passed'] for c in checks),checks=checks))
    dump(OUT/'manifest.json',dict(input_sha256=sha(ROOT/'gp_samples.csv'),
        protocol_source_sha256=sha(ROOT/'protocol.json'),gp_model_sha256=sha(ROOT/'gp_model.py'),
        worker_sha256=sha(Path(__file__)),shared_worker_sha256=sha(ROOT/'worker.py'),
        alignment_summary_sha256=sha(ROOT/'alignment_summary.json'),
        runtime=dict(python=platform.python_version(),torch=torch.__version__,gpytorch=gpytorch.__version__,
                     numpy=np.__version__,scipy=scipy.__version__),
        outputs={p.relative_to(OUT).as_posix():sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()}))
    print('FULL FIT COMPLETE',json.dumps(hp['lengthscales_m_m_s']),flush=True)


if __name__=='__main__': main()

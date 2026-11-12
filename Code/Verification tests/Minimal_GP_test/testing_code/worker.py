"""Docker worker; reuses the existing gp_model implementation without changes."""
import csv
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path
import numpy as np
import scipy
from scipy.linalg import cho_factor, cho_solve, solve_triangular
import torch
import gpytorch
from gp_model import fit_gp, load_gp

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'result'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def save_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)


def metrics(y,mu,var=None):
    error=mu-y
    result=dict(rmse_c=float(np.sqrt(np.mean(error**2))),mae_c=float(np.mean(np.abs(error))))
    if var is not None:
        sigma=np.sqrt(var)
        result.update(coverage_95=float(np.mean(np.abs(error)<=1.959963984540054*sigma)),
                      mean_interval_width_c=float(np.mean(2*1.959963984540054*sigma)),
                      marginal_nlpd=float(np.mean(.5*(np.log(2*np.pi*var)+error**2/var))))
    else:
        result.update(coverage_95=None,mean_interval_width_c=None,marginal_nlpd=None)
    return result


def independent_posterior(fitted, query):
    """Direct NumPy/SciPy RBF + Cholesky crosscheck of GPyTorch predictions."""
    model=fitted.model
    tx=model.train_inputs[0].detach().numpy(); ty=model.train_targets.detach().numpy()
    q=fitted.scaler.transform_x(query)
    ls=np.concatenate([k.lengthscale.detach().numpy().ravel() for k in model.rbf_components()])
    scale=float(model.covar_module.outputscale.detach())
    noise=float(fitted.likelihood.noise.detach()); mean=float(model.mean_module.constant.detach())
    def kernel(a,b):
        return scale*np.exp(-.5*np.sum(((a[:,None,:]-b[None,:,:])/ls)**2,axis=-1))
    k=kernel(tx,tx)+noise*np.eye(len(tx)); cross=kernel(tx,q)
    factor=cho_factor(k,lower=True)
    mu=mean+cross.T@cho_solve(factor,ty-mean)
    v=scale-np.sum(solve_triangular(factor[0],cross,lower=True)**2,axis=0)
    return mu*fitted.scaler.y_std+fitted.scaler.y_mean,(v+noise)*fitted.scaler.y_std**2


def main():
    cfg=json.loads((ROOT/'protocol.json').read_text(encoding='utf-8'))
    with (ROOT/'gp_samples.csv').open(encoding='utf-8',newline='') as f:
        records=list(csv.DictReader(f))
    n=len(records)
    x=np.array([[float(r[k]) for k in ('x_east_m','y_north_m','t_s')] for r in records])
    x[:,2]/=3600.
    y=np.array([float(r['temperature_c']) for r in records])
    ids=[r['sample_id'] for r in records]
    assert len(set(ids))==n and np.all(np.diff(x[:,2])>0) and np.isfinite(x).all() and np.isfinite(y).all()
    OUT.mkdir(exist_ok=True)
    b=cfg['blocked_reconstruction']; lo=int(n*b['start_fraction']); hi=int(n*b['end_fraction'])
    test=np.arange(lo,hi)
    guard=b['guard_seconds']/3600.
    train=np.flatnonzero((x[:,2]<x[lo,2]-guard)|(x[:,2]>x[hi-1,2]+guard))
    cut=int(n*cfg['forward_prediction']['training_fraction'])
    splits={'blocked_reconstruction':(train,test),'forward_prediction':(np.arange(cut),np.arange(cut,n))}
    checks=[]
    def check(name,condition,detail=None):
        checks.append(dict(name=name,passed=bool(condition),detail=detail))
        if not condition: raise AssertionError(name)
    split_records=[]; split_info={}
    for task,(tr,te) in splits.items():
        buffer=sorted(set(range(n))-set(tr)-set(te))
        check(task+':disjoint_and_complete',not(set(tr)&set(te)) and len(tr)+len(te)+len(buffer)==n)
        if task=='forward_prediction':
            check(task+':strict_causality',x[tr[-1],2]<x[te[0],2])
        else:
            sep=np.min(np.abs(x[tr,2,None]-x[te,2][None,:]))*3600
            check(task+':guard',sep>b['guard_seconds']-1e-8,{'minimum_seconds':float(sep)})
        nearest=np.min(np.linalg.norm(x[te,None,:2]-x[None,tr,:2],axis=-1),axis=1)
        split_info[task]=dict(train_count=len(tr),test_count=len(te),buffer_count=len(buffer),
            train_time_range_utc=[records[tr[0]]['timestamp_utc'],records[tr[-1]]['timestamp_utc']],
            test_time_range_utc=[records[te[0]]['timestamp_utc'],records[te[-1]]['timestamp_utc']],
            nearest_train_distance_m=dict(min=float(nearest.min()),median=float(np.median(nearest)),max=float(nearest.max())),
            forecast_horizon_seconds=float((x[te[-1],2]-x[tr[-1],2])*3600) if task=='forward_prediction' else None)
        for role,index in [('train',tr),('test',te),('buffer',buffer)]:
            for i in index:
                split_records.append(dict(task=task,role=role,row_index=int(i),sample_id=ids[i],timestamp_utc=records[i]['timestamp_utc']))
    # Freeze all partitions before fitting or evaluating any model.
    save_csv(OUT/'splits.csv',split_records); dump(OUT/'splits.json',split_info)
    dump(OUT/'protocol.json',cfg)
    all_metrics=[]; prediction_rows=[]; fit_rows=[]; histories=[]; hyperparameters={}
    for task,(tr,te) in splits.items():
        print(f'TASK {task}: train={len(tr)}, test={len(te)}',flush=True)
        task_hyp={}
        names=['train_mean']+(['last_observation'] if task=='forward_prediction' else [])+['temporal_gp','spatiotemporal_gp']
        for name in names:
            var=latent=None
            if name in ('train_mean','last_observation'):
                mu=np.full(len(te),y[tr].mean() if name=='train_mean' else y[tr[-1]])
            else:
                cols=[2] if name=='temporal_gp' else [0,1,2]
                tx=x[tr][:,cols]; q=x[te][:,cols]
                best=None; best_loss=math.inf; candidates=[]
                for initial in cfg['initial_lengthscales_scaled']:
                    settings=dict(cfg['fit'],initial_lengthscale_scaled=initial)
                    fit=fit_gp(tx,y[tr],None,None,settings,seed=cfg['seed'],select_by_validation=False)
                    fit.model.train(); fit.likelihood.train()
                    with torch.no_grad(),gpytorch.settings.max_cholesky_size(10000):
                        loss=float(-gpytorch.mlls.ExactMarginalLogLikelihood(fit.likelihood,fit.model)(
                            fit.model(fit.model.train_inputs[0]),fit.model.train_targets))
                    row=dict(task=task,model=name,initial_lengthscale_scaled=initial,
                             final_negative_training_mll=loss,fit_seconds=fit.fit_seconds)
                    candidates.append(row)
                    histories.extend(dict(task=task,model=name,initial_lengthscale_scaled=initial,**h) for h in fit.history)
                    check(f'{task}:{name}:init{initial}:finite_training',math.isfinite(loss) and all(math.isfinite(h['negative_mll_scaled']) for h in fit.history))
                    if loss<best_loss:
                        best_loss=loss; best=fit; chosen=initial
                    print(f'  {name} init={initial}: training negative MLL={loss:.6f} ({fit.fit_seconds:.1f}s)',flush=True)
                for row in candidates:
                    row['selected']=row['initial_lengthscale_scaled']==chosen
                    fit_rows.append(row)
                best.selection_method='Fixed 300 Adam steps per initialization; initialization selected by final training marginal likelihood only'
                pred=best.predict(q); mu=pred['mean']; var=pred['observed_var']; latent=pred['latent_var']
                hp=best.hyperparameters()
                if name=='temporal_gp':
                    hp['kernel']='ScaleKernel(RBF_t); temporal-only baseline on a moving sensor'
                    hp['lengthscale_seconds']=hp['lengthscale_h']*3600
                else:
                    hp['lengthscales_m_m_s']=[*hp['lengthscales_m_m_h'][:2],hp['lengthscales_m_m_h'][2]*3600]
                hp['selected_initial_lengthscale_scaled']=chosen
                hp['final_negative_training_mll']=best_loss
                task_hyp[name]=hp
                check(task+':'+name+':train_only_scaler',np.allclose(best.scaler.x_mean,tx.mean(0),rtol=0,atol=1e-12)
                      and abs(best.scaler.y_mean-y[tr].mean())<1e-12)
                check(task+':'+name+':finite_positive_prediction',np.isfinite(mu).all() and np.isfinite(var).all() and np.all(var>0))
                check(task+':'+name+':noise_added_once',np.allclose(var-latent,hp['noise_variance_c2'],rtol=1e-7,atol=1e-10))
                manual_mu,manual_var=independent_posterior(best,q)
                err=max(float(np.max(np.abs(manual_mu-mu))),float(np.max(np.abs(manual_var-var))))
                check(task+':'+name+':independent_cholesky',err<1e-6,{'max_abs_error':err})
                path=OUT/task/(name+'.pt'); path.parent.mkdir(parents=True,exist_ok=True); best.save(path)
                reloaded=load_gp(path).predict(q[:5])
                check(task+':'+name+':checkpoint_roundtrip',np.allclose(reloaded['mean'],mu[:5],atol=1e-9,rtol=0)
                      and np.allclose(reloaded['observed_var'],var[:5],atol=1e-9,rtol=0))
            met=metrics(y[te],mu,var)
            all_metrics.append(dict(task=task,model=name,n_test=len(te),**met))
            for j,i in enumerate(te):
                prediction_rows.append(dict(task=task,model=name,sample_id=ids[i],timestamp_utc=records[i]['timestamp_utc'],
                    t_s=float(x[i,2]*3600),x_east_m=float(x[i,0]),y_north_m=float(x[i,1]),
                    observed_temperature_c=float(y[i]),predicted_temperature_c=float(mu[j]),
                    latent_variance_c2=float(latent[j]) if latent is not None else '',
                    observed_variance_c2=float(var[j]) if var is not None else '',
                    lower_95_c=float(mu[j]-1.959963984540054*np.sqrt(var[j])) if var is not None else '',
                    upper_95_c=float(mu[j]+1.959963984540054*np.sqrt(var[j])) if var is not None else ''))
            print(f'  {name}: RMSE={met["rmse_c"]:.6f} C',flush=True)
        hyperparameters[task]=task_hyp
    save_csv(OUT/'metrics.csv',all_metrics); save_csv(OUT/'predictions.csv',prediction_rows)
    save_csv(OUT/'training_history.csv',histories); save_csv(OUT/'initialization_comparison.csv',fit_rows)
    dump(OUT/'hyperparameters.json',hyperparameters)
    # Independent per-record metric reconstruction after CSV serialization.
    with (OUT/'predictions.csv').open(newline='') as f: saved=list(csv.DictReader(f))
    for row in all_metrics:
        rr=[r for r in saved if r['task']==row['task'] and r['model']==row['model']]
        errors=[float(r['predicted_temperature_c'])-float(r['observed_temperature_c']) for r in rr]
        check(row['task']+':'+row['model']+':saved_metrics',
              abs(math.sqrt(math.fsum(e*e for e in errors)/len(errors))-row['rmse_c'])<1e-12
              and abs(math.fsum(abs(e) for e in errors)/len(errors)-row['mae_c'])<1e-12)
    dump(OUT/'verification.json',dict(passed=all(c['passed'] for c in checks),checks=checks))
    dump(OUT/'manifest.json',dict(input_sha256=sha(ROOT/'gp_samples.csv'),
        protocol_source_sha256=sha(ROOT/'protocol.json'),gp_model_sha256=sha(ROOT/'gp_model.py'),worker_sha256=sha(Path(__file__)),
        runtime=dict(python=platform.python_version(),torch=torch.__version__,gpytorch=gpytorch.__version__,numpy=np.__version__,scipy=scipy.__version__),
        outputs={p.relative_to(OUT).as_posix():sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()}))
    print('COMPLETE',flush=True)


if __name__=='__main__':
    main()

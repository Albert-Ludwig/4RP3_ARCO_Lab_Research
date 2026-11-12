"""OFFLINE ONLY: this is the sole consumer of withheld temperature labels."""
import argparse
from pathlib import Path
import numpy as np
from .gp import GaussianProcess
from .storage import read_json,read_csv,save_json,save_csv,digest


def evaluate(bundle,comparison):
    bundle=Path(bundle);comparison=Path(comparison);online=bundle/'online'
    hidden=read_csv(bundle/'offline/hidden.csv');acq=read_csv(online/'acquisition.csv');history=read_csv(online/'history.csv')
    if {r['record_id'] for r in hidden}&{r['record_id'] for r in acq+history}:raise ValueError('Hidden record identity leakage')
    q=np.array([[float(r[k]) for k in ('x_m','y_m','t_s')] for r in hidden]);truth=np.array([float(r['temperature_c']) for r in hidden])
    hp=read_json(online/'parameters.json');config=read_json(online/'config.json');summaries=[];curves=[];predictions=[]
    hx=np.array([[float(r[k]) for k in ('x_m','y_m','t_s')] for r in history]);hy=np.array([float(r['temperature_c']) for r in history])
    def metrics(gp,method,time_s,count):
        mean=gp.posterior(q)[0];e=mean-truth
        return dict(method=method,time_s=time_s,measurements=count,rmse_c=float(np.sqrt(np.mean(e**2))),mae_c=float(np.mean(abs(e)))),mean
    base=GaussianProcess(hx,hy,hp);m,pred=metrics(base,'History Only',0.,0)
    summaries.append(dict(m,distance_m=0.,duration_s=0.,return_home_success=None,status='no_navigation_control'))
    curves.append(m);curves.append(dict(m,time_s=config['budget_s']))
    for r,prediction in zip(hidden,pred):predictions.append(dict(method='History Only',record_id=r['record_id'],true_c=float(r['temperature_c']),predicted_c=float(prediction)))
    for policy in ('mi','emse','wivr'):
        run=comparison/policy
        if not (run/'mission.json').is_file():continue
        mission=read_json(run/'mission.json')
        for snapshot in read_json(run/'snapshots.json'):
            data=np.load(run/snapshot['gp_file']);gp=GaussianProcess(data['x'],data['y'],hp)
            row,_=metrics(gp,policy.upper(),snapshot['time_s'],snapshot['measurements'])
            row['mean_query_variance_c2']=snapshot['mean_query_variance_c2'];curves.append(row)
        data=np.load(run/'gp_final.npz');final=GaussianProcess(data['x'],data['y'],hp)
        m,pred=metrics(final,policy.upper(),mission['duration_s'],mission['successful_measurements'])
        summaries.append(dict(m,distance_m=mission['distance_m'],duration_s=mission['duration_s'],
            return_home_success=mission['return_home_success'],status=mission['status']))
        for r,prediction in zip(hidden,pred):predictions.append(dict(method=policy.upper(),record_id=r['record_id'],true_c=float(r['temperature_c']),predicted_c=float(prediction)))
    save_csv(comparison/'summary.csv',summaries);save_csv(comparison/'reconstruction_curves.csv',curves)
    save_csv(comparison/'hidden_predictions.csv',predictions)
    result=dict(mode=config['mode'],hidden_count=len(hidden),hidden_sha256=digest(bundle/'offline/hidden.csv'),
        shared_hidden_for_all_methods=True,retrospective_reconstruction=True,
        full_real_comparison_complete=config['mode']=='real_recorded' and len(summaries)==4 and all(r['status']=='passed' for r in summaries[1:]),
        synthetic_fixture=config['mode']=='synthetic_fixture',summary=summaries,
        note=config.get('fixture_note','Fixed hidden inputs for all policies. Curves re-evaluate the full interval retrospectively, not future forecasting.'))
    save_json(comparison/'evaluation.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',required=True);p.add_argument('--comparison',required=True)
    a=p.parse_args();print(evaluate(a.bundle,a.comparison))

"""Offline synthetic recorded fixture. Never imported by the online mission."""
import argparse
from pathlib import Path
import numpy as np
from .storage import read_json,save_json,save_csv
from .prepare import ROOT,REPO,FIELDS,prepare


def generate(output,full_budget=False):
    output=Path(output).resolve()
    if output.exists(): raise FileExistsError(output)
    output.mkdir(parents=True)
    config=read_json(ROOT/'config/real.json')
    scene=read_json(REPO/'Code/ROS_simulation/Nav2_test/testing_code/src/well_e_nav2_test/config/scenario.json')
    # Nodes on both sides require Nav2 to travel around the known central box.
    points=[[0.,0.],[0.,2.],[6.,2.],[6.,-2.]]
    spatial=[[float(x),float(y)] for x in np.arange(-.5,7.,1.5)
             for y in (-2.5,0.,2.5) if not (2.0<=x<=4.0 and abs(y)<1.5)]
    parameters=dict(lengthscales_m_m_s=[2.5,2.5,180.],output_variance_c2=1.,
                    noise_variance_c2=.01,mean_c=20.)
    save_json(output/'parameters.json',parameters)
    paths=[]
    # Analytic fixture values are materialized OFFLINE; online can only look up records.
    def temperature(x,y,t):
        return 20.+.9*np.sin(x/3)+.4*np.cos(y/2)+.55*np.sin(t/170+x/6)
    for run in range(5):
        rows=[]
        times=np.linspace(-400+run*80,-345+run*80,6) if run<4 else np.arange(0.,901.,1.)
        sites=spatial if run<4 else points[1:]
        for i,(x,y) in enumerate(sites):
            for j,t in enumerate(times):
                rows.append(dict(zip(FIELDS,[f'synthetic-{run}-{i}-{j}',f'synthetic-{run}',x,y,float(t),float(temperature(x,y,t))])))
        path=output/f'dataset_{run}.csv';save_csv(path,rows,FIELDS);paths.append(str(path))
    config.update(mode='synthetic_fixture',dataset_files=paths,
        recorded_epoch_utc='2026-01-01T00:00:00Z',mission_start_s=0.,scene=scene,
        candidate_xy_m=points,query_xy_m=spatial,
        priority_roi={'bounds_m':[-1.,7.5,1.,3.], 'multiplier':3.,
                      'rationale':'Predeclared upper-field monitoring priority; no temperature-based tuning.'},
        parameters_file=str(output/'parameters.json'),fit_initial_gp=False,
        max_measurements=None if full_budget else 3,
        fixture_note=('SYNTHETIC 900-second-budget Nav2 comparison; no acquisition cap; NOT a real-field result.' if full_budget else
                      'SYNTHETIC integration smoke test, capped at 3 acquisitions; NOT a real-field or full-budget policy ranking.'))
    save_json(output/'config.json',config)
    return prepare(output/'config.json',output/'bundle')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--full-budget',action='store_true');a=p.parse_args()
    print(generate(a.output,a.full_budget))

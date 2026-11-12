"""Offline preparation and compatibility gate; only this stage splits run 4."""
import argparse
from collections import defaultdict
from pathlib import Path
import math
import numpy as np
from .storage import read_json,save_json,read_csv,save_csv,digest
from .objectives import query_grid,priority_weights

ROOT=Path(__file__).resolve().parents[2]
REPO=ROOT.parents[1]
FIELDS=['record_id','run_id','x_m','y_m','t_s','temperature_c']


def validate_records(rows):
    if not rows: raise ValueError('Empty recorded dataset')
    ids=[r['record_id'] for r in rows]
    if len(set(ids))!=len(ids): raise ValueError('Duplicate record identity within dataset')
    for r in rows:
        if not all(k in r for k in FIELDS): raise ValueError('Canonical schema required: '+','.join(FIELDS))
        for k in FIELDS[2:]:
            if not math.isfinite(float(r[k])): raise ValueError('Nonfinite recorded value')


def split_experiment(rows,points,tolerance):
    """Alternate eligible records; retain off-candidate records for hidden evaluation."""
    groups=defaultdict(list); excluded=[]
    for r in rows:
        d=np.linalg.norm(np.asarray(points)-[float(r['x_m']),float(r['y_m'])],axis=1)
        index=int(np.argmin(d))
        if d[index]>tolerance: excluded.append(r); continue
        groups[index].append(r)
    acquired=[]; withheld=[]
    for index,group in sorted(groups.items()):
        group.sort(key=lambda r:(float(r['t_s']),r['record_id']))
        acquired.extend(group[::2]); withheld.extend(group[1::2])
    # Candidate placement must not discard independent measurements elsewhere in
    # the field. They cannot be acquired online but remain useful hidden labels.
    withheld.extend(excluded)
    return acquired,withheld,[]


def prepare(config_path,output):
    config_path=Path(config_path).resolve(); config=read_json(config_path); output=Path(output).resolve()
    if output.exists(): raise FileExistsError('Bundle exists; use a new directory')
    def resolve(p):
        path=Path(p); return path if path.is_absolute() else REPO/path
    required=['recorded_epoch_utc','mission_start_s','scene','candidate_xy_m','query_xy_m','priority_roi']
    missing=[name for name in required if config.get(name) is None]
    paths=config.get('dataset_files',[])
    if len(paths)!=5: missing.append('five datasets (0-4)')
    else:
        missing.extend(f'Dataset {i}' for i,p in enumerate(paths) if p is None or not resolve(p).is_file())
    if config['mode']=='real_recorded' and (not config.get('map_yaml') or not resolve(config['map_yaml']).is_file()):
        missing.append('navigation map YAML/image registered to map-frame data')
    if missing:
        output.mkdir(parents=True)
        status=dict(status='blocked_missing_inputs',missing=missing,config=str(config_path),
                    final_real_comparison_complete=False)
        save_json(output/'preflight.json',status)
        return status
    records=[read_csv(resolve(p)) for p in paths]
    for rows in records: validate_records(rows)
    allids=[r['record_id'] for rows in records for r in rows]
    if len(set(allids))!=len(allids): raise ValueError('Record IDs must be globally unique across datasets 0-4')
    epoch=config['recorded_epoch_utc']
    from datetime import datetime
    stamp=datetime.fromisoformat(epoch.replace('Z','+00:00'))
    if stamp.tzinfo is None: raise ValueError('recorded_epoch_utc must include timezone')
    start=float(config['mission_start_s']); end=start+float(config['budget_s'])
    history=[r for rows in records[:4] for r in rows]
    if max(float(r['t_s']) for r in history)>=start:
        raise ValueError('Historical records must precede the common mission start')
    # Translate the common time origin, never compress or stretch recorded time.
    for rows in records:
        for r in rows: r['t_s']=float(r['t_s'])-start
    points=np.asarray(config['candidate_xy_m'],dtype=float)
    if points.ndim!=2 or points.shape[1]!=2 or len(points)<3 or len(np.unique(points,axis=0))!=len(points):
        raise ValueError('At least Home + two distinct sensing nodes required')
    scene=config['scene']; xmin,xmax,ymin,ymax=scene['map_bounds_m']
    if not all(xmin<x<xmax and ymin<y<ymax for x,y in points): raise ValueError('Candidate outside map bounds')
    if not np.allclose(points[0],scene['start'][:2]): raise ValueError('Home must equal the initial robot pose')
    experiment=[r for r in records[4] if 0<=r['t_s']<=config['budget_s']]
    acquired,withheld,excluded=split_experiment(experiment,points,config['acquisition_space_tolerance_m'])
    if len(withheld)<config['minimum_hidden_records']: raise ValueError('Insufficient withheld records')
    counts=[sum(np.linalg.norm(points[i]-[float(r['x_m']),float(r['y_m'])])<=config['acquisition_space_tolerance_m'] for r in acquired) for i in range(1,len(points))]
    if min(counts)<config['minimum_acquisition_records_per_node']:
        raise ValueError('Acquisition coverage insufficient for one or more sensing nodes; do not interpolate missing labels')
    t=np.arange(0,config['budget_s'],config['query_time_step_s']).tolist()+[config['budget_s']]
    q=query_grid(config['query_xy_m'],t); weights=priority_weights(q,config['priority_roi'])
    # A Euclidean lower bound can reject obviously unreachable record windows;
    # actual feasibility is gated again using Nav2 path travel times at runtime.
    possible=[]
    for i,p in enumerate(points[1:],start=1):
        lower=float(np.linalg.norm(p-points[0])/config['nominal_speed_m_s'])
        possible.append(any(np.linalg.norm(p-[float(r['x_m']),float(r['y_m'])])<=config['acquisition_space_tolerance_m']
                            and lower<=r['t_s']<=config['budget_s']-lower-config['sensing_s'] for r in acquired))
    if not all(possible): raise ValueError('Recorded availability incompatible even with minimum travel times')
    output.mkdir(parents=True)
    online=output/'online'; offline=output/'offline'; online.mkdir(); offline.mkdir()
    save_csv(online/'history.csv',history,FIELDS); save_csv(online/'acquisition.csv',acquired,FIELDS)
    save_csv(offline/'hidden.csv',withheld,FIELDS)
    save_csv(offline/'excluded.csv',excluded,FIELDS)
    np.savez_compressed(online/'query.npz',queries=q,weights=weights)
    if config.get('fit_initial_gp'):
        # Reuse unchanged fitted GP implementation; time unit conversion is explicit.
        import importlib.util,sys
        source=REPO/'Code/GP3D_baseline_test/10.16_testing_code/3D_time&spatial temp field rebuild & prediction/gp_model.py'
        spec=importlib.util.spec_from_file_location('original_training_gp',source)
        module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
        hx=np.array([[float(r['x_m']),float(r['y_m']),r['t_s']/3600] for r in history])
        hy=np.array([float(r['temperature_c']) for r in history])
        fit=module.fit_gp(hx,hy,None,None,config['historical_fit'],seed=0,select_by_validation=False)
        hp=fit.hyperparameters(); hp['lengthscales_m_m_s']=[*hp['lengthscales_m_m_h'][:2],hp['lengthscales_m_m_h'][2]*3600]
        fit.save(online/'initial_training.pt');save_json(online/'parameters.json',hp)
        save_csv(online/'fit_history.csv',fit.history)
    else:
        hp=read_json(resolve(config['parameters_file']));save_json(online/'parameters.json',hp)
    if config['mode']=='real_recorded':
        from shutil import copy2
        import yaml
        map_path=resolve(config['map_yaml']); mapdata=yaml.safe_load(map_path.read_text())
        image=(map_path.parent/mapdata['image']).resolve()
        target=online/('map_image'+image.suffix);copy2(image,target)
        mapdata['image']=target.name
        (online/'map.yaml').write_text(yaml.safe_dump(mapdata))
        config['map_yaml']='map.yaml'
    public={k:v for k,v in config.items() if k not in ('dataset_files','parameters_file','fit_initial_gp','hidden_split')}
    public.update(recorded_epoch_utc=epoch,recorded_time_offset_s=start,mission_start_s=0.,
                  history_file='history.csv',acquisition_file='acquisition.csv',parameters_file='parameters.json',query_file='query.npz')
    save_json(online/'config.json',public)
    status=dict(status='prepared',mode=config['mode'],history_count=len(history),acquisition_count=len(acquired),
                hidden_count=len(withheld),excluded_count=len(excluded),query_count=len(q),temporal_queries_s=t,
                historical_source_runs=[str(p) for p in paths[:4]],experiment_source_run=str(paths[4]),
                online_inputs_exclude_hidden=True,source_sha256={str(p):digest(resolve(p)) for p in paths},
                time_mapping='recorded t = mission_start_s + (ROS clock - clock at mission start); no time scaling',
                nav2_compatibility='Euclidean lower-bound gate passed; Nav2-time availability checked before mission execution')
    save_json(output/'preflight.json',status)
    save_json(output/'manifest.json',{'files':{p.relative_to(output).as_posix():digest(p) for p in output.rglob('*') if p.is_file()}})
    return status


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    a=p.parse_args(); result=prepare(a.config,a.output);print(result)
    raise SystemExit(0 if result['status']=='prepared' else 2)


if __name__=='__main__':main()

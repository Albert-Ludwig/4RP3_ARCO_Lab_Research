"""Offline common-input, no-hidden-update and physical-recording acceptance audit."""
import argparse
import hashlib
from pathlib import Path
import numpy as np
from .storage import read_json,read_csv,save_json,digest


def audit(bundle,comparison):
    import sys
    sys.path.append('/usr/lib/python3/dist-packages')
    import yaml
    from .runner import verify_bag
    bundle=Path(bundle);comparison=Path(comparison);online=bundle/'online'
    config=read_json(online/'config.json');save_json(comparison/'run_configuration.json',config)
    hidden=read_csv(bundle/'offline/hidden.csv');hidden_ids={r['record_id'] for r in hidden}
    pool={r['record_id']:r for r in read_csv(online/'acquisition.csv')}
    history=read_csv(online/'history.csv');parameters=digest(online/'parameters.json')
    hx=np.array([[float(r[k]) for k in ('x_m','y_m','t_s')] for r in history]);hy=np.array([float(r['temperature_c']) for r in history])
    checks=dict(hidden_pool_disjoint=not (hidden_ids&set(pool)),multiple_updates=True,within_budget_and_home=True,
        original_record_tuples=True,hidden_not_used_for_update=True,measurement_tolerances=True,
        identical_initial_GP=True,identical_parameters=True,physical_rosbag_acceptance=True,
        reset_physical_pose_and_clock=True,independent_bag_return_home=True)
    worlds=[];maps=[];navigation_configs=[];shared_input_hashes=[]
    for policy in ('mi','emse','wivr'):
        run=comparison/policy;mission=read_json(run/'mission.json');records=read_csv(run/'measurements.csv')
        trajectory=read_csv(run/'trajectory.csv');first_pose=trajectory[0]
        checks['reset_physical_pose_and_clock'] &= (0<=mission['clock_origin_s']<10 and
            __import__('math').hypot(float(first_pose['x_m'])-config['scene']['start'][0],
                                    float(first_pose['y_m'])-config['scene']['start'][1])<.02)
        checks['multiple_updates'] &= mission['successful_measurements']>=2 and mission['planning_decisions']>=2
        checks['within_budget_and_home'] &= mission['within_budget'] and mission['return_home_success'] and mission['status']=='passed'
        checks['identical_parameters'] &= mission['initial_parameters_sha256']==parameters
        snapshots=read_json(run/'snapshots.json');initial=np.load(run/snapshots[0]['gp_file'])
        checks['identical_initial_GP'] &= np.array_equal(initial['x'],hx) and np.array_equal(initial['y'],hy)
        final=np.load(run/'gp_final.npz')
        checks['identical_initial_GP'] &= np.array_equal(final['x'][:len(hx)],hx) and np.array_equal(final['y'][:len(hy)],hy)
        for index,r in enumerate(records):
            checks['hidden_not_used_for_update'] &= r['record_id'] not in hidden_ids
            original=pool[r['record_id']]
            checks['original_record_tuples'] &= all(float(r[k])==float(original[k]) for k in ('x_m','y_m','t_s','temperature_c'))
            checks['original_record_tuples'] &= np.array_equal(final['x'][len(hx)+index],[float(r[k]) for k in ('x_m','y_m','t_s')])
            checks['original_record_tuples'] &= float(final['y'][len(hy)+index])==float(r['temperature_c'])
            checks['measurement_tolerances'] &= (float(r['position_error_m'])<=config['acquisition_space_tolerance_m'] and
                abs(float(r['time_error_s']))<=config['acquisition_time_tolerance_s'] and
                float(r['update_time_s'])-float(r['actual_t_s'])>=config['sensing_s']-1e-6)
        bag=verify_bag(run,config);checks['physical_rosbag_acceptance'] &= bag['status']=='passed'
        # Independently verify final physical Home position from the recording,
        # rather than relying only on the mission's navigation success flag.
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from nav_msgs.msg import Odometry
        reader=rosbag2_py.SequentialReader()
        reader.open(rosbag2_py.StorageOptions(uri=str(run/'rosbag'),storage_id='mcap'),rosbag2_py.ConverterOptions('',''))
        reader.set_filter(rosbag2_py.StorageFilter(topics=['/simulation/ground_truth/odom']))
        last=None
        while reader.has_next():
            _,data,_=reader.read_next();last=deserialize_message(data,Odometry).pose.pose.position
        home_error=__import__('math').hypot(last.x-config['scene']['start'][0],last.y-config['scene']['start'][1]) if last is not None else None
        checks['independent_bag_return_home'] &= home_error is not None and home_error<=config['acquisition_space_tolerance_m']
        bag['recorded_final_Home_error_m']=home_error
        save_json(run/'rosbag_summary.json',bag)
        worlds.append(digest(run/'generated/world.sdf'));maps.append(digest(run/'generated/map.pgm') if (run/'generated/map.pgm').exists() else digest(run/'generated/map.yaml'))
        params=yaml.safe_load((run/'generated/nav2_params.yaml').read_text());params['map_server']['ros__parameters']['yaml_filename']='<same per-run map>'
        navigation_configs.append(hashlib.sha256(yaml.safe_dump(params,sort_keys=True).encode()).hexdigest())
        first=__import__('json').loads((run/'events.jsonl').read_text().splitlines()[0]);shared_input_hashes.append(first['online_input_hashes'])
    checks['identical_world']=len(set(worlds))==1;checks['identical_map']=len(set(maps))==1
    checks['identical_Nav2_parameters_except_run_path']=len(set(navigation_configs))==1
    checks['identical_public_bundle']=all(item==shared_input_hashes[0] for item in shared_input_hashes)
    predictions=read_csv(comparison/'hidden_predictions.csv')
    checks['identical_hidden_inputs_for_all_methods']=all({r['record_id'] for r in predictions if r['method']==method}==hidden_ids for method in ('History Only','MI','EMSE','WIVR'))
    result=dict(passed=all(checks.values()),checks={k:bool(v) for k,v in checks.items()},
        hidden_sha256=digest(bundle/'offline/hidden.csv'),world_sha256=worlds[0],
        normalized_Nav2_parameters_sha256=navigation_configs[0],
        notes='Audit uses offline saved logs and bag readback. Does not certify sensor synchronization, real-map registration or field policy superiority.')
    save_json(comparison/'fairness.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',required=True);p.add_argument('--comparison',required=True)
    a=p.parse_args();result=audit(a.bundle,a.comparison);print(result)
    raise SystemExit(0 if result['passed'] else 1)

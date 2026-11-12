"""Online mission: public bundle only; no hidden-label file is opened here."""
import argparse
from pathlib import Path
import math
import time
import traceback
import numpy as np
# Keep the GP venv's NumPy/Torch, then allow ROS apt-installed Python dependencies.
import sys
sys.path.append('/usr/lib/python3/dist-packages')
import rclpy
from rclpy.signals import SignalHandlerOptions
from .storage import read_json,read_csv,save_json,save_csv,EventLog,digest
from .gp import GaussianProcess
from .objectives import Objective
from .acquisition import Acquisition
from .planner import make_planner
from .navigation import Navigation
from .prepare import REPO


def run(online,output,policy):
    online=Path(online).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    config=read_json(online/'config.json');log=EventLog(output/'events.jsonl')
    history=read_csv(online/config['history_file'])
    x=[[float(r['x_m']),float(r['y_m']),float(r['t_s'])] for r in history]
    gp=GaussianProcess(x,[float(r['temperature_c']) for r in history],read_json(online/config['parameters_file']))
    query=np.load(online/config['query_file']);obj=Objective(policy,query['queries'],query['weights'])
    acq=Acquisition(online/config['acquisition_file'],config['acquisition_space_tolerance_m'],config['acquisition_time_tolerance_s'])
    points=np.asarray(config['candidate_xy_m']);n=len(points);budget=config['budget_s']
    reserve=config['return_reserve_s'];compute_reserve=config['computation_reserve_s']
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO);nav=Navigation(config,log)
    legs=[];measurements=[];snapshots=[];plans=[];matrix_rows=[];home_success=False
    status='failed';error=None;distance=0.;clock_origin=None;blocked=set();stop_reason=None
    def snapshot():
        variance=obj.mean_variance(gp)
        name=f'gp_{len(snapshots):03d}.npz';gp.save(output/name)
        snapshots.append(dict(time_s=nav.now(),measurements=len(measurements),gp_file=name,mean_query_variance_c2=variance))
    def estimate(a,b,label):
        value=nav.estimate(a,b)
        matrix_rows.append(dict(time_s=nav.now(),label=label,start_x_m=float(a[0]),start_y_m=float(a[1]),
            goal_x_m=float(b[0]),goal_y_m=float(b[1]),estimated_s=value['seconds'] if math.isfinite(value['seconds']) else None,
            planned_length_m=value['length_m'] if math.isfinite(value['length_m']) else None))
        return value
    try:
        nav.ready();clock_origin=nav.start()
        log.write('mission_start',policy=policy,mode=config['mode'],ros_clock_origin_s=clock_origin,
                  recorded_time_offset_s=config['recorded_time_offset_s'],budget_s=budget,
                  online_input_hashes={p.name:digest(p) for p in online.iterdir() if p.is_file()})
        # Fixed candidate-to-candidate paths are cached; actual-start row is refreshed
        # at every decision and all cached paths are rebuilt after a failure.
        def build_cache():
            matrix=np.zeros((n,n))
            for i in range(n):
                for j in range(n):
                    if i!=j:matrix[i,j]=estimate([*points[i],0.],[*points[j],0.],f'{i}->{j}')['seconds']
            return matrix
        cached=build_cache()
        np.save(output/'initial_travel_matrix_s.npy',cached)
        compatible=[]
        for i in range(1,n):
            available=acq.next_time(points[i],nav.now()+cached[0,i],budget-cached[i,0]-reserve-config['sensing_s'])
            if math.isfinite(available):compatible.append(i)
        save_json(output/'nav2_compatibility.json',dict(compatible_nodes=compatible,total_sensing_nodes=n-1,
                  passed=len(compatible)>=2,checked_using='actual Nav2 path costs and public recorded times'))
        if len(compatible)<2:raise RuntimeError('Recorded pool incompatible with Nav2 timing for multiple sensing decisions')
        blocked.update(set(range(1,n))-set(compatible));snapshot()
        while nav.now()<budget-reserve:
            if config['max_measurements'] is not None and len(measurements)>=config['max_measurements']:
                stop_reason='declared_synthetic_smoke_acquisition_cap';break
            nav.phase='planning';actual=nav.position()
            home=estimate(actual,[*points[0],0.],'actual->home')
            if not math.isfinite(home['seconds']):raise RuntimeError('Home unreachable from actual robot position')
            if nav.now()+home['seconds']+reserve+compute_reserve>=budget:
                stop_reason='return_reserve';break
            dynamic_points=np.vstack([points,actual[:2]])
            travel=np.full((n+1,n+1),np.inf);travel[:n,:n]=cached;travel[n,n]=0.
            for j in range(n):travel[n,j]=estimate(actual,[*points[j],0.],f'actual->{j}')['seconds']
            planner=make_planner(REPO/'Code/Robot_close_loop_system/testing_code/baseline/grg.py',
                                 dynamic_points,travel,acq,obj,config)
            elapsed=nav.now();wall=time.perf_counter()
            plan=planner.plan(gp,n,elapsed,max(0,budget-reserve-compute_reserve-elapsed),blocked=[0,n,*blocked])
            entry=dict(index=len(plans),time_s=elapsed,finish_compute_s=nav.now(),route=plan.route,
                arrivals_s=plan.arrivals_s,planned_finish_s=plan.finish_s if math.isfinite(plan.finish_s) else None,
                score=plan.gain,policy=policy,wall_compute_s=time.perf_counter()-wall,
                ros_compute_s=nav.now()-elapsed,evaluations=plan.evaluations,gp_version=gp.version,
                objective_calls=obj.calls,objective_cache_hits=obj.cache_hits)
            plans.append(entry);log.write('grg_plan',**entry)
            print(f'{policy}: t={nav.now():.1f}s route={plan.route} gain={plan.gain:.5f} gp={gp.version}',flush=True)
            if len(plan.route)<3 or plan.route[1]==0:
                stop_reason='no_feasible_informative_acquisition';break
            node=plan.route[1];leg_est=travel[n,node];return_est=cached[node,0]
            # Planned times may be stale after computation; recheck acquisition and
            # return with the current ROS clock before sending an actual goal.
            latest=budget-return_est-reserve-config['sensing_s']
            availability=acq.next_time(points[node],nav.now()+leg_est,latest)
            if not math.isfinite(availability):blocked.add(node);continue
            nav.phase='navigating';start=nav.now()
            result=nav.navigate([*points[node],0.],min(latest,nav.now()+max(90.,leg_est*3)),
                guard=lambda:nav.now()+return_est+reserve<budget)
            leg=dict(index=len(legs),node=int(node),phase='sensing',start_s=start,finish_s=nav.now(),
                     estimated_s=float(leg_est),planned_length_m=None,**result)
            legs.append(leg);log.write('navigation_result',**leg)
            if not result['success']:
                # No measurement on failure; discard cached paths before replanning.
                blocked.add(node);cached=build_cache();continue
            actual=nav.position();home=estimate(actual,[*points[0],0.],'arrival->home')
            def safe():return nav.now()+home['seconds']+reserve+config['sensing_s']<budget
            sample_time=acq.next_time(actual[:2],nav.now(),budget-home['seconds']-reserve-config['sensing_s'])
            if not math.isfinite(sample_time):blocked.add(node);continue
            nav.phase='waiting'
            if not nav.wait_until(sample_time,safe):stop_reason='return_reserve_during_wait';break
            record=acq.acquire(nav.position()[:2],nav.now())
            if record is None:
                log.write('acquisition_miss',node=int(node),time_s=nav.now());continue
            nav.phase='sensing'
            if not nav.wait_until(nav.now()+config['sensing_s'],safe):
                stop_reason='return_reserve_during_dwell';break
            # Update with the original recorded tuple. Actual pose/time and mismatch
            # are retained separately; no relabeling at a fabricated coordinate.
            gp.append([record['x_m'],record['y_m'],record['t_s']],record['temperature_c'])
            record.update(node=int(node),update_time_s=nav.now(),gp_version=gp.version)
            measurements.append(record);log.write('gp_update',**record)
            from std_msgs.msg import String
            import json
            msg=String();msg.data=json.dumps(dict(event='gp_update',record_id=record['record_id'],time_s=nav.now(),gp_version=gp.version))
            nav.events.publish(msg);snapshot()
        nav.phase='returning'
        actual=nav.position();home=estimate(actual,[*points[0],0.],'return->home');start=nav.now()
        result=nav.navigate([*points[0],0.],budget)
        legs.append(dict(index=len(legs),node=0,phase='return',start_s=start,finish_s=nav.now(),
                         estimated_s=home['seconds'] if math.isfinite(home['seconds']) else None,planned_length_m=home['length_m'],**result))
        home_success=result['success'] and math.dist(nav.position()[:2],points[0])<=.25
        nav.phase='finished';snapshot()
        if home_success and nav.now()<=budget and len(measurements)>=2 and len(plans)>=2:status='passed'
        else:error='Acceptance requires multiple decisions/updates, timely return, and physical Home arrival'
    except Exception as exc:
        error=f'{type(exc).__name__}: {exc}';log.write('exception',error=error,traceback=traceback.format_exc())
        print(traceback.format_exc(),flush=True)
        if clock_origin is not None:
            try:
                nav.phase='emergency_return';result=nav.navigate([*points[0],0.],budget)
                home_success=result['success']
            except Exception:pass
    finally:
        duration=nav.now() if clock_origin is not None else 0.
        trajectory=list(nav.trajectory)
        distance=sum(math.hypot(b['x_m']-a['x_m'],b['y_m']-a['y_m']) for a,b in zip(trajectory,trajectory[1:]))
        save_csv(output/'trajectory.csv',trajectory,['time_s','x_m','y_m','yaw_rad','phase'])
        save_csv(output/'measurements.csv',measurements,list(dict.fromkeys(k for r in measurements for k in r)) or ['record_id'])
        save_csv(output/'navigation.csv',legs);save_csv(output/'travel_estimates.csv',matrix_rows)
        save_json(output/'plans.json',plans);save_json(output/'snapshots.json',snapshots)
        gp.save(output/'gp_final.npz')
        summary=dict(status=status,policy=policy,mode=config['mode'],error=error,stop_reason=stop_reason,
            successful_measurements=len(measurements),planning_decisions=len(plans),gp_updates=gp.version,
            distance_m=distance,duration_s=duration,return_home_success=home_success,
            budget_s=budget,within_budget=duration<=budget,clock_origin_s=clock_origin,
            final_pose=nav.truth,synthetic_acquisition_cap=config['max_measurements'],
            hidden_data_accessed_online=False,initial_parameters_sha256=digest(online/config['parameters_file']))
        save_json(output/'mission.json',summary);nav.close();rclpy.shutdown()
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--online',required=True);p.add_argument('--output',required=True)
    p.add_argument('--policy',choices=['mi','emse','wivr'],required=True);a=p.parse_args()
    result=run(a.online,a.output,a.policy);print(result,flush=True)
    raise SystemExit(0 if result['status']=='passed' else 1)

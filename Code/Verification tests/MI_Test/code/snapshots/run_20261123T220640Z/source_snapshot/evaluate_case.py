"""Audit latest-GP scheduling, data retention, mutual information scores and arrival-only route handoff."""
from pathlib import Path
import site,sys,os,json,math
site.addsitedir('/usr/lib/python3/dist-packages');os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1];REPO=ROOT.parents[2];VER=REPO/'Code/Verification tests';SUITE=Path(sys.argv[1]);RUN=Path(sys.argv[2])
sys.path.insert(0,str(VER/'Continuous_sensing_test/testing_code'));sys.path.insert(0,str(VER/'Scoring_policy_test/testing_code'));sys.path.insert(0,str(VER/'SRF_test/testing_code'))
from components import BatchGP
from scoring import AdaptiveWeights,Grid,Objective
from srf_field import TemperatureField
def read(p):return json.loads(p.read_text())
def lines(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def model(j):
    p=dict(lengthscales_m_m_s=j['length'],output_variance_c2=j['variance'],noise_variance_c2=j['noise'],mean_c=j['mean'])
    return BatchGP(np.array(j['x']),np.array(j['y']),p,[str(i) for i in range(len(j['y']))])
events=lines(RUN/'events.jsonl');m=read(RUN/'mission.json');worker=read(RUN/'worker_status.json')
samples=[e for e in events if e['event']=='temperature' and e.get('ok')];updates=[e for e in events if e['event']=='gp_update']
initial=read(RUN/'initial.json');final=read(RUN/'final_gp.json');base=model(initial);gp=model(final)
assert len(samples)==m['measurements']
assert len(updates)==m['gp_updates']
assert len(gp.x)==len(base.x)+len(samples)
np.testing.assert_allclose(gp.x[len(base.x):],[[r['x'],r['y'],r['t']] for r in samples],atol=1e-12,rtol=0)
np.testing.assert_allclose(gp.y[len(base.y):],[r['temperature'] for r in samples],atol=1e-12,rtol=0)
field=TemperatureField(read(RUN/'field.json'));measured=field.measure(gp.x[len(base.x):],[str(r['id']) for r in samples])
assert np.allclose(measured,gp.y[len(base.y):],atol=1e-12,rtol=0)
plans=sorted([read(p) for p in RUN.glob('plan_*.json')],key=lambda j:j['request']['id'])
assert [p['request']['id'] for p in plans]+[j['id'] for j in read(RUN/'unprocessed_requests.json')]==list(range(1,m['grg_requested']+1))
assert len(plans)+worker['unprocessed']==m['grg_requested']
initial_plan=read(RUN/'initial_plan.json')
policy,=read(RUN/'protocol.json')['policies']
assert plans and all(p['policy']==policy for p in [initial_plan,*plans])
audits=[]
grid=Grid([30,20],5,None,np.arange(0,301,2))
selected=[(0,0,initial_plan)]
for index in sorted({0,len(plans)//2,len(plans)-1}):
    p=plans[index]
    selected.append((p['request']['id'],p['request']['assimilated_new_count'],p))
for plan_id,version,p in selected:
    frozen=dict(initial)
    frozen['x']=initial['x']+[[r['x'],r['y'],r['t']] for r in samples[:version]]
    frozen['y']=initial['y']+[r['temperature'] for r in samples[:version]]
    g=model(frozen)
    weight=AdaptiveWeights(grid).refresh(g,p['request']['weight_t'] if plan_id else 0) if policy=='wivr' else None
    value=Objective(policy,g,grid,weight).score(np.array(p['events']).reshape(-1,3))
    error=abs(value-p['score'])
    assert np.isclose(value,p['score'],atol=1e-8,rtol=1e-8),(plan_id,value,p['score'])
    audits.append(dict(id=plan_id,absolute_error=error,passed=True))
# Use recorded Gazebo trajectories, not planned line segments.
import rosbag2_py
from rclpy.serialization import deserialize_message
from nav_msgs.msg import Odometry
reader=rosbag2_py.SequentialReader();reader.open(rosbag2_py.StorageOptions(uri=str(ROOT/'recordings'/SUITE.name/RUN.name/'rosbag'),storage_id='mcap'),rosbag2_py.ConverterOptions('',''))
trajectory=[]
while reader.has_next():
    topic,data,_=reader.read_next()
    if topic!='/simulation/ground_truth/odom':continue
    p=deserialize_message(data,Odometry);t=p.header.stamp.sec+p.header.stamp.nanosec/1e9-m['origin']
    if 0<=t<=m['duration_s']:trajectory.append([t,p.pose.pose.position.x+15,p.pose.pose.position.y+10])
trajectory=np.array(trajectory);np.save(RUN/'trajectory.npy',trajectory)
home_xy=np.array(read(RUN/'candidates.json')[0])
assert np.array_equal(home_xy,read(RUN/'protocol.json')['home_m'])
spawn_offset_m=float(np.linalg.norm(trajectory[0,1:3]-home_xy))
spawn_verified=bool(trajectory[0,0]<=.20 and spawn_offset_m<=.30)
resources=lines(RUN/'resources.jsonl');cpu_deltas=[]
for before,after in zip(resources,resources[1:]):
    prior={p['pid']:p['cpu_s'] for p in before['processes']};dt=after['wall']-before['wall']
    if dt>0:cpu_deltas.append(100*sum(max(0,p['cpu_s']-prior.get(p['pid'],p['cpu_s'])) for p in after['processes'])/dt)
peak=max(sum(p['rss'] for p in r['processes']) for r in resources)/2**20
backlog=[];queued=done=0
for e in events:
    if e['event']=='grg_start':queued+=1
    if e['event']=='grg_complete':done+=1
    if e['event'] in ('grg_start','grg_complete'):backlog.append([e['event_t'],queued-done])
np.save(RUN/'backlog.npy',np.array(backlog))
q=np.load(SUITE/'common/evaluation_grid.npz');query=q['queries'];truth=q['truth']
def predict(g):return np.concatenate([g.posterior(query[i:i+256])[0] for i in range(0,len(query),256)])
pred=predict(gp);base_pred=predict(base)
def stats(values):
    a=np.array(values,dtype=float)
    return dict(mean=float(a.mean()),p95=float(np.percentile(a,95)),max=float(a.max())) if len(a) else None
deadline_success=sum(p['latency_s']<=m['update_interval_s'] and not p['timeout'] and not p['finished_after_window'] for p in plans)
queries=[e for e in events if e['event']=='query']
last_tick=min(math.floor(m['sampling_duration_s']+1e-9),math.ceil(m['sampling_budget_s'])-1)
expected=list(range(1,last_tick+1))
checks=dict(sensing_complete=[r['id'] for r in samples]==[r['id'] for r in queries]==expected and m['missed_ticks']==0 and m['stale_pose']==0,
    all_samples_assimilated=len(gp.x)==len(base.x)+len(samples),
    no_silent_request_drops=len(plans)+worker['unprocessed']==m['grg_requested'],
    sampled_completed_route_scores_audited=all(x['passed'] for x in audits),
    no_navigation_preemptions=m['goal_preemptions_before_deadline']==0,
    spawned_near_home=spawn_verified,
    configured_policy_used=True,no_search_timeouts=all(not p["timeout"] for p in [initial_plan,*plans]),
    no_navigation_failures=m["navigation_failures"]==0,
    returned_home=m['home_returned'] and m['home_error_m']<=.3,
    within_sampling_budget=0<m['sampling_duration_s']<=m['sampling_budget_s']
        and all(0<=e['t']<m['sampling_budget_s'] for e in samples))
arrival_choices=[e for e in events if e['event']=='arrival_selection']
summary=dict(policy=policy,home_xy_m=home_xy.tolist(),first_recorded_pose=trajectory[0].tolist(),spawn_offset_m=spawn_offset_m,initial_route=initial_plan["route"],mission=m,worker=worker,checks=checks,score_audits=audits,
    diagnostic_all_searches_within_1s=worker['completed_total']==m['grg_requested'] and deadline_success==m['grg_requested'],
    diagnostic_searches_within_1s=deadline_success,grg_completed_during_window=m['grg_completed_at_stop'],
    grg_compute_s=stats([p['total_compute_s'] for p in plans]),grg_search_s=stats([p['seconds'] for p in plans]),
    grg_queue_wait_s=stats([p['queue_wait_s'] for p in plans]),grg_latency_s=stats([p['latency_s'] for p in plans]),
    grg_timeouts=sum(p['timeout'] for p in plans),consumed_plan_age_s=stats([e['plan_age_s'] for e in arrival_choices if 'plan_age_s' in e]),gp_update_total_s=sum(u['seconds'] for u in updates),grg_compute_total_s=sum(p['total_compute_s'] for p in plans),gp_update_s=stats([u['seconds'] for u in updates]),
    initial_plan_s=read(RUN/'initial_plan.json')['seconds'],sampled_process_peak_mib=peak,sampled_aggregate_cpu_peak_percent=max(cpu_deltas),sampled_aggregate_cpu_mean_percent=float(np.mean(cpu_deltas)),
    srf_source_ms=stats([r['source_ms'] for r in samples]),sensor_roundtrip_ms=stats([r['roundtrip_ms'] for r in samples]),
    distance_m=float(np.linalg.norm(np.diff(trajectory[:,1:3],axis=0),axis=1).sum()),
    real_time_factor=m['duration_s']/m['wall_s'],final_reconstruction_rmse_c=float(np.sqrt(np.mean((pred-truth)**2))),
    history_only_rmse_c=float(np.sqrt(np.mean((base_pred-truth)**2))),
    arrival_route_sources={k:sum(e['route_source']==k for e in arrival_choices) for k in ('fresh_completed','retained_tail')},
    interpretation='One replaceable latest-state signal; GRG starts with latest offered GP and navigation context, never restarts for each GP update. Retain the executable sampling tail; switch only at arrival during sampling. Final return begins after sampling stops. No hard GP age rejection. All observations assimilated at 1 Hz. Actual started jobs have contiguous IDs; coalesced state signals are explicitly counted. Initial planning precedes mission. Final in-flight search may complete after 300 s.')
starts=[e for e in events if e['event']=='grg_start']
assert len(starts)==m['grg_requested']
latest=0;latest_epoch=0;fresh_at_start=True
for e in events:
    if e['event']=='gp_update':latest=e['version']
    if e['event']=='nav_goal':latest_epoch=e['goal_epoch']
    if e['event']=='grg_start':
        # A log write can race the next GP update, so use the locked snapshot's recorded queue time.
        available=[u['version'] for u in updates if u['wall']<=e['queued_wall']]
        fresh_at_start &= e['gp_version']==max(available,default=0)
summary['checks'].update(single_pending_state=worker['peak_queue']<=1,
 fresh_snapshot_when_offered=bool(fresh_at_start),all_started_jobs_recorded=len(plans)==len(starts),
 retained_route_handoff=all(e.get('remaining_route',[None])[0] in range(11) for e in arrival_choices))
# Independently replay the logged navigation state and verify result eligibility.
plan_by_id={p['request']['id']:p for p in plans}
active_goal=False;current_node=0;arrived_epoch=0;arrival_only=True;matching=True;budget_ok=True
edge_cost={(j['from'],j['to']):j['duration_s'] for j in read(RUN/'graph.json')}
for e in events:
    if e['event']=='nav_goal':
        arrival_only &= not active_goal
        active_goal=True
    elif e['event']=='arrived':
        active_goal=False;current_node=e['target'];arrived_epoch=e['goal_epoch']
    elif e['event']=='sampling_goal_ended':
        active_goal=False
    elif e['event']=='arrival_selection':
        matching &= e['remaining_route'][0]==current_node
        if e['route_source']=='fresh_completed':
            req=plan_by_id[e['selected_plan_id']]['request']
            matching &= req['anchor']==current_node and req['goal_epoch']==arrived_epoch
        budget_ok &= e['event_t']+edge_cost[current_node,e['next']]<=m['sampling_budget_s']+.05
[sampling_stop]=[e for e in events if e['event']=='sampling_finished']
[return_start]=[e for e in events if e['event']=='return_started']
no_edge_fits=all(sampling_stop['event_t']+cost>m['sampling_budget_s']
                 for (a,b),cost in edge_cost.items() if a==sampling_stop['last_waypoint'] and a!=b)
[return_goal]=[e for e in events if e['event']=='nav_goal' and e['returning']]
summary['checks'].update(arrival_only_from_event_trace=bool(arrival_only),
 matching_route_anchor_and_epoch=bool(matching),sampling_legs_fit_budget=bool(budget_ok),
 sampling_plan_cost_matches_executed_prefix=all(
     abs(p['predicted_navigation_s']-p.get('request',{}).get('planning_start_t',0)
         -sum(edge_cost[a,b] for a,b in zip(p['route'],p['route'][1:])))<=1e-8
     and (p['predicted_navigation_s']<=m['sampling_budget_s']+1e-8 or len(p['route'])==1)
     for p in [initial_plan,*plans]),
 stop_reason_verified=(no_edge_fits and not sampling_stop['has_feasible_next_leg']
     if m['sampling_stop_reason']=='no_sampling_leg_fits' else m['sampling_duration_s']==m['sampling_budget_s']),
 return_after_sampling=return_goal['target']==0 and return_start['event_t']>=sampling_stop['event_t']
     and m['return_duration_s']>0 and all(q['t']<=sampling_stop['event_t'] for q in queries))
summary['sampling_distance_m']=float(np.linalg.norm(np.diff(trajectory[trajectory[:,0]<sampling_stop['event_t'],1:3],axis=0),axis=1).sum())
summary['sampling_intermediate_home_visits']=sum(e['event']=='arrived' and e['target']==0
    and e['event_t']<sampling_stop['event_t'] for e in events)
summary['sampling_stop']=sampling_stop
summary['grg_snapshot_age_at_start_s']=stats([p['request']['queued_t']+p['queue_wait_s']-p['request']['sample_t'] for p in plans])
summary['grg_feature_total_s']=sum(p['feature_seconds'] for p in plans)
summary['grg_objective_total_s']=sum(p['objective_seconds'] for p in plans)
summary['empty_event_searches']=sum(not p['events'] for p in plans)
summary['scheduler_acceptance']=all(summary['checks'].values())
(RUN/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print('METRICS',m['update_interval_s'],json.dumps({k:v for k,v in summary.items() if k not in ('checks','score_audits','interpretation')}),flush=True)
# Functional pass/fail is recorded rather than hiding failed load-test outcomes.

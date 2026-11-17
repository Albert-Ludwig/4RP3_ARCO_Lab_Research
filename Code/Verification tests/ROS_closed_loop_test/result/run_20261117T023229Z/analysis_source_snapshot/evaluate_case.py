"""Audit the one-policy load test; distinguish current GP from queued GRG versions."""
from pathlib import Path
import site,sys,os,json
site.addsitedir('/usr/lib/python3/dist-packages');os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1];REPO=ROOT.parents[2];VER=REPO/'Code/Verification tests';SUITE=Path(sys.argv[1]);RUN=Path(sys.argv[2])
sys.path.insert(0,str(VER/'Continuous_sensing_test/testing_code'));sys.path.insert(0,str(VER/'Scoring_policy_test/testing_code'));sys.path.insert(0,str(VER/'SRF_test/testing_code'))
from components import BatchGP
from scoring import Grid,AdaptiveWeights,Objective
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
audits=[]
if plans:
    for index in sorted(set([0,len(plans)//2,len(plans)-1])):
        p=plans[index];version=p['request']['assimilated_new_count'];frozen=dict(initial)
        frozen['x']=initial['x']+[[r['x'],r['y'],r['t']] for r in samples[:version]];frozen['y']=initial['y']+[r['temperature'] for r in samples[:version]]
        g=model(frozen);grid=Grid([30,20],5,None,np.arange(0,301,2));weight=AdaptiveWeights(grid).refresh(g,p['request']['weight_t'])
        value=Objective('wivr',g,grid,weight).score(np.array(p['events']).reshape(-1,3));error=abs(value-p['score'])
        assert np.isclose(value,p['score'],atol=1e-8,rtol=1e-8)
        audits.append(dict(id=p['request']['id'],absolute_error=error,passed=True))
# Use recorded Gazebo trajectories, not planned line segments.
import rosbag2_py
from rclpy.serialization import deserialize_message
from nav_msgs.msg import Odometry
reader=rosbag2_py.SequentialReader();reader.open(rosbag2_py.StorageOptions(uri=str(RUN/'rosbag'),storage_id='mcap'),rosbag2_py.ConverterOptions('',''))
trajectory=[]
while reader.has_next():
    topic,data,_=reader.read_next()
    if topic!='/simulation/ground_truth/odom':continue
    p=deserialize_message(data,Odometry);t=p.header.stamp.sec+p.header.stamp.nanosec/1e9-m['origin']
    if 0<=t<=m['duration_s']:trajectory.append([t,p.pose.pose.position.x+15,p.pose.pose.position.y+10])
trajectory=np.array(trajectory);np.save(RUN/'trajectory.npy',trajectory)
resources=lines(RUN/'resources.jsonl');cpu_deltas=[]
for before,after in zip(resources,resources[1:]):
    prior={p['pid']:p['cpu_s'] for p in before['processes']};dt=after['wall']-before['wall']
    if dt>0:cpu_deltas.append(100*sum(max(0,p['cpu_s']-prior.get(p['pid'],p['cpu_s'])) for p in after['processes'])/dt)
peak=max(sum(p['rss'] for p in r['processes']) for r in resources)/2**20
backlog=[];queued=done=0
for e in events:
    if e['event']=='gp_update' and not e['terminal']:queued+=1
    if e['event']=='grg_complete':done+=1
    if e['event'] in ('gp_update','grg_complete'):backlog.append([e['event_t'],queued-done])
np.save(RUN/'backlog.npy',np.array(backlog))
q=np.load(SUITE/'common/evaluation_grid.npz');query=q['queries'];truth=q['truth']
def predict(g):return np.concatenate([g.posterior(query[i:i+256])[0] for i in range(0,len(query),256)])
pred=predict(gp);base_pred=predict(base)
def stats(values):
    a=np.array(values,dtype=float)
    return dict(mean=float(a.mean()),p95=float(np.percentile(a,95)),max=float(a.max())) if len(a) else None
deadline_success=sum(p['latency_s']<=m['update_interval_s'] and not p['timeout'] and not p['finished_after_window'] for p in plans)
checks=dict(sensing_complete=len(samples)==299 and m['missed_ticks']==0 and m['stale_pose']==0,
    all_samples_assimilated=len(gp.x)==1279 and [r['id'] for r in samples]==list(range(1,300)),
    no_silent_request_drops=len(plans)+worker['unprocessed']==m['grg_requested'],
    sampled_completed_route_scores_audited=all(x['passed'] for x in audits),
    no_navigation_preemptions=m['goal_preemptions_before_deadline']==0,
    returned_home=m['home_returned'] and m['home_error_m']<=.3,full_window=300<=m['duration_s']<300.5)
arrival_choices=[e for e in events if e['event']=='arrival_selection']
summary=dict(mission=m,worker=worker,checks=checks,score_audits=audits,
    update_interval_feasible=worker['completed_total']==m['grg_requested'] and deadline_success==m['grg_requested'],
    grg_deadlines_met=deadline_success,grg_completed_during_window=m['grg_completed_at_stop'],
    grg_compute_s=stats([p['total_compute_s'] for p in plans]),grg_search_s=stats([p['seconds'] for p in plans]),
    grg_queue_wait_s=stats([p['queue_wait_s'] for p in plans]),grg_latency_s=stats([p['latency_s'] for p in plans]),
    grg_timeouts=sum(p['timeout'] for p in plans),consumed_plan_age_s=stats([e['plan_age_s'] for e in arrival_choices]),gp_update_total_s=sum(u['seconds'] for u in updates),grg_compute_total_s=sum(p['total_compute_s'] for p in plans),gp_update_s=stats([u['seconds'] for u in updates]),
    initial_plan_s=read(RUN/'initial_plan.json')['seconds'],sampled_process_peak_mib=peak,sampled_aggregate_cpu_peak_percent=max(cpu_deltas),sampled_aggregate_cpu_mean_percent=float(np.mean(cpu_deltas)),
    srf_source_ms=stats([r['source_ms'] for r in samples]),sensor_roundtrip_ms=stats([r['roundtrip_ms'] for r in samples]),
    distance_m=float(np.linalg.norm(np.diff(trajectory[:,1:3],axis=0),axis=1).sum()),
    real_time_factor=m['duration_s']/m['wall_s'],final_reconstruction_rmse_c=float(np.sqrt(np.mean((pred-truth)**2))),
    history_only_rmse_c=float(np.sqrt(np.mean((base_pred-truth)**2))),
    interpretation='Fixed-WIVR GP/GRG update-interval load test, not a scoring-policy comparison. FIFO requests are not coalesced. GP updates follow the configured 1 or 5 s interval; each completed GRG uses that request-specific frozen GP and its sample-time weights. Goal changes only at arrival. Unfinished requests are preserved, not counted as executed. At 300 s no new work starts; one in-flight computation may finish afterwards.')
(RUN/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print('METRICS',m['update_interval_s'],json.dumps({k:v for k,v in summary.items() if k not in ('checks','score_audits','interpretation')}),flush=True)
# Functional pass/fail is recorded rather than hiding failed load-test outcomes.

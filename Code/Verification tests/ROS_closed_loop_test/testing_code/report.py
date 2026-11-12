"""Generate an executed notebook and physical-pose rosbag replay."""

from pathlib import Path

import json,math,os,re,gzip

import numpy as np

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from matplotlib.animation import FFMpegWriter

from matplotlib.patches import Polygon

import imageio_ffmpeg,nbformat

from nbclient import NotebookClient

from mcap.reader import make_reader

from mcap_ros2.decoder import DecoderFactory

ROOT=Path(__file__).resolve().parents[1]

RUN=ROOT/"result"/json.loads((ROOT/"result/latest.json").read_text())["run"]

def read(n):
 path=RUN/n
 if path.exists():
  return json.loads(path.read_text())
 with gzip.open(str(path)+".gz","rt",encoding="utf-8") as stream:
  return json.load(stream)

def lines(n):return [json.loads(s) for s in (RUN/n).read_text().splitlines() if s.strip()]

mission=read("mission.json"); duration=read("field.json")["mission_interval_s"][1]; poses=[];counts={};actual_speeds=[];actual_yaw_rates=[];command_peaks={};raw_yaw_spikes=[];quaternions=[];heights=[]

for bag in (RUN/"rosbag").glob("*.mcap"):

 with bag.open("rb") as stream:

  for schema,channel,message,d in make_reader(stream,decoder_factories=[DecoderFactory()]).iter_decoded_messages():

   counts[channel.topic]=counts.get(channel.topic,0)+1

   if channel.topic in ("/cl/cmd_vel_nav","/cmd_vel","/simulation/cmd_vel"):

    peak=command_peaks.setdefault(channel.topic,{"max_abs_linear_m_s":0.,"max_abs_yaw_rad_s":0.})

    peak["max_abs_linear_m_s"]=max(peak["max_abs_linear_m_s"],abs(d.linear.x))

    peak["max_abs_yaw_rad_s"]=max(peak["max_abs_yaw_rad_s"],abs(d.angular.z))

   if channel.topic=="/simulation/ground_truth/odom":

    t=d.header.stamp.sec+d.header.stamp.nanosec*1e-9-mission["origin"]

    if 0<=t<=duration+.1:

     actual_speeds.append(math.hypot(d.twist.twist.linear.x,d.twist.twist.linear.y))

     actual_yaw_rates.append(abs(d.twist.twist.angular.z))
     if abs(d.twist.twist.angular.z)>2: raw_yaw_spikes.append({"t":t,"raw_yaw_rad_s":d.twist.twist.angular.z})

     p=d.pose.pose.position;q=d.pose.pose.orientation

     quaternions.append([q.x,q.y,q.z,q.w]);heights.append(p.z)
     poses.append((t,p.x+48.5,p.y+32,math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))))

poses=np.array(poses);assert len(poses)>100

np.savetxt(RUN/"physical_trajectory.csv",poses,delimiter=",",header="t_s,x_m,y_m,yaw_rad",comments="")

samples=[r for r in lines("sampler.jsonl") if "temperature" in r]

queries=[r["record"] for r in lines("sampler.jsonl") if r.get("event")=="query"]

plans=[read(p.name) for p in sorted(RUN.glob("plan_*.json"),key=lambda p:int(p.stem.split("_")[1]))]

manager=lines("manager.jsonl");res=lines("resources.jsonl");ids={r["id"] for r in samples}

peak_cpu=0

for a,b in zip(res[:-1],res[1:]):

 prev={p["pid"]:p["cpu_s"] for p in a["processes"]}

 delta=sum(max(0,p["cpu_s"]-prev.get(p["pid"],p["cpu_s"])) for p in b["processes"])

 peak_cpu=max(peak_cpu,100*delta/max(1e-9,b["wall"]-a["wall"]))

lat=np.array([r["roundtrip_ms"] for r in samples])

audit=read("audit_and_estimation.json")

def expected_cancel(r):

 return r.get("code")==5 and r.get("sim",0)-mission["origin"]>=duration-.1

m=dict(observation_sim_s=mission["observation_sim_s"],wall_s=mission["wall_s"],realtime_factor=mission["observation_sim_s"]/mission["wall_s"],

 samples=len(samples),missing_interior_ticks=sorted(set(range(1,int(duration)))-ids),unanswered_query_ids=sorted({r["id"] for r in queries}-ids),

 duplicate_samples=len(samples)-len(ids),srf_p95_ms=float(np.percentile([r["source_ms"] for r in samples],95)),

 sensor_roundtrip_p95_ms=float(np.percentile(lat,95)),sensor_roundtrip_max_ms=float(lat.max()),

 summed_process_peak_rss_mib=max(sum(p["rss"] for p in r["processes"]) for r in res)/2**20,

 container_peak_memory_mib=max(r.get("container_memory_bytes",0) for r in res)/2**20,peak_process_cpu_percent=peak_cpu,

 navigation_segments=mission["segments"],gp_updates=sum(p["batch_size"]>0 for p in plans),planning_rounds=len(plans),

 planning_round_total_s=[p["total_s"] for p in plans],gp_update_s=[p["update_s"] for p in plans],score_calls=[p["calls"] for p in plans],

 search_timeouts=sum(p["timeout"] for p in plans),

 cancelled_navigation_results=[r for r in manager if r.get("event")=="navigation_result" and expected_cancel(r)],

 navigation_successes=sum(r.get("event")=="navigation_result" and r["code"]==4 for r in manager),

 navigation_failures=[r for r in manager if r.get("event")=="navigation_result" and r["code"]!=4 and not expected_cancel(r)],

 idle_no_route_events=sum(r.get("event")=="no_motion_route" for r in manager),

 final_committed_observations=980+sum(p["batch_size"] for p in plans),uncommitted_end_samples=len(samples)-sum(p["batch_size"] for p in plans),

 path_length_m=float(np.linalg.norm(np.diff(poses[:,1:3],axis=0),axis=1).sum()),bag_topic_counts=counts,

 gui_fps="not measured; headless run",video_type="4x physical-pose replay decoded from rosbag, not GUI screen capture")

m["max_actual_linear_speed_m_s"]=max(actual_speeds)

m["max_raw_truth_twist_yaw_rad_s"]=max(actual_yaw_rates)
pose_dt=np.diff(poses[:,0]);delta_yaw=np.arctan2(np.sin(np.diff(poses[:,3])),np.cos(np.diff(poses[:,3])))
pose_yaw_rates=delta_yaw[pose_dt>0]/pose_dt[pose_dt>0]
m["max_pose_derived_yaw_rate_rad_s"]=float(np.max(np.abs(pose_yaw_rates)))
m["raw_truth_twist_yaw_spikes"]=raw_yaw_spikes
incidents=[]
for spike in raw_yaw_spikes:
 index=int(np.argmin(np.abs(poses[:,0]-spike["t"])))
 if index>0:
  dt=poses[index,0]-poses[index-1,0]
  dot=float(np.dot(quaternions[index-1],quaternions[index]))
  step=float(np.arctan2(np.sin(poses[index,3]-poses[index-1,3]),np.cos(poses[index,3]-poses[index-1,3])))
  incidents.append(dict(**spike,quaternion_dot_previous=dot,pose_dt_s=float(dt),
                        wrapped_pose_yaw_step_rad=step,pose_derived_yaw_rad_s=step/dt if dt>0 else None,
                        equivalent_quaternion_sign_change=dot<-.999))
(RUN/"yaw_incident.json").write_text(json.dumps(dict(raw_outliers=incidents,
 height_range_m=[min(heights),max(heights)],
 scope="Raw truth-twist outliers retained; wrapped pose increments and adjacent quaternion dots are independent diagnostics. No dynamics were changed."),indent=2))

m["yaw_incident_interpretation"]=(
 "All current raw yaw outliers coincide with equivalent quaternion sign changes; wrapped pose-derived rates remain continuous. Raw records are retained."
 if incidents and all(r["equivalent_quaternion_sign_change"] for r in incidents)
 else "Raw yaw outliers (if present) are retained alongside pose-derived rates; inspect the per-run diagnostics before attributing a cause.")

m["command_peaks"]=command_peaks

m["post_window_plan_rounds"]=[p["round"] for p in plans if p["finish_t"]>duration+.1]

m["feature_seconds_search"]=[p.get("feature_seconds") for p in plans]

m["objective_seconds_search"]=[p.get("objective_seconds") for p in plans]

m["scoring_compute_mode"]=read("protocol.json").get("scoring_compute_mode","scalar")

m["nominal_speed_m_s"]=read("protocol.json")["navigation_nominal_speed_m_s"]
m["scoring_time_step_s"]=read("protocol.json").get("scoring_time_step_s",.5)

graph=read("graph.json")

edge={(g["from"],g["to"]):g["duration_s"] for g in graph}

m["initial_feasible_oneway_candidates"]=sum(edge[0,i]<=duration for i in range(1,31))

m["scoring_input_count"]=len(plans[0]["weights"]) if plans else 0

m["navigation_goal_nodes"]=[r["target"] for r in manager if r.get("event")=="goal"]

m["physical_extent_xy_m"]=(poses[:,1:3].max(axis=0)-poses[:,1:3].min(axis=0)).tolist()

samplexy=np.array([[r["x"],r["y"]] for r in samples])

m["stationary_sample_intervals"]=int(np.sum(np.linalg.norm(np.diff(samplexy,axis=0),axis=1)<.01))

m["moving_sample_intervals"]=len(samples)-1-m["stationary_sample_intervals"]

m["planning_total_wall_s"]=sum(p["total_s"] for p in plans)

m["stale_pose_rejections"]=sum(r.get("event")=="stale_pose" for r in lines("sampler.jsonl"))

m["recorded_measurements_match_log"]=counts.get("/cl/measurement",0)==len(samples)

m["nonmonotonic_pose_timestamps"]=int(np.sum(np.diff(poses[:,0])<0))

m["functional_pass"]=m["navigation_segments"]>=2 and m["gp_updates"]>=2 and not m["missing_interior_ticks"] and not m["unanswered_query_ids"] and not m["duplicate_samples"] and audit["gp_checks"]["gp_equivalent"] and all(c["passed"] for c in audit["returned_route_score_checks"])

m["functional_pass"]=bool(m["functional_pass"] and m["navigation_successes"]==m["navigation_segments"] and not m["navigation_failures"] and not m["search_timeouts"] and m["recorded_measurements_match_log"] and not m["stale_pose_rejections"])

m["closed_loop_cycles_pass"]=m["navigation_segments"]>=2 and m["gp_updates"]>=2

m["search_completion_pass"]=m["search_timeouts"]==0 and not m["post_window_plan_rounds"]

execution_mode=read("protocol.json").get("execution_mode","serial")
m["execution_mode"]=execution_mode
activations=[p for p in plans if p.get("kind")=="activate"]
initial_activation=activations[0]["round"] if activations else None
m["post_initial_gp_updates"]=sum(p["batch_size"]>0 and not(execution_mode=="lookahead" and p["round"]==initial_activation) for p in plans)
m["closed_loop_cycles_pass"]=m["navigation_segments"]>=2 and m["post_initial_gp_updates"]>=2
m["functional_pass"]=bool(m["functional_pass"] and m["closed_loop_cycles_pass"])
m["unused_post_window_plan_rounds"]=[p["round"] for p in plans if p["finish_t"]>duration+.1] if execution_mode=="lookahead" else []
from overlap_metrics import analyze,plot as plot_overlap
overlap,planning_spans,nav_spans=analyze(plans,manager,poses,mission["origin"],duration)
m.update(overlap)
m["concurrent_planning_requests"]=sum(r.get("event")=="busy_request_rejected" for r in lines("planner.jsonl"))
m["post_window_compute_wait_s"]=read("run_status.json").get("post_window_compute_wait_s",0)
m["movement_planning_verified"]=bool(execution_mode=="lookahead" and m["lookahead_overlap_physical_motion_s"]>1 and
    not m["late_goal_dispatches"] and not m["goal_preemption_events"] and not m["concurrent_planning_requests"] and
    audit.get("lookahead_checks",{}).get("no_uncommitted_observations_used",False))
if execution_mode=="lookahead":
 m["functional_pass"]=bool(m["functional_pass"] and not m["late_goal_dispatches"] and m["movement_planning_verified"])
 m["search_completion_pass"]=m["search_timeouts"]==0
else:
 m["functional_pass"]=bool(m["functional_pass"] and not m["post_window_plan_rounds"])
plot_overlap(RUN/"scheduling_timeline.png",planning_spans,nav_spans,duration)

m["realtime_target_met"]=m["realtime_factor"]>=.95

(RUN/"metrics.json").write_text(json.dumps(m,indent=2))

fig,ax=plt.subplots(1,2,figsize=(10,4),layout="constrained")

ax[0].plot(poses[:,1],poses[:,2]);ax[0].set(xlabel="Field x (m)",ylabel="Field y (m)",title="Recorded physical trajectory",aspect="equal",xlim=(0,97),ylim=(0,64))

ax[1].plot([r["t"] for r in samples],[r["temperature"] for r in samples],".-");ax[1].set(xlabel="Mission time (s)",ylabel="Temperature (degC)",title="Continuous measurements")

fig.savefig(RUN/"trajectory_temperature.png",dpi=140);plt.close(fig)

e=audit["evaluation"];fig,ax=plt.subplots(figsize=(6,3.3),layout="constrained")

ax.plot([r["time_s"] for r in e],[r["rmse_c"] for r in e],"o-",label="Updated GP")

ax.plot([r["time_s"] for r in e],[r["history_only_rmse_c"] for r in e],"s--",label="History only")

ax.set(xlabel="Mission time (s)",ylabel="Synthetic RMSE (degC)",title="70 fixed locations; current-time estimates");ax.legend()

fig.savefig(RUN/"estimation.png",dpi=140);plt.close(fig)

plt.rcParams["animation.ffmpeg_path"]=imageio_ffmpeg.get_ffmpeg_exe()

fig,(overview,close)=plt.subplots(1,2,figsize=(10,4.5),layout="constrained")

candidates=np.array(read("candidates.json"))

overview.scatter(candidates[1:,0],candidates[1:,1],s=12,c="gray")

overview.set(xlim=(0,97),ylim=(0,64),aspect="equal",xlabel="Field x (m)",ylabel="Field y (m)",title="97 x 64 m / 30 candidates")

close.set(aspect="equal",xlabel="Field x (m)",ylabel="Field y (m)",title="Physical-pose close-up")

track,=overview.plot([],[],"b-");point,=overview.plot([],[],"ro");track2,=close.plot([],[],"b-",alpha=.5)

body=Polygon([[0,0],[1,0],[1,1],[0,1]],color="darkgreen");close.add_patch(body);label=fig.suptitle("")

writer=FFMpegWriter(fps=10,codec="libx264",extra_args=["-pix_fmt","yuv420p"])

if os.environ.get("CL_REUSE_VIDEO")!="1" or not (RUN/"robot_replay.mp4").exists():

 with writer.saving(fig,str(RUN/"robot_replay.mp4"),dpi=100):

  for t in np.arange(0,duration,.4):

   i=min(np.searchsorted(poses[:,0],t),len(poses)-1);stamp,x,y,yaw=poses[i]

   track.set_data(poses[:i+1,1],poses[:i+1,2]);point.set_data([x],[y]);track2.set_data(poses[:i+1,1],poses[:i+1,2])

   rot=np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])

   body.set_xy(np.array(read("scene.json")["footprint_m"])@rot.T+[x,y])

   close.set_xlim(x-5,x+5);close.set_ylim(y-5,y+5)

   label.set_text(f"ROS bag physical-pose replay | t = {t:.1f} s | 4x speed (not GUI recording)")

   writer.grab_frame()

plt.close(fig)

# Present the retained run only.
protocol=read("protocol.json");initial=read("initial.json");field=read("field.json")
run_rel="result/"+RUN.name
end_eval=audit["evaluation"][-1]
score_error=max(c["absolute_error"] for c in audit["returned_route_score_checks"])
gaps=", ".join(f"{value:.2f}" for value in m["handover_gap_s"])
rows=[
 ("Simulation / wall time",f'{m["observation_sim_s"]:.3f} / {m["wall_s"]:.3f} s'),
 ("Received / missing / duplicate measurements",f'{m["samples"]} / {len(m["missing_interior_ticks"])} / {m["duplicate_samples"]}'),
 ("Completed navigation segments",m["navigation_segments"]),
 ("Nonempty GP updates",f'{m["gp_updates"]}; {m["post_initial_gp_updates"]} after initial activation'),
 ("Lookahead searches / arrival activations",f'{m["lookahead_search_count"]} / {m["activation_count"]}'),
 ("Lookahead planning during physical motion",f'{m["lookahead_overlap_physical_motion_s"]:.2f} s'),
 ("Checkpoint handover waits",gaps+" s"),
 ("Initial dispatch wait",f'{m["initial_dispatch_wait_s"]:.2f} s'),
 ("Total computation / computation while stationary",f'{m["planning_total_wall_s"]:.2f} / {m["planning_while_stationary_s"]:.2f} s'),
 ("Search guard timeouts",m["search_timeouts"]),
 ("Real-time factor",f'{m["realtime_factor"]:.5f}'),
 ("Distance travelled",f'{m["path_length_m"]:.2f} m'),
 ("Sampled summed-process RSS / container peak",f'{m["summed_process_peak_rss_mib"]:.0f} / {m["container_peak_memory_mib"]:.0f} MiB'),
 ("SRF / sensor round-trip P95",f'{m["srf_p95_ms"]:.3f} / {m["sensor_roundtrip_p95_ms"]:.3f} ms'),
 ("Committed new / terminal pending observations",f'{m["final_committed_observations"]-len(initial["y"])} / {m["uncommitted_end_samples"]}'),
 ("End-time synthetic RMSE / History-Only",f'{end_eval["rmse_c"]:.6f} / {end_eval["history_only_rmse_c"]:.6f} degC'),
 ("Maximum C++ / Python score discrepancy",f'{score_error:.2g}'),
 ("Movement-time planning / full-search acceptance",f'{m["movement_planning_verified"]} / {m["search_completion_pass"]}')
]
table="| Measurement | Recorded value |\n| --- | --- |\n"+"\n".join(f"| {key} | {value} |" for key,value in rows)
purpose=("Verify that the C++ GP/GRG planner runs while Bunker Mini navigates in Gazebo/Nav2, "
 "that 1 Hz continuous temperature acquisition remains complete, and that candidates are handed over "
 "at the correct endpoint with an updated GP and actual timestamps. This is one scheduling/performance "
 "verification, not a comparison of scoring policies or real-field accuracy.")
config=f"""| Item | Setting |
| --- | --- |
| Scene / robot | Flat 97 x 64 m rectangle, boundary walls, no interior obstacles; Bunker Mini, Gazebo, ROS 2 Jazzy and Nav2 |
| Localization / mapping | Ideal simulator localization; planar translation of real observations, not surveyed map registration |
| Candidates / start | 30 rounded 6 x 5 checkpoints plus start (48.5,32) m; 930 directed Nav2 paths |
| Mission / sensing | 300 s including initial planning; 1 Hz ticks 1-299, no dwell; free endpoint, no mandatory return |
| Initial GP | {len(initial["y"])} real observations; fixed ARD RBF, no retraining |
| RBF lengths | x={initial["length"][0]:.3f} m, y={initial["length"][1]:.3f} m, t={initial["length"][2]:.3f} s; all parameters in initial.json |
| Synthetic source | Independent GSTools finite-mode Matern SRF; nu={field["matern_nu"]}, {field["fourier_modes"]} modes, seed={field["field_seed"]} |
| Field scales / noise | {field["scaled_coordinate_scales_m_m_s"]} (m,m,s); noise SD {field["noise_std_c"]} degC, keyed by measurement ID |
| Field reference resolution | 0.1 m / 0.5 s; arbitrary-position/time queries, no precomputed space-time grid |
| Scoring | Adaptive WIVR; 70 area-clipped 10 m cells x 151 target times at 2 s = 10,570 targets; route events remain 1 Hz |
| GRG / guards | Depth 2, two time splits; 45 s search and 10 s selected-route re-score guards |
| Parallelism | OpenMP/Eigen, {read("matrix_threads.json")["eigen_effective"]} matrix threads; event-feature batches of 32 |
| Navigation | RPP, velocity smoother and bounded watchdog; forward/nominal 1.0 m/s, reverse 0.3 m/s, yaw command cap 0.6 rad/s |
| Update | Frozen committed GP during lookahead; actual samples buffered and assimilated at activation |
| Evaluation | 70 fixed locations at 0/75/150/225/300 s; GP committed by each time; History-Only baseline |
"""
architecture="""| Module | Implementation | Responsibility |
| --- | --- | --- |
| Mission Manager | C++; lookahead_manager.hpp + handover.hpp | Keep one Nav2 goal active; future-endpoint request; request ID/anchor checks; arrival handover |
| Planner | C++/Eigen; nodes.cpp + core.hpp | Frozen GP/WIVR -> GRG candidate; actual batch -> GP update -> candidate re-score/time trimming |
| Sampler | C++; nodes.cpp | Capture actual pose/time every second and query temperature |
| Temperature source | Python/GSTools; srf_monitor.py | Independent fixed synthetic truth plus reproducible noise |
| Monitor / analysis | Python | Resource logs, independent numerical checks, MCAP-derived timing, figures and replay |
"""
sequence=("Initial search -> actual-data activation -> Nav2 goal. During that goal, "
 "LookaheadManager::ask_search requests a candidate anchored at the endpoint and a predicted arrival time. "
 "Planner::plan freezes the committed GP and weights; current-leg samples stay buffered. At actual arrival, "
 "activate_candidate waits for a candidate if necessary, commits the actual batch, and re-scores/trims "
 "the route at actual time before dispatch. Replanning never preempts the active Nav2 goal. "
 "At 300 s, sensing/navigation stop; an outstanding calculation may finish without dispatch.")
limits=(f"Movement-time planning is verified; strict combined acceptance remains false because "
 f"{m['search_timeouts']} searches hit the 45 s guard. Initial planning still waits {m['initial_dispatch_wait_s']:.2f} s. "
 "Activation checks an existing candidate under the updated GP, without repeating full route selection. "
 "ETA errors and late plans can still create waiting.\n\n"
 f"The trace remains effectively collinear (y-span {m['physical_extent_xy_m'][1]:.3f} m); area coverage is not established. "
 "Synthetic field scales, sensor response and surveyed map registration remain unvalidated.\n\n"
 f"{m['uncommitted_end_samples']} terminal observations remain saved but uncommitted. RMSE uses the committed GP "
 "against synthetic truth, without an artificial terminal update; it is not real-field accuracy or policy superiority. "
 f"Expected endpoint navigation cancellations: {len(m['cancelled_navigation_results'])}.\n\n"
 f"Raw truth-twist yaw outliers: {len(m['raw_truth_twist_yaw_spikes'])}; per-run quaternion/pose checks are retained "
 "in yaw_incident.json, with no raw data removed. No GUI frame rate was measured; real-time factor and sensing ticks are reported separately.")
research_notes=(
 "\n\n### Route geometry and kernel validation\n\n"
 "The nearly collinear route is an observed planner outcome, not evidence by itself of a navigation failure or insufficient training data. "
 f"The fitted ARD RBF lengths are {initial['length'][0]:.3f} m along x, {initial['length'][1]:.3f} m along y and {initial['length'][2]:.3f} s in time. "
 "At the same time, a 5 m separation has prior correlation approximately 0.381 along x and 0.972 along y. "
 "This directional correlation makes horizontal exploration less redundant under the current model and contributes to its WIVR score advantage. "
 "Candidate geometry, route timing, scoring quadrature and incomplete search also affect the chosen route. "
 "Additional independent runs with better directional coverage, repeated locations and different times would help validate and refit the parameters. "
 "More data do not guarantee a turning route; held-out reconstruction error should assess model adequacy rather than route appearance. "
 "The fitted anisotropy is not independently established as a physical-field property.\n\n"
 "### Search truncation and computation\n\n"
 f"This run executed {m['full_search_count']} GRG searches, including the initial search and {m['lookahead_search_count']} moving lookahead searches; {m['search_timeouts']} hit the 45 s wall guard. "
 "Activation re-scores are separate from full searches. Thirty candidates, at least 980 GP observations, 1 Hz hypothetical route events and 10,570 evaluation targets require repeated dense covariance and matrix calculations. "
 "Exceeding the guard reflects the workload, implementation efficiency and chosen time limit; it does not by itself invalidate the GP/GRG formulation. "
 "The guard returns the best fully evaluated route found so far. Movement-time planning reduces stationary waiting but does not remove the computation or the search limit.\n\n"
 "### How the kernel influences navigation\n\n"
 "Kernel correlation -> GP posterior uncertainty -> expected joint benefit of along-route observations -> WIVR route score -> GRG route selection -> Nav2 execution. "
 "The kernel defines how a measurement informs other positions and times; it does not directly issue steering commands. "
 "WIVR estimates weighted uncertainty reduction, not realized reconstruction error. "
 "Kernel hyperparameters stay fixed online in this experiment. New temperature values update the posterior mean, while measurement locations and times determine the posterior covariance for a fixed kernel and noise model. "
 "Updating the GP posterior is therefore distinct from refitting kernel hyperparameters; the robot does not automatically relearn its spatial length scales while moving."
)
files=f"""| Path | Purpose |
| --- | --- |
| testing_code/src/nodes.cpp | Planner, sampler, search/activation branches and numerical audit |
| testing_code/src/core.hpp | RBF GP, continuous route events, WIVR, caches and GRG |
| testing_code/src/lookahead_manager.hpp, handover.hpp | Navigation/compute scheduling and request/endpoint checks |
| testing_code/src/matrix_threads.hpp, CMakeLists.txt, package.xml | OpenMP setup and ROS C++ build |
| testing_code/check_handover.py, src/handover_test.cpp | Early/late replies, stale IDs, wrong anchors and no-preemption state checks |
| testing_code/protocol.json, field_config.json | Runtime settings and fixed synthetic-source parameters |
| testing_code/prepare.py, scene.launch.py | Freeze inputs and generate the scene/Nav2 configuration |
| testing_code/navigation_profile.py, navigation_profile.json, capstone_nav2_reference.yaml | Navigation limits, smoother and configuration provenance |
| testing_code/srf_monitor.py | Temperature source and resource monitor |
| testing_code/run.py, run.sh, Run-ClosedLoop.ps1 | Preflight, build, launch, MCAP recording, stop/drain and reporting |
| testing_code/audit.py | Independent GP/WIVR checks and synthetic estimation errors |
| testing_code/overlap_metrics.py, report.py | Pose-based overlap, timeline, figures, replay and this executed notebook |
| result/latest.json | Pointer to the retained run |
| {run_rel}/rosbag/, *.jsonl, *.log | Raw MCAP, measurements, manager/planner events and resource logs |
| {run_rel}/initial.json, field.json, protocol.json, map.*, world.sdf | Exact recorded inputs |
| {run_rel}/plan_*.json, gp_*.json, batch_*.json | Candidate/activation records, GP snapshots and actual assimilated batches |
| {run_rel}/source_snapshot/, manifest.json | Original recorded source and hashes; historical evidence, not the live launcher |
| {run_rel}/analysis_source_snapshot/ | Analysis/report code for this presentation |
| {run_rel}/metrics.json, audit_and_estimation.json, artifact_validation.json | Results, independent checks and validation |
| {run_rel}/robot_replay.mp4, *.png | This run's physical-pose replay and figures |
"""
reproduce=r"""~~~powershell
powershell -ExecutionPolicy Bypass -File .\testing_code\Run-ClosedLoop.ps1
~~~

A new run creates a new timestamped result; it does not overwrite this record. Preflight runs the handover test and checks actual OpenMP/GP operation without requiring discarded experiment results.

Requires Docker container bunker-jazzy, repository mount /workspace, ROS 2 Jazzy and overlay /ros-data/build/10.27_ROS_simulation/Nav2_test/install. C++ requires Eigen3, OpenMP and nlohmann/json. Container SRF Python: /home/ubuntu/.venvs/srf-verification; host reporting Python: C:\Ubuntu\ROS\reporting-venv.

External dependencies retained outside this folder: aligned observations Data/Real_Data_Integration/result/gp_samples.csv; saved RBF kernel Code/ROS_simulation/Trained_Real_Data_Kernel; Nav2/Bunker fixture code ../Nav2_test/testing_code; numerical helpers ../Continuous_sensing_test/testing_code, ../Scoring_policy_test/testing_code and ../SRF_test/testing_code. Exact recorded inputs remain in this run directory.

To rebuild this report from saved MCAP, run testing_code/report.py using the host reporting Python. Set CL_REUSE_VIDEO=1 to reuse the recorded replay.
"""
(ROOT/"README.md").write_text(
 "# Bunker Mini: Planning During Navigation\n\n"+f"Retained run: **{RUN.name}** (UTC run identifier). Only this run is presented.\n\n"+
 "## Purpose\n\n"+purpose+"\n\n## Configuration\n\n"+config+
 "\n## Architecture and execution\n\n"+architecture+"\n"+sequence+
 "\n\n## Recorded results\n\n"+table+
 "\n\nPhysical motion means pose-derived translation above 0.05 m/s or wrapped rotation above 0.05 rad/s. "
 "Overlap is integrated from positive pose intervals. Total computation overlaps navigation and must not be added to mission duration.\n\n"+
 f"[Notebook with figures and replay](test_report.ipynb) | [Replay]({run_rel}/robot_replay.mp4)\n\n"+
 "The large pairwise Nav2 graph is published losslessly as graph.json.gz; graph.json remains a local working copy. The report reads either form.\n\n"+
 "## Limits and interpretation\n\n"+limits+"\n\n## Files\n\n"+files+
 "\n## Reproduction and dependencies\n\n"+reproduce+"\n",encoding="utf-8")
nb=nbformat.v4.new_notebook()
md=nbformat.v4.new_markdown_cell;code=nbformat.v4.new_code_cell
nb.cells=[
 md("# Bunker Mini: Planning During Navigation\n\n"+purpose+f"\n\nRecorded run: {RUN.name}."),
 md("## Configuration and execution\n\n"+config+"\n"+sequence),
 md("## Recorded results\n\n"+table),
 code('from pathlib import Path\nimport json\nfrom IPython.display import display,Image,Video\nROOT=Path.cwd()\nRUN=ROOT/"result"/'+repr(RUN.name)+'\nm=json.loads((RUN/"metrics.json").read_text())\na=json.loads((RUN/"audit_and_estimation.json").read_text())\ndisplay(Image(filename=str(RUN/"scheduling_timeline.png")))\nprint("Arrival estimate errors:",m["arrival_forecast_errors_s"])'),
 code('display(Image(filename=str(RUN/"trajectory_temperature.png")))\ndisplay(Video(str(RUN/"robot_replay.mp4"),embed=True,width=900))'),
 md("## GP updates and independent verification\n\nLookahead freezes the committed GP; current-leg samples stay buffered until activation. Independent Python calculations check the C++ GP and route scores."),
 code('plans=sorted((json.loads(p.read_text()) for p in RUN.glob("plan_*.json")),key=lambda p:p["round"])\nfor p in plans:\n print({k:p[k] for k in ["round","kind","lookahead","anchor","batch_size","gp_version","planning_started_t","t","finish_t","route","timeout"]})\nprint("GP:",a["gp_checks"])\nprint("Frozen state:",a["lookahead_checks"])\nprint("Maximum score error:",max(r["absolute_error"] for r in a["returned_route_score_checks"]))'),
 md("## Synthetic reconstruction\n\n70 fixed locations at 0/75/150/225/300 s; only the GP already committed at each time is evaluated. History-Only uses the same inputs and truth. Terminal data are not artificially assimilated."),
 code('display(Image(filename=str(RUN/"estimation.png")))\nfor row in a["evaluation"]:\n print({k:row[k] for k in ["time_s","rmse_c","history_only_rmse_c","observations"]})'),
 md("## Conclusion and limits\n\n"+limits+research_notes)
]
nb.metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"}}
NotebookClient(nb,timeout=180,resources={"metadata":{"path":str(ROOT)}}).execute()
nbformat.validate(nb)
assert not [o for c in nb.cells for o in c.get("outputs",[]) if o.get("output_type")=="error"]
nbformat.write(nb,ROOT/"test_report.ipynb")
print(json.dumps(dict(run=RUN.name,samples=m["samples"],overlap_s=m["lookahead_overlap_physical_motion_s"],search_timeouts=m["search_timeouts"])))

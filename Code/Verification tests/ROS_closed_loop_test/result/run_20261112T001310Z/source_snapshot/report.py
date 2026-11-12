"""Generate an executed notebook and physical-pose rosbag replay."""

from pathlib import Path

import json,math,os,re

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

def read(n):return json.loads((RUN/n).read_text())

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

m["yaw_incident_interpretation"]="Raw truth-twist outliers (if present) are retained alongside wrapped pose-derived rates. Quaternion sign-change evidence was established for the archived 230921 run; do not assume it for an uninspected new outlier."

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

m["functional_pass"]=bool(m["functional_pass"] and not m["post_window_plan_rounds"])

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



nb=nbformat.v4.new_notebook()

md=nbformat.v4.new_markdown_cell;code=nbformat.v4.new_code_cell

history=[("120 s / scalar / 0.25 m/s","run_20261111T205538Z"),

         ("300 s / scalar / 0.25 m/s","run_20261111T212025Z"),

         ("300 s / batched / 0.5 s grid / 1.0 m/s","run_20261111T223631Z"),
         ("300 s / 2 s grid / Eigen serial","run_20261111T230921Z")]
if RUN.name not in [name for _,name in history]:
 history.append(("300 s / 2 s grid / OpenMP",RUN.name))

records=[]

for label,name in history:

 r=ROOT/"result"/name

 if r.exists() and name not in [x["run"] for x in records]:

  records.append(dict(label=label,run=name,metrics=json.loads((r/"metrics.json").read_text()),

      evaluation=json.loads((r/"audit_and_estimation.json").read_text())["evaluation"],
      protocol=json.loads((r/"protocol.json").read_text()) if (r/"protocol.json").exists() else {}))

def row(label,fn):

 return "| "+label+" | "+" | ".join(str(fn(x)) for x in records)+" |"

headers="| Measured item | "+" | ".join(x["label"] for x in records)+" |\n| --- | "+" | ".join("---" for _ in records)+" |"

table="\n".join([headers,

 row("Run",lambda x:x["run"]),
 row("Eigen threads (recorded runtime)",lambda x:json.loads((ROOT/"result"/x["run"]/"matrix_threads.json").read_text())["eigen_effective"] if (ROOT/"result"/x["run"]/"matrix_threads.json").exists() else "1 (prior serial build)"),
 row("Scoring target dt (s)",lambda x:x["protocol"].get("scoring_time_step_s",.5)),

 row("Observed simulation / wall time (s)",lambda x:f'{x["metrics"]["observation_sim_s"]:.3f} / {x["metrics"]["wall_s"]:.3f}'),

 row("Collected / missed ticks",lambda x:f'{x["metrics"]["samples"]} / {len(x["metrics"]["missing_interior_ticks"])}'),

 row("Completed segments / nonempty GP updates",lambda x:f'{x["metrics"]["navigation_segments"]} / {x["metrics"]["gp_updates"]}'),

 row("Planning rounds (s)",lambda x:", ".join(f"{v:.2f}" for v in x["metrics"]["planning_round_total_s"])),

 row("Planning total (s)",lambda x:f'{sum(x["metrics"]["planning_round_total_s"]):.2f}'),

 row("Search guard timeouts",lambda x:x["metrics"]["search_timeouts"]),
 row("Noncached scoring calls",lambda x:", ".join(str(v) for v in x["metrics"]["score_calls"])),

 row("RTF",lambda x:f'{x["metrics"]["realtime_factor"]:.3f}'),

 row("Sampled peak summed RSS / container (MiB)",lambda x:f'{x["metrics"]["summed_process_peak_rss_mib"]:.0f} / {x["metrics"]["container_peak_memory_mib"]:.0f}'),

 row("SRF / sensor roundtrip P95 (ms)",lambda x:f'{x["metrics"]["srf_p95_ms"]:.3f} / {x["metrics"]["sensor_roundtrip_p95_ms"]:.3f}'),

 row("Distance (m)",lambda x:f'{x["metrics"]["path_length_m"]:.2f}'),

 row("Committed new / pending samples",lambda x:f'{x["metrics"]["final_committed_observations"]-980} / {x["metrics"]["uncommitted_end_samples"]}'),

 row("End-time RMSE / History-Only (degC)",lambda x:f'{x["evaluation"][-1]["rmse_c"]:.5f} / {x["evaluation"][-1]["history_only_rmse_c"]:.5f}'),

 row("Strict combined check",lambda x:x["metrics"]["functional_pass"])])

comparison_notes=(

 "\n\nThe two scalar runs used the earlier 0.25 m/s RPP fixture. The latest uses the separately verified 1.0 m/s manufacturer-capped RPP/smoother profile as well as batched scoring. All retain 30 candidates, GRG depth 2 / two splits, 1 Hz sensing, the same 980 initial real observations, saved RBF kernel and fixed SRF realization. The 120 s run required return Home and used 16,870 scoring inputs; subsequent 300 s runs have free endpoints; scoring time steps are shown per run from their saved protocol. The SRF's 0.1 m / 0.5 s reference settings are unchanged."

 "\n\nAcross the early runs, navigation speed/profile and available routes also changed. The archived batched 0.5 s versus serial 2 s runs retain the same navigation profile and differ in evaluation-grid density; coarser quadrature can change scores, selected routes and subsequent data. These are single descriptive traces, not repeated statistical trials. Faster scoring can also change which routes are explored before the unchanged 45 s guard. Fixed-workload old/new scoring checks below isolate the batching change. RMSE values at 120 and 300 s refer to different times and do not rank reconstruction methods."

 "\n\nThe original 120 s run verified a small two-candidate return workload; the scalar 300 s run exposed long planning delays and remained mainly collinear. Interpret latest route coverage and performance from its measured trace. Guarded best-so-far searches are not claimed to be complete searches. Expected cancellation at the observation endpoint is distinct from a navigation failure. Pending end samples remain saved; no forced terminal GP update is added.")

batch=read("batch_validation.json") if (RUN/"batch_validation.json").exists() else None

batch_text=""

if batch:

 batch_text=(f'\n\n**Controlled scoring check:** {batch["cases_per_trial"]} saved workload cases, scalar/batched/batched/scalar order; identical GP snapshots, event positions/times, weights, grid and noise. Maximum absolute WIVR difference {batch["max_score_absolute_error"]:.3g}. Median total cold score time {batch["median_total_cold_score_s"]["scalar"]:.3f} → {batch["median_total_cold_score_s"]["batched"]:.3f} s ({batch["cold_score_speedup"]:.2f}x). This excludes GP/query-grid setup and is not whole-GRG speedup. Cache-hit and shared-prefix scores are also checked.')

nb.cells=[

 md("# ROS Closed-Loop Verification: Recorded Runs and Batched Scoring\n\n"+table+comparison_notes+batch_text),

 code('from pathlib import Path\nimport json\nfrom IPython.display import display,Markdown,Image,Video\nROOT=Path.cwd()\nRUN=ROOT/"result"/json.loads((ROOT/"result/latest.json").read_text())["run"]\nm=json.loads((RUN/"metrics.json").read_text())\na=json.loads((RUN/"audit_and_estimation.json").read_text())\nprint(json.dumps(m,indent=2))'),

 md("## Implementation and calculation\n\nOnly uncached event-feature calculations are batched in blocks of 32 columns. With n real/committed observations, m hypothetical events and q scoring inputs: F_A = L^-1 K(X,A), shape n x m; C = K(Q,A) - F_Q^T F_A, shape q x m. The joint noisy event covariance, Cholesky solve, normalized WIVR weights and recursive GRG are unchanged. Cached columns are copied before any eviction; weights and GP stay frozen within a search. Batch memory is bounded, while the existing full event cross-covariance and query features remain.\n\nManager::ask → Planner::plan → GP::append → Search::events/score/recurse → next node → Nav2. Sampling is independent during both movement and planning. Segment success seals the next actual-observation batch. Expected events never become training data. The selected route is rescored after computation; trailing segments are trimmed if they no longer fit before the endpoint."),

 code('plans=sorted((json.loads(p.read_text()) for p in RUN.glob("plan_*.json")),key=lambda p:p["round"])\nfor p in plans:\n print({k:p.get(k) for k in ["round","batch_size","gp_version","search_route","route","t","finish_t","calls","cache_hits","feature_batches","feature_misses","feature_seconds","objective_seconds","update_s","total_s","timeout"]})\nprint("First returned event shape:",(len(plans[0]["events"]),3),"first events:",plans[0]["events"][:3])\nprint("Batch equivalence:",json.loads((RUN/"batch_validation.json").read_text()))\nprint("Independent Python checks:",a["gp_checks"],a["returned_route_score_checks"])'),

 md("## Physical-pose recordings\n\nEach 4x animation is decoded from its own rosbag/Gazebo physical poses, with a full-field view and robot close-up. These are not GUI recordings; GUI FPS was not measured.")

]

for rec in records:

 nb.cells.append(md("### "+rec["label"]+"\n\n"+rec["run"]))

 nb.cells.append(code('D=ROOT/"result"/'+repr(rec["run"])+'\ndisplay(Video(str(D/"robot_replay.mp4"),embed=True,width=900))'))

nb.cells += [

 code('display(Image(filename=str(RUN/"trajectory_temperature.png")))'),

 md("## Synthetic temperature estimates\n\n70 fixed spatial cells at 0/75/150/225/300 s for 300 s runs; only the GP committed by each time is used. Compare History-Only at identical inputs. Pre-assimilation errors use the GP available before each measurement, against noise-free SRF truth and noisy measurements separately. These are current-field estimates; neither a policy ranking nor real-field accuracy."),

 code('display(Image(filename=str(RUN/"estimation.png")))\nprint(json.dumps(a["evaluation"],indent=2))\nprint("Pre-assimilation:",a.get("pre_assimilation"))'),

 md("## Interpretation and limits\n\nThe archived 230921 truth-twist stream contains two extreme yaw-rate samples associated with equivalent quaternion sign changes, while pose differences remain continuous. Each run lists its own raw and pose-derived rates without removing raw samples. New outliers require their own inspection; archived evidence does not automatically explain them.\n\nInspect strict combined, repeated-cycle, real-time and search-completion flags separately. Long searches can keep the robot stationary even when no sensing ticks are missed. A 45 s guard checked between objectives may overshoot; its returned route is best-so-far. At most the next segment is executed before replanning, so a proposed full route is not an execution promise. Nominal path-length/speed timing omits explicit turns, acceleration and dispatch delays. Temporary map alignment, field calibration and the approximate Bunker contact dynamics remain provisional. Sampled memory covers startup/shutdown as well as online work.")

]

check_pointer=ROOT/"result/nav2_profile_latest.json"

if check_pointer.exists():

 nb.cells.append(md("## Navigation profile evidence\n\nThe 1.0 m/s cap follows the [current manufacturer listing](https://global.agilex.ai/products/bunker-mini-tracked-mobile-robot). Angular manufacturer maximum is unconfirmed; the previous 0.6 rad/s command cap remains. RPP is retained; Capstone smoothing/limit settings were adapted. The isolated two-leg check is separate from GP/GRG performance. Raw RPP angular requests are limited downstream; actual yaw in approximate contact dynamics can differ slightly."))

 nb.cells.append(code('D=ROOT/"result"/json.loads((ROOT/"result/nav2_profile_latest.json").read_text())["run"]\nv=json.loads((D/"verification.json").read_text())\nprint({k:v.get(k) for k in ["passed","goals","max_actual_speed_m_s","max_actual_yaw_rate_rad_s","watchdog_limits","command_bound_scope"]})'))


grid_pointer=ROOT/"result/grid_check_latest.json"
if grid_pointer.exists():
 nb.cells.append(md("## Fixed-workload check: 2 s WIVR grid\n\nCurrent source sets the online target grid to 2 s (10,570 inputs). Real/expected sensing stays 1 Hz and SRF is unchanged. The ROS runs above retain the evaluation step recorded in their own protocol; the latest full run tests 2 s. Three fixed workloads match independent Python scoring. Small fixed-case score changes do not prove that every route ranking is preserved. Planning/movement overlap is not yet implemented."))
 nb.cells.append(code('D=ROOT/"result"/json.loads((ROOT/"result/grid_check_latest.json").read_text())["run"]\ng=json.loads((D/"summary.json").read_text())\nprint(json.dumps(g,indent=2))'))

matrix_text=""
if (RUN/"matrix_validation.json").exists():
 v=read("matrix_validation.json")
 proof=read("matrix_threads.json")
 compact={k:v[k] for k in ("passed","selected_threads","medians","max_score_absolute_error","scope")}
 matrix_text="\n\n**OpenMP implementation verification:** The prior build lacked OpenMP despite requesting two Eigen threads. Both C++ executables now link OpenMP, with configuration before GP construction, a 1-4 thread bound and dynamic teams disabled. Runtime proof: "+json.dumps(proof)+". Fixed-workload 1/2/4-thread scores are numerically checked. Full recursive search completion must be assessed separately."
 matrix_text+="\n\n| Threads | Query setup (s) | Cold scoring (s) | Setup + all score cases (s) |\n| --- | --- | --- | --- |\n"
 for threads,values in v["medians"].items():
  matrix_text+=f'| {threads} | {values["setup_s"]:.3f} | {values["cold_score_s"]:.3f} | {values["setup_s"]+values["all_score_s"]:.3f} |\n'
 matrix_text+="\nTwo repetitions per count on three saved workloads; target grid, sensing events and mathematical scores unchanged. Both scalar and batched code in the separate regression check are compiled with OpenMP; that regression timing is not an isolated measurement of enabling OpenMP."
 nb.cells.append(md("## Matrix parallelism verification"+matrix_text))
 nb.cells.append(code('v=json.loads((RUN/"matrix_validation.json").read_text())\nprint(json.dumps({k:v[k] for k in ["passed","selected_threads","medians","max_score_absolute_error","scope"]},indent=2))\nprint("Runtime:",json.loads((RUN/"matrix_threads.json").read_text()))'))
previous=json.loads((ROOT/"result/run_20261111T230921Z/metrics.json").read_text())
assessment=(f'Latest run: {m["samples"]} samples; {m["navigation_segments"]} completed segments and {m["gp_updates"]} nonempty GP updates. '
 f'Planning total {m["planning_total_wall_s"]:.2f} s versus {previous["planning_total_wall_s"]:.2f} s in the prior serial 2 s run. '
 f'Search timeouts: {m["search_timeouts"]}; strict combined acceptance: {m["functional_pass"]}. '
 f'Observed y-span {m["physical_extent_xy_m"][1]:.2f} m; RTF {m["realtime_factor"]:.3f}. '
 'This is one descriptive ROS trace. A fixed-workload matrix speedup is not proof of complete recursive search or policy superiority. '
 'All prior records are preserved; movement and planning remain sequential.')
if m["physical_extent_xy_m"][1]<.5:
 assessment+=" The measured trajectory remains effectively collinear; higher scoring throughput does not establish broader spatial coverage."
nb.cells.append(md("## Latest run assessment\n\n"+assessment))


diagnostic_text=""
diagnostic_pointer=ROOT/"result/route_diagnosis_latest.json"
if diagnostic_pointer.exists():
 diagnostic_run=ROOT/"result"/json.loads(diagnostic_pointer.read_text())["run"]
 diagnostic=json.loads((diagnostic_run/"geometry.json").read_text())
 verified=json.loads((diagnostic_run/"verification.json").read_text())
 complete=json.loads((diagnostic_run/"complete.json").read_text())["search"]
 lines_table=["| Matched route (180 observations) | Fitted RBF / 10 m cells | Fitted RBF / 5 m cells | Swapped axes / 10 m cells |",
              "| --- | --- | --- | --- |"]
 for route_name in ("straight","previous_bent","lower_bent","vertical"):
  line=[route_name]+[f'{verified["scores"][key][route_name]:.6f}' for key in ("fitted_kernel_10m","fitted_kernel_5m","axis_swapped_counterfactual_10m")]
  lines_table.append("| "+" | ".join(line)+" |")
 diagnostic_text=("## Offline diagnosis: route geometry and planning limits\n\n"
  "No new ROS run was performed for this diagnosis. The executed scheduler, 30 candidates, protocol, saved kernel and WELL-E logic document remain unchanged.\n\n"
  "**Facts:** The saved RBF length scales are x = 3.598 m, y = 21.094 m, t = 21.136 s. "
  "At equal time, a 5 m separation gives prior correlation 0.381 along x and 0.972 along y. "
  "All route checks use the same 980 real observations, initial time 0.132 s, noise, weighting rule and first 180 global 1 Hz ticks. "
  "Scores are normalized weighted variance reduction in degC squared.\n\n"+
  "\n".join(lines_table)+
  "\n\n**Interpretation:** The fitted directional correlation favors horizontal exploration; swapping only the spatial length scales for an untrained counterfactual reverses the horizontal/vertical preference. "
  "The model also treats sufficiently delayed revisits as informative: at the same position, a 90 s separation has prior temporal correlation about 0.000115. "
  "These are properties of the current fitted model, not independently established physical-field anisotropy. "
  "The 5 m diagnostic preserves the ranking of these four routes, but substantially changes the vertical score; no full quadrature-convergence claim is made. "
  "The production scoring grid and SRF resolution were not changed.\n\n"
  f'**Uncapped search:** The original 30-candidate, depth-2 GRG completed in {complete["complete_wall_s"]:.2f} wall seconds, with {complete["calls"]} noncached scores, timeout = {complete["timeout"]}, and route {complete["route"]}. '
  "That route is still on y = 32 m. This completed the configured GRG search, not a global exhaustive route optimization. "
  f'Independent Python error is {verified["max_cpp_python_error"]:.2g} for matched routes and {verified["complete_search_score_error"]:.2g} for the completed route. '
  "Offline time is held fixed during this benchmark; it is not a completed ROS mission.\n\n"
  "**Scheduler diagnosis (not implemented):** Current Manager requests a new plan only after Nav2 success. "
  "A background worker alone does not overlap movement and planning. "
  "A future implementation would request a plan from the active segment endpoint at a predicted arrival time, freeze its GP/weights, retain actual measurements separately, and validate/re-time the candidate after actual arrival and a GP update. "
  "Request IDs, endpoint checks and serialized goal handover are required. If search is not ready at arrival, waiting remains necessary. "
  "Initial planning cannot overlap movement before any usable route exists. No such scheduler is enabled or claimed validated here.\n\n"
  "**Next decision:** An approximately 248 s initial search leaves about 52 s in the current 300 s window. "
  "Reducing candidates or extending the experiment requires a separate configuration decision; neither has been applied.\n\n"
  "Reproduce with testing_code/diagnose_routes.py, then verify_route_diagnosis.py and report_diagnosis.py. "
  "Diagnostic inputs and outputs are in result/"+diagnostic_run.name+"/.\n")
 nb.cells.append(md(diagnostic_text))
 nb.cells.append(code('D=ROOT/"result"/'+repr(diagnostic_run.name)+'\ndisplay(Image(filename=str(D/"route_diagnosis.png")))\nprint(json.dumps(json.loads((D/"verification.json").read_text()),indent=2))'))
readme=ROOT/"README.md";text=readme.read_text(encoding="utf-8")

marker="## Recorded results"

start=next((text.index(x) for x in [marker,"## Both recorded results","## Results"] if x in text),None)

end=text.index("## Files",start)

media="\n\nVideos and figures for all recorded runs are embedded in [test_report.ipynb](test_report.ipynb). Raw inputs, source snapshots, structured logs and MCAP are preserved per run. Latest recording: [MP4](result/"+RUN.name+"/robot_replay.mp4), a 4x rosbag physical-pose replay, not a GUI capture.\n\n"

prefix=re.sub(r"Latest run: \*\*[^*]+\*\*", "Latest run: **"+RUN.name+"**",text[:start],count=1)
readme_text=prefix+marker+"\n\n"+table+comparison_notes+batch_text+matrix_text+"\n\n"+assessment+media+text[end:]
diagnostic_marker="## Offline diagnosis: route geometry and planning limits"
if diagnostic_marker in readme_text:
 readme_text=readme_text[:readme_text.index(diagnostic_marker)].rstrip()+"\n\n"
if diagnostic_text:
 readme_text=readme_text.rstrip()+"\n\n"+diagnostic_text
readme.write_text(readme_text,encoding="utf-8")

nb.metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"}}

NotebookClient(nb,timeout=180,resources={"metadata":{"path":str(ROOT)}}).execute()

nbformat.validate(nb);nbformat.write(nb,ROOT/"test_report.ipynb")

assert not [o for c in nb.cells for o in c.get("outputs",[]) if o.get("output_type")=="error"]

print(json.dumps(m),flush=True)


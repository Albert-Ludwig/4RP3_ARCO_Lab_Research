"""Run the C++ closed loop, independent SRF, Nav2 and rosbag; preserve failures."""
from pathlib import Path
import os,sys,json,time,subprocess,signal,hashlib
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]; CODE=ROOT/"testing_code"
OUT=ROOT/"result"/datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ");OUT.mkdir(parents=True)
protocol=json.loads((CODE/"protocol.json").read_text())
matrix_threads=str(protocol.get("matrix_threads",2))
os.environ.update(CL_MATRIX_THREADS=matrix_threads,OMP_DYNAMIC="FALSE",CL_OUTPUT=str(OUT),ROS_DOMAIN_ID="79",GZ_PARTITION="well_e_cpp_closed_loop",ROS_HOME="/ros-data",OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS=matrix_threads)
env=os.environ.copy()
(OUT/"manifest.json").write_text(json.dumps(dict(started=datetime.now(timezone.utc).isoformat(),sources={str(p.relative_to(CODE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in CODE.rglob("*") if p.is_file() and "__pycache__" not in p.parts}),indent=2))
import shutil
shutil.copytree(CODE,OUT/"source_snapshot",ignore=shutil.ignore_patterns("__pycache__"))
processes=[]
def start(name,args):
 log=(OUT/(name+".log")).open("w")
 p=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
 processes.append((name,p,log));return p
failure=None
drain_wait_s=0
try:
 check_root=ROOT/"result"/json.loads((ROOT/"result/batch_check_latest.json").read_text())["run"]
 check=json.loads((check_root/"summary.json").read_text())
 assert check["passed"] and check["source_sha256"]["testing_code/src/core.hpp"]==hashlib.sha256((CODE/"src/core.hpp").read_bytes()).hexdigest(), "Batch validation missing/stale"
 shutil.copy2(check_root/"summary.json",OUT/"batch_validation.json")
 matrix_root=ROOT/"result"/json.loads((ROOT/"result/matrix_threads_latest.json").read_text())["run"]
 matrix_check=json.loads((matrix_root/"summary.json").read_text())
 assert matrix_check["passed"] and matrix_check["selected_threads"]==int(matrix_threads)
 for relative,digest in matrix_check["source_sha256"].items():
  assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==digest, "Matrix check source stale"
 shutil.copy2(matrix_root/"summary.json",OUT/"matrix_validation.json")
 build=Path("/tmp/cl_build/well_e_closed_loop/CMakeFiles/closed_loop.dir")
 for filename in ("flags.make","link.txt"):
  shutil.copy2(build/filename,OUT/("build_"+filename))
 binary=Path("/tmp/cl_install/well_e_closed_loop/lib/well_e_closed_loop/closed_loop")
 (OUT/"binary_sha256.json").write_text(json.dumps(dict(path=str(binary),sha256=hashlib.sha256(binary.read_bytes()).hexdigest())))

 if protocol.get("execution_mode")=="lookahead":
  handover_root=ROOT/"result"/json.loads((ROOT/"result/handover_check_latest.json").read_text())["run"]
  handover_check=json.loads((handover_root/"summary.json").read_text())
  assert handover_check["passed"]
  for relative,digest in handover_check["source_sha256"].items():
   assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==digest
  shutil.copy2(handover_root/"summary.json",OUT/"handover_validation.json")
 subprocess.run(["/home/ubuntu/.venvs/srf-verification/bin/python",str(CODE/"prepare.py")],env=env,check=True)
 subprocess.run(["/tmp/cl_install/well_e_closed_loop/lib/well_e_closed_loop/closed_loop","audit"],env=env,check=True)
 subprocess.run(["/home/ubuntu/.venvs/srf-verification/bin/python",str(CODE/"audit.py"),str(OUT)],env=env,check=True)
 start("scene",["ros2","launch",str(CODE/"scene.launch.py")])
 time.sleep(8)
 start("bag",["ros2","bag","record","-s","mcap","-o",str(OUT/"rosbag"),"/clock","/simulation/ground_truth/odom","/odom","/tf","/tf_static","/cl/control","/cl/query","/cl/temperature","/cl/measurement","/cl/request","/cl/plan","/simulation/cmd_vel","/cl/cmd_vel_nav","/cmd_vel"])
 start("source",["/home/ubuntu/.venvs/srf-verification/bin/python",str(CODE/"srf_monitor.py")])
 start("monitor",["/home/ubuntu/.venvs/srf-verification/bin/python",str(CODE/"srf_monitor.py"),"monitor"])
 for role in ("sampler","planner","manager"):
  start(role,["/tmp/cl_install/well_e_closed_loop/lib/well_e_closed_loop/closed_loop",role,"--ros-args","-p","use_sim_time:=true"])
 deadline=time.monotonic()+1000;last=0
 while not (OUT/"mission.json").exists():
  if time.monotonic()>deadline:raise TimeoutError("Preparation/mission wall guard exceeded")
  for name,p,_ in processes:
   if p.poll() is not None:raise RuntimeError(f"{name} exited {p.returncode}")
  if time.monotonic()-last>20:
   print("RUN",OUT.name,"elapsed",round(1000-(deadline-time.monotonic())),flush=True)
   for name in ("planner","manager"):
    f=OUT/(name+".jsonl")
    if f.exists() and f.stat().st_size:print(f.read_text().splitlines()[-1][:350],flush=True)
   last=time.monotonic()
  time.sleep(.5)
 mission=json.loads((OUT/"mission.json").read_text())
 if mission["status"]!="window_completed":
  raise RuntimeError("Mission failed: "+mission["status"])
 drain_begin=time.monotonic()
 # Stop sensing/navigation at 300 s, but let an already-running search finish.
 # This preserves its diagnostic result without dispatching a late goal.
 while not (OUT/"planner_idle.json").exists():
  if time.monotonic()>deadline:
   raise TimeoutError("Planner did not drain before the outer run guard")
  if any(p.poll() is not None for _,p,_ in processes):
   raise RuntimeError("Process exited during planner drain")
  time.sleep(.2)
 drain_wait_s=time.monotonic()-drain_begin
 time.sleep(3)
except (Exception,KeyboardInterrupt) as e:
 failure=str(e) or type(e).__name__;print("FAILURE",failure,flush=True)
finally:
 for name,p,log in reversed(processes):
  if p.poll() is None:
   os.killpg(p.pid,signal.SIGINT)
   try:p.wait(timeout=12)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
  log.close()
 (OUT/"run_status.json").write_text(json.dumps(dict(failure=failure,post_window_compute_wait_s=drain_wait_s,processes={n:p.returncode for n,p,_ in processes}),indent=2))
 (ROOT/"result/latest.json").write_text(json.dumps(dict(run=OUT.name)))
 print("SAVED",OUT,flush=True)

if failure is not None:
 sys.exit(1)

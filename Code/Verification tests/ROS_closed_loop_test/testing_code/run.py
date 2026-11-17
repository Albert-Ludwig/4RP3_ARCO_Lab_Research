"""Paired 300 s WIVR runs: 1 Hz sensing, 1 s versus 5 s GP/GRG updates."""
from pathlib import Path
from datetime import datetime,timezone
import os,sys,json,subprocess,time,signal,shutil,hashlib
ROOT=Path(__file__).resolve().parents[1];CODE=ROOT/'testing_code'
OUT=ROOT/'result'/datetime.now(timezone.utc).strftime('run_%Y%m%dT%H%M%SZ');OUT.mkdir(parents=True)
env=os.environ.copy();env.update(ROS_HOME='/ros-data',CL_MATRIX_THREADS='4',OMP_NUM_THREADS='4',OMP_DYNAMIC='FALSE',OPENBLAS_NUM_THREADS='1',SP_COMMON=str(OUT/'common'))
PY='/home/ubuntu/.venvs/srf-verification/bin/python';BIN='/tmp/cl_install/well_e_closed_loop/lib/well_e_closed_loop/closed_loop'
shutil.copytree(CODE,OUT/'source_snapshot',ignore=shutil.ignore_patterns('__pycache__'))
(OUT/'manifest.json').write_text(json.dumps(dict(sources={str(p.relative_to(CODE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in CODE.rglob('*') if p.is_file() and '__pycache__' not in p.parts},binary_sha256=hashlib.sha256(Path(BIN).read_bytes()).hexdigest()),indent=2))
subprocess.run([PY,str(CODE/'prepare.py')],env=env,check=True)
failure=None
for interval in (1,5):
    folder=OUT/f'update_{interval}s';shutil.copytree(OUT/'common',folder,ignore=shutil.ignore_patterns('evaluation_grid.npz'))
    cfg=json.loads((folder/'protocol.json').read_text());cfg['update_interval_s']=interval;(folder/'protocol.json').write_text(json.dumps(cfg,indent=2))
    if interval==5:shutil.copy2(OUT/'update_1s/graph.json',folder/'graph.json')
    env.update(CL_OUTPUT=str(folder),ROS_DOMAIN_ID=str(97+interval),GZ_PARTITION='gp_grg_'+OUT.name+str(interval))
    processes=[]
    def start(name,args):
        log=(folder/(name+'.log')).open('w');p=subprocess.Popen(args,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);processes.append((name,p,log))
    try:
        start('scene',['ros2','launch',str(CODE/'scene.launch.py')]);time.sleep(8)
        start('bag',['ros2','bag','record','-s','mcap','-o',str(folder/'rosbag'),'/clock','/simulation/ground_truth/odom','/odom','/tf','/tf_static','/cl/query','/cl/temperature','/cmd_vel'])
        start('source',[PY,str(CODE/'srf_monitor.py')]);start('monitor',[PY,str(CODE/'srf_monitor.py'),'monitor']);start('experiment',[BIN,'--ros-args','-p','use_sim_time:=true'])
        begin=time.monotonic();last=0
        while not (folder/'mission.json').exists():
            if time.monotonic()-begin>800:raise TimeoutError('Outer wall guard')
            for n,p,_ in processes:
                if p.poll() is not None:raise RuntimeError(f'{n} exited {p.returncode}')
            if time.monotonic()-last>20:
                f=folder/'events.jsonl';line=f.read_text().splitlines()[-1] if f.exists() and f.stat().st_size else 'waiting'
                print(interval,'s updates',round(time.monotonic()-begin),'wall s',line[:300],flush=True);last=time.monotonic()
            time.sleep(.2)
        print('MISSION',interval,(folder/'mission.json').read_text(),flush=True)
        if json.loads((folder/'mission.json').read_text())['status']!='window_completed':raise RuntimeError('Mission failed')
        deadline=time.monotonic()+185
        while not (folder/'worker_status.json').exists():
            if time.monotonic()>deadline:raise TimeoutError('In-flight worker did not finish')
            time.sleep(.2)
        time.sleep(2)
    except Exception as exc:failure=str(exc);print('FAILURE',failure,flush=True)
    finally:
        for n,p,log in reversed(processes):
            if p.poll() is None:
                os.killpg(p.pid,signal.SIGINT)
                try:p.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
            log.close()
        (folder/'run_status.json').write_text(json.dumps(dict(failure=failure,processes={n:p.returncode for n,p,_ in processes}),indent=2))
    if failure:break
(OUT/'suite_status.json').write_text(json.dumps(dict(failure=failure)))
(ROOT/'result/latest.json').write_text(json.dumps(dict(run=OUT.name)))
print('SAVED',OUT,flush=True)
if failure:sys.exit(1)
subprocess.run([PY,str(CODE/'evaluate.py'),str(OUT)],env=env,check=True)

"""One closed-loop sampling run with a separate return, using the saved WIVR fixture."""
from pathlib import Path
from datetime import datetime,timezone
import os,sys,json,subprocess,time,signal,shutil,hashlib
ROOT=Path(__file__).resolve().parents[1];CODE=ROOT/'code'
policy,=json.loads((CODE/'protocol.json').read_text())['policies']
domain='110' if policy=='mi' else '109'
OUT=ROOT/'results'/datetime.now(timezone.utc).strftime('run_%Y%m%dT%H%M%SZ');OUT.mkdir(parents=True)
env=os.environ.copy();env.update(ROS_HOME='/ros-data',CL_MATRIX_THREADS='4',OMP_NUM_THREADS='4',OMP_DYNAMIC='FALSE',OPENBLAS_NUM_THREADS='1',SP_COMMON=str(OUT/'common'))
PY='/home/ubuntu/.venvs/srf-verification/bin/python';BIN='/tmp/ros_mi_closed_loop_install/well_e_closed_loop/lib/well_e_closed_loop/closed_loop'
SNAPSHOT=CODE/'snapshots'/OUT.name/'source_snapshot'
shutil.copytree(CODE,SNAPSHOT,ignore=shutil.ignore_patterns('__pycache__','snapshots'))
(OUT/'manifest.json').write_text(json.dumps(dict(sources={str(p.relative_to(CODE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in CODE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and 'snapshots' not in p.relative_to(CODE).parts},binary_sha256=hashlib.sha256(Path(BIN).read_bytes()).hexdigest(),source_snapshot=SNAPSHOT.relative_to(ROOT).as_posix()),indent=2))
reference=ROOT.parent/'WIVR_test/results'/json.loads((CODE/'protocol.json').read_text())['reference_fixture_run']
shutil.copytree(reference/'common',OUT/'common')
shutil.copy2(CODE/'protocol.json',OUT/'common/protocol.json')
print('Loaded saved fixture from',reference,flush=True)
failure=None
for interval in (1,):
    folder=OUT/'latest_gp';shutil.copytree(OUT/'common',folder,ignore=shutil.ignore_patterns('evaluation_grid.npz'))
    cfg=json.loads((folder/'protocol.json').read_text());cfg['update_interval_s']=interval;(folder/'protocol.json').write_text(json.dumps(cfg,indent=2))
    shutil.copy2(reference/'latest_gp/graph.json',folder/'graph.json')
    nav_file=folder/'nav2_params.yaml'
    nav_text=nav_file.read_text()
    old_map=next(line for line in nav_text.splitlines() if 'yaml_filename:' in line)
    nav_file.write_text(nav_text.replace(old_map, f"    yaml_filename: {folder/'map.yaml'}", 1))
    (OUT/'fixture.json').write_text(json.dumps({
        'reference': str(reference.relative_to(ROOT.parent)),
        'inputs_sha256': {name: hashlib.sha256((folder/name).read_bytes()).hexdigest()
                          for name in ('initial.json','candidates.json','field.json','graph.json','world.sdf','map.yaml','map.pgm')},
        'evaluation_grid_sha256': hashlib.sha256((OUT/'common/evaluation_grid.npz').read_bytes()).hexdigest()
    },indent=2))
    env.update(CL_OUTPUT=str(folder),ROS_DOMAIN_ID=domain,GZ_PARTITION='gp_grg_'+policy+'_'+OUT.name)
    BAG=ROOT/'recordings'/OUT.name/'latest_gp'/'rosbag'
    BAG.parent.mkdir(parents=True,exist_ok=True)
    processes=[]
    def start(name,args):
        log=(folder/(name+'.log')).open('w');p=subprocess.Popen(args,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);processes.append((name,p,log))
    try:
        start('scene',['ros2','launch',str(CODE/'scene.launch.py')]);time.sleep(8)
        start('bag',['ros2','bag','record','-s','mcap','-o',str(BAG),'/clock','/simulation/ground_truth/odom','/odom','/tf','/tf_static','/cl/query','/cl/temperature','/cmd_vel'])
        start('source',[PY,str(CODE/'srf_monitor.py')]);start('monitor',[PY,str(CODE/'srf_monitor.py'),'monitor']);start('experiment',[BIN,'--ros-args','-p','use_sim_time:=true'])
        begin=time.monotonic();last=0
        while not (folder/'mission.json').exists():
            if time.monotonic()-begin>800:raise TimeoutError('Outer wall guard')
            for n,p,_ in processes:
                if p.poll() is not None:raise RuntimeError(f'{n} exited {p.returncode}')
            if time.monotonic()-last>20:
                f=folder/'events.jsonl';line=f.read_text().splitlines()[-1] if f.exists() and f.stat().st_size else 'waiting'
                print('Latest-GP',round(time.monotonic()-begin),'wall s',line[:300],flush=True);last=time.monotonic()
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
print('SAVED',OUT,flush=True)
if failure:
    sys.exit(1)
subprocess.run([PY,str(CODE/'evaluate_case.py'),str(OUT),str(folder)],env=env,check=True)
(ROOT/'results/latest.json').write_text(json.dumps(dict(run=OUT.name))+'\n')

"""Independent reset, real Gazebo/Nav2 processes, MCAP recording, offline evaluation."""
import argparse
import importlib.util
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from .storage import read_json,save_json,digest
from .prepare import ROOT,REPO
from .evaluation import evaluate


def stop(process):
    if process is None or process.poll() is not None:return
    os.killpg(process.pid,signal.SIGINT)
    try:process.wait(timeout=25)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid,signal.SIGTERM)
        try:process.wait(timeout=8)
        except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()


def verify_bag(run,config):
    sys.path.append('/usr/lib/python3/dist-packages')
    import rosbag2_py
    from collections import Counter
    reader=rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(run/'rosbag'),storage_id='mcap'),rosbag2_py.ConverterOptions('',''))
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    geometry_path=REPO/'Code/ROS_simulation/Nav2_test/testing_code/src/well_e_nav2_test/scripts/geometry.py'
    spec=importlib.util.spec_from_file_location('independent_footprint_geometry',geometry_path)
    geometry=importlib.util.module_from_spec(spec);spec.loader.exec_module(geometry)
    types={item.name:item.type for item in reader.get_all_topics_and_types()}
    truth_type=get_message(types['/simulation/ground_truth/odom'])
    clock_type=get_message(types['/clock'])
    obstacle=geometry.rectangle(config['scene']['obstacle']['center_m'],config['scene']['obstacle']['size_m'])
    counts=Counter();clearances=[];bounds=[];clock=[];poses=[]
    while reader.has_next():
        topic,data,_=reader.read_next();counts[topic]+=1
        if topic=='/simulation/ground_truth/odom':
            message=deserialize_message(data,truth_type);p=message.pose.pose.position
            body=geometry.footprint([p.x,p.y,geometry.yaw(message.pose.pose.orientation)],config['scene']['footprint_m'])
            if config['mode']=='synthetic_fixture':clearances.append(geometry.clearance(body,obstacle))
            xmin,xmax,ymin,ymax=config['scene']['map_bounds_m']
            bounds.append(all(xmin<=x<=xmax and ymin<=y<=ymax for x,y in body));poses.append([p.x,p.y])
        elif topic=='/clock':
            msg=deserialize_message(data,clock_type);clock.append(msg.clock.sec+msg.clock.nanosec*1e-9)
    required=['/clock','/tf','/map','/simulation/ground_truth/odom','/cmd_vel','/test/navigation_goal',
              '/navigate_to_pose/_action/status','/policy/events']
    checks=dict(required_topics=all(counts[t]>0 for t in required),
        clock_monotonic=bool(clock) and all(b>=a for a,b in zip(clock,clock[1:])),
        physical_footprint_in_bounds=bool(bounds) and all(bounds),
        physical_minimum_clearance=not clearances or min(clearances)>=config['scene']['minimum_clearance_m'])
    result=dict(status='passed' if all(checks.values()) else 'failed',checks=checks,
        topics=dict(counts),message_count=sum(counts.values()),required_topics=required,
        physical_pose_count=len(poses),minimum_footprint_clearance_m=min(clearances) if clearances else None,
        trajectory_distance_m=sum(__import__('math').dist(a,b) for a,b in zip(poses,poses[1:])),
        clearance_scope='known central obstacle for synthetic fixture; arbitrary real-map collision auditing not yet validated')
    save_json(run/'rosbag_summary.json',result)
    return result


def run(bundle,output,policies):
    bundle=Path(bundle).resolve();output=Path(output).resolve()
    if output.exists():raise FileExistsError('Use a new output directory; previous runs are preserved')
    manifest=read_json(bundle/'manifest.json')
    for relative,expected in manifest['files'].items():
        if digest(bundle/relative)!=expected:raise ValueError('Prepared bundle changed: '+relative)
    output.mkdir(parents=True);config=read_json(bundle/'online/config.json')
    save_json(output/'run_configuration.json',config)
    save_json(output/'source_hashes.json',{str(p.relative_to(REPO)):digest(p) for p in (ROOT/'testing_code').rglob('*.py')})
    oldtools=REPO/'Code/ROS_simulation/Nav2_test/testing_code/tools/rosbag_capture.py'
    spec=importlib.util.spec_from_file_location('reused_recorder',oldtools);record=importlib.util.module_from_spec(spec);spec.loader.exec_module(record)
    qos=REPO/'Code/ROS_simulation/Nav2_test/testing_code/src/well_e_nav2_test/config/rosbag_qos.yaml'
    runs=[]
    for policy in policies:
        run=output/policy;run.mkdir();launch=None;recorder=None
        env=os.environ.copy();env.update(ROS_DOMAIN_ID=str(config['ros_domain_id']),GZ_PARTITION=config['gz_partition'],
            POLICY_ONLINE=str(bundle/'online'),POLICY_GENERATED=str(run/'generated'),PYTHONDONTWRITEBYTECODE='1',
            PYTHONUNBUFFERED='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
        try:
            with (run/'simulation.log').open('w') as simlog,(run/'rosbag.log').open('w') as baglog,(run/'mission.log').open('w') as missionlog:
                launch=subprocess.Popen(['ros2','launch',str(ROOT/'testing_code/simulation.launch.py')],env=env,stdout=simlog,stderr=subprocess.STDOUT,start_new_session=True)
                command=record.record_command(run,qos)
                command.extend(['/policy/events'])
                recorder=subprocess.Popen(command,env=env,stdout=baglog,stderr=subprocess.STDOUT,start_new_session=True)
                record.wait_recorder(recorder,run)
                child=subprocess.Popen([sys.executable,'-m','nav2_policy.mission','--online',str(bundle/'online'),
                    '--output',str(run),'--policy',policy],env=env,stdout=missionlog,stderr=subprocess.STDOUT,start_new_session=True)
                end=time.monotonic()+max(300.,config['budget_s']*3)
                while child.poll() is None:
                    if launch.poll() is not None:stop(child);raise RuntimeError('Gazebo/Nav2 launch exited')
                    if time.monotonic()>end:stop(child);raise TimeoutError('Mission wall-clock watchdog')
                    time.sleep(.25)
                if child.returncode:raise RuntimeError('Mission failed; inspect mission.log and events.jsonl')
            record.stop_recorder(recorder);recorder=None
            bag=verify_bag(run,config)
            if bag['status']!='passed':raise RuntimeError('rosbag readback missing required telemetry')
            runs.append(dict(policy=policy,status='passed'))
            print(policy,'ROS mission and MCAP readback passed',flush=True)
        except Exception as exc:
            runs.append(dict(policy=policy,status='failed',error=str(exc)));print(policy,'FAILED:',exc,flush=True)
        finally:
            if recorder is not None:record.stop_recorder(recorder)
            stop(launch)
            if (run/'generated').is_dir():
                save_json(run/'scene_hashes.json',{p.name:digest(p) for p in (run/'generated').iterdir() if p.is_file()})
            time.sleep(2)
        # First establish a working MI baseline; stop immediately if it fails.
        if runs[-1]['status']!='passed':break
    summary=dict(mode=config['mode'],runs=runs,bundle=str(bundle),
        all_requested_policies_passed=len(runs)==len(policies) and all(r['status']=='passed' for r in runs))
    save_json(output/'execution.json',summary)
    if runs and all(r['status']=='passed' for r in runs):
        evaluate(bundle,output)
        if set(policies)=={'mi','emse','wivr'}:
            from .audit import audit
            checks=audit(bundle,output)
            summary['all_requested_policies_passed'] &= checks['passed']
            save_json(output/'execution.json',summary)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',required=True);p.add_argument('--output',required=True)
    p.add_argument('--policies',nargs='+',choices=['mi','emse','wivr'],default=['mi','emse','wivr'])
    a=p.parse_args();result=run(a.bundle,a.output,a.policies);print(result,flush=True)
    raise SystemExit(0 if result['all_requested_policies_passed'] else 1)

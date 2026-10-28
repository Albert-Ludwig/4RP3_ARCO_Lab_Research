"""Run the isolated Nav2 obstacle case and preserve all execution artifacts."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess

from rosbag_capture import TOPICS, record_command, wait_recorder, stop_recorder

def save(path, obj):
    path.write_text(json.dumps(obj, indent=2)+'\n', encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gui', action='store_true')
    args = parser.parse_args()
    code = Path(__file__).resolve().parents[1]
    stage, experiment = code.parent, code.parents[1]
    started = datetime.now(timezone.utc)
    output = stage/'result'/started.strftime('run_%Y%m%dT%H%M%S_%fZ')
    output.mkdir(parents=True, exist_ok=False)
    scenario = code/'src/well_e_nav2_test/config/scenario.json'
    env = dict(os.environ, ROS_DOMAIN_ID='72', GZ_PARTITION='well_e_nav2_test',
               NAV2_TEST_OUTPUT_DIR=str(output), NAV2_TEST_SCENARIO=str(scenario),
               NAV2_REQUIRE_ROSBAG='1',
               PYTHONDONTWRITEBYTECODE='1')
    if args.gui:
        env.update(DISPLAY=':0', LIBGL_ALWAYS_SOFTWARE='1')
    roots = [code, experiment/'BunkerMini_drive_test/testing_code']
    qos = code/'src/well_e_nav2_test/config/rosbag_qos.yaml'
    recorder_command = record_command(output, qos)
    manifest = {'started_utc': started.isoformat(), 'stage': 'Nav2_test', 'gui': args.gui,
                'scenario': json.loads(scenario.read_text()), 'ros_distro': env.get('ROS_DISTRO'),
                'nav2_version': subprocess.check_output(
                    ['dpkg-query', '-W', '-f=${Version}', 'ros-jazzy-nav2-bringup'], text=True).strip(),
                'mesh_sha256': sha(Path(env.get('ROS_HOME', '/ros-data'))/'assets/bunker_mini/base_link.STL'),
                'recording': {'backend': 'rosbag2', 'storage_id': 'mcap', 'directory': 'rosbag',
                              'use_sim_time': True, 'topics': TOPICS, 'command': recorder_command},
                'source_sha256': {p.relative_to(experiment).as_posix(): sha(p)
                    for root in roots for p in sorted(root.rglob('*')) if p.is_file()
                    and '__pycache__' not in p.parts and p.name not in ('report.py', 'requirements-report.txt')},
                'gazebo_version': subprocess.check_output(['gz', 'sim', '--versions'], text=True).strip()}
    save(output/'manifest.json', manifest)
    clean_shutdown, code_result, failure = False, 1, None
    recorder, bag_closed, bag_verified = None, False, False
    with (output/'launch.log').open('w') as log, (output/'rosbag_record.log').open('w') as bag_log:
        process = subprocess.Popen(['bash', str(code/'run_nav2.sh'), f'gui:={str(args.gui).lower()}'],
                                   env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            # Wait for a live clock before subscribing to latched map/static TF data.
            subprocess.run(['ros2', 'topic', 'echo', '/clock', 'rosgraph_msgs/msg/Clock', '--once',
                            '--qos-reliability', 'best_effort'], env=env,
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=45, check=True)
            recorder = subprocess.Popen(recorder_command, env=env, stdout=bag_log,
                                        stderr=subprocess.STDOUT, start_new_session=True)
            wait_recorder(recorder, output)
            with (output/'console.log').open('w') as console:
                result = subprocess.run(['python3', str(code/'src/well_e_nav2_test/scripts/navigation_probe.py')],
                                        env=env, stdout=console, stderr=subprocess.STDOUT, timeout=310)
            code_result = result.returncode
            if code_result == 0:
                topics = subprocess.run(['ros2', 'topic', 'list', '--no-daemon'], env=env,
                                        capture_output=True, text=True, timeout=10, check=True)
                (output/'topics.txt').write_text(topics.stdout)
        except Exception as exc:
            failure = f'{type(exc).__name__}: {exc}'
        finally:
            # Flush MCAP and metadata while the simulation is still publishing.
            bag_closed = stop_recorder(recorder)
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=20)
                    clean_shutdown = True
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
    generated = Path(env.get('ROS_HOME', '/ros-data'))/'generated/Nav2_test'
    for name in ('world.sdf', 'map.pgm', 'map.yaml', 'nav2_params.yaml'):
        if (generated/name).exists():
            shutil.copy2(generated/name, output/name)
    metrics = json.loads((output/'metrics.json').read_text()) if (output/'metrics.json').exists() else {}
    if code_result == 0 and bag_closed:
        try:
            with (output/'rosbag_verification.log').open('w') as verification_log:
                subprocess.run(['python3', str(code/'tools/rosbag_capture.py'), str(output)],
                               env=env, stdout=verification_log, stderr=subprocess.STDOUT,
                               timeout=70, check=True)
            bag_verified = True
        except Exception as exc:
            failure = f'rosbag verification: {type(exc).__name__}: {exc}'
    status = ('passed' if code_result == 0 and metrics.get('status') == 'passed' and clean_shutdown
              and bag_closed and bag_verified else 'failed')
    manifest.update(status=status, finished_utc=datetime.now(timezone.utc).isoformat(),
                    clean_shutdown=clean_shutdown, rosbag_closed_cleanly=bag_closed,
                    rosbag_read_and_replay_verified=bag_verified)
    save(output/'verification.json', {'status': status, 'checks': {
        **metrics.get('checks', {}), 'probe_exit_success': code_result == 0,
        'simulation_shutdown_completed': clean_shutdown, 'rosbag_closed_cleanly': bag_closed,
        'rosbag_read_and_replay_verified': bag_verified}})
    if failure:
        manifest['error'] = failure
    manifest['artifact_sha256'] = {p.relative_to(output).as_posix(): sha(p) for p in sorted(output.rglob('*'))
                                  if p.is_file() and p.name != 'manifest.json'}
    save(output/'manifest.json', manifest)
    save(stage/'result/latest.json', {'run': output.name, 'status': status})
    save(stage/'result/status.json', {
        'stage': 'Nav2_test', 'status': status, 'implementation_status': 'known_static_obstacle_only',
        'implemented_checks': ['single_goal', 'known_static_obstacle_detour', 'physical_footprint_clearance'],
        'pending_checks': ['unmapped_obstacle_perception', 'ordered_waypoints', 'unreachable_goal', 'cancel_and_stop'],
        'metrics': f'{output.name}/metrics.json', 'recording': f'{output.name}/rosbag'})
    print(json.dumps(metrics, indent=2), flush=True)
    print(f'Artifacts: {output}', flush=True)
    return 0 if status == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())

#!/usr/bin/env python3
"""Start an isolated test launch and shut down only the processes it owns."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gui', action='store_true')
    parser.add_argument('--output', type=Path, help='New result directory; existing directories are protected')
    parser.add_argument('--probe', choices=('smoke_test', 'turning_test', 'steering_validation'),
                        default='smoke_test')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    stage = root.parent
    started = datetime.now(timezone.utc)
    run_id = started.strftime('run_%Y%m%dT%H%M%S_%fZ')
    output = args.output.resolve() if args.output else stage / 'result' / run_id
    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, ROS_DOMAIN_ID='71', GZ_PARTITION='well_e_bunker_mini')
    env['BUNKER_TEST_OUTPUT_DIR'] = str(output)
    if args.gui:
        env.update(DISPLAY=':0', LIBGL_ALWAYS_SOFTWARE='1')
    runtime = Path(env.get('ROS_HOME', '/ros-data'))
    version = subprocess.run(['gz', 'sim', '--versions'], env=env, capture_output=True,
                             text=True, timeout=10, check=True).stdout.strip()
    manifest = {
        'run_id': output.name, 'status': 'running', 'started_utc': started.isoformat(),
        'stage': 'BunkerMini_drive_test', 'mode': 'simulation', 'gui': args.gui,
        'ros_distro': env.get('ROS_DISTRO'), 'gazebo_version': version,
        'python_version': sys.version.split()[0], 'ros_domain_id': 71,
        'gz_partition': env['GZ_PARTITION'], 'world': 'bunker_test',
        'probe': args.probe,
        'drive_config': json.loads((root / 'src/well_e_bunker_sim/config/drive.json').read_text()),
        'units': {'position': 'm', 'yaw': 'rad', 'time': 's'},
        'model_commit': '27633a956c845903ee630538afeb17fe70afdd84',
        'mesh_sha256': sha256(runtime / 'assets/bunker_mini/base_link.STL'),
        'source_sha256': {p.relative_to(stage).as_posix(): sha256(p)
                          for p in sorted(root.rglob('*')) if p.is_file()
                          and '__pycache__' not in p.parts
                          and p.name not in ('report.py', 'requirements-report.txt')},
    }
    save(output / 'manifest.json', manifest)
    code, shutdown_complete, failure = 1, False, None
    # Build and run the relocated source; logs in ROS_HOME stay outside the repository.
    with (output / 'launch.log').open('w', encoding='utf-8') as log:
        command = ['bash', str(root / 'run_sim.sh'), f'gui:={str(args.gui).lower()}']
        process = subprocess.Popen(command, env=env,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            # The probe uses source imports, so it can wait while colcon builds the package.
            with (output / 'console.log').open('w', encoding='utf-8') as console:
                result = subprocess.run([
                    'python3', str(root / f'src/well_e_bunker_sim/scripts/{args.probe}.py')],
                    env=env, timeout=240, stdout=console, stderr=subprocess.STDOUT, check=False)
            code = result.returncode
            if process.poll() is not None:
                raise RuntimeError(f'Launch exited early; inspect {output / "launch.log"}')
            topics = subprocess.run(['ros2', 'topic', 'list', '--no-daemon'], env=env,
                                    capture_output=True, text=True, timeout=10, check=True)
            (output / 'topics.txt').write_text(topics.stdout, encoding='utf-8')
        except Exception as exc:
            failure, code = f'{type(exc).__name__}: {exc}', 1
        finally:
            if process.poll() is None:
                # ROS launch forwards SIGINT to its children; avoid delivering it twice.
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=15)
                    shutdown_complete = True
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
    for name in ('world.sdf', 'robot.urdf'):
        artifact = runtime / 'generated/well_e_bunker_sim' / name
        if artifact.exists():
            shutil.copy2(artifact, output / name)
    check = subprocess.run(['gz', 'sdf', '-k', str(output / 'world.sdf')], env=env,
                           capture_output=True, text=True, timeout=15, check=False)
    (output / 'model_check.log').write_text(check.stdout+check.stderr, encoding='utf-8')
    metrics_path = output / 'metrics.json'
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    checks = {
        'probe_exit_success': code == 0,
        'motion_checks_passed': metrics.get('status') == 'passed',
        'trajectory_saved': (output / 'trajectory.csv').is_file(),
        'topics_saved': (output / 'topics.txt').is_file(),
        'sdf_valid': check.returncode == 0,
        'simulation_shutdown_completed': shutdown_complete,
    }
    passed = all(checks.values())
    save(output / 'verification.json', {'status': 'passed' if passed else 'failed', 'checks': checks})
    manifest.update(status='passed' if passed else 'failed',
                    finished_utc=datetime.now(timezone.utc).isoformat())
    if failure:
        manifest['error'] = failure
    manifest['artifact_sha256'] = {p.name: sha256(p) for p in sorted(output.iterdir())
                                  if p.is_file() and p.name != 'manifest.json'}
    save(output / 'manifest.json', manifest)
    if output.parent == stage / 'result' and args.probe == 'smoke_test':
        save(stage / 'result/latest.json', {'run': output.name, 'status': manifest['status']})
    print(json.dumps(metrics, indent=2), flush=True)
    print(f'Artifacts: {output}', flush=True)
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())

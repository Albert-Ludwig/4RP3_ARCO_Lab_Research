"""Build three concise reports from recorded artifacts; never execute ROS here."""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import textwrap

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

ROOT = Path(__file__).resolve().parents[3]
STAGES = ('BunkerMini_drive_test', 'Nav2_test', 'Integration_test')
ROWS = [
    ('forward_ground_truth_m', 'Physical forward displacement', 'm', '0.3 < x < 0.9'),
    ('forward_odometry_m', 'Odometry forward displacement', 'm', '0.3 < x < 0.9'),
    ('positive_rotation_ground_truth_rad', 'Physical positive rotation', 'rad', '1.14 < x < 1.26'),
    ('turn_relative_error', 'Turn relative error', 'fraction', 'x < 0.05'),
    ('turn_odometry_error_rad', 'Corrected odometry turn error', 'rad', 'x < 0.05'),
    ('reverse_ground_truth_m', 'Reverse displacement along heading', 'm', '-0.5 < x < -0.15'),
    ('watchdog_stopped_drift_m', 'Drift over 1 s after timeout stop', 'm', 'x < 0.03'),
    ('odom_tf_translation_error_m', 'Odometry/TF translation difference', 'm', 'x < 0.03'),
]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '|'+'---|'*len(headers)] +
                     ['| '+' | '.join(str(v) for v in row)+' |' for row in rows])


def md(text):
    return nbformat.v4.new_markdown_cell(textwrap.dedent(text).strip())


def code(text):
    return nbformat.v4.new_code_cell(textwrap.dedent(text).strip())


def setup(stage):
    return code(f'''\
        from pathlib import Path
        import csv, hashlib, html, json
        from IPython.display import Image, Markdown, display
        name = {stage!r}
        locations = [candidate for parent in [Path.cwd(), *Path.cwd().parents]
                     for candidate in [parent, parent/name,
                         parent/'Code'/'10.27_ROS_simulation'/name]]
        root = next(p for p in locations if p.name == name and (p/'result').is_dir())
        def read(path):
            return json.loads(path.read_text(encoding='utf-8'))
        def show(headers, rows):
            display(Markdown('\\n'.join(['| '+' | '.join(headers)+' |',
                '|'+'---|'*len(headers)] +
                ['| '+' | '.join(str(v) for v in row)+' |' for row in rows])))
        ''')


def trajectory_figure(run):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with (run/'trajectory.csv').open(encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    rows = [row for row in rows if row['odom_x_m']]
    def values(key):
        return [float(row[key]) for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    axes[0].plot(values('truth_x_m'), values('truth_y_m'), label='Physical pose')
    axes[0].plot(values('odom_x_m'), values('odom_y_m'), '--', label='Corrected track odometry')
    axes[0].set(xlabel='x (m)', ylabel='y (m)', title='Planar trajectory', aspect='equal')
    axes[1].plot(values('time_s'), values('truth_yaw_rad'), label='Physical pose')
    axes[1].plot(values('time_s'), values('odom_yaw_rad'), '--', label='Corrected track odometry')
    raw = [row for row in rows if row.get('raw_odom_yaw_rad')]
    if raw:
        axes[1].plot([float(row['time_s']) for row in raw],
                     [float(row['raw_odom_yaw_rad']) for row in raw], ':', label='Raw track odometry')
    axes[1].set(xlabel='Simulation time (s)', ylabel='Yaw (rad)', title='Turning and slip')
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    fig.savefig(run/'trajectory.png', dpi=150)
    plt.close(fig)


def steering_report(stage, manifest, metrics):
    config = manifest['drive_config']
    calibration = stage/'result'/config['calibration_run']
    baseline = stage/'result/run_20261027T215129_313855Z'
    before = read(baseline/'metrics.json')['checks']['positive_rotation_ground_truth_rad']
    candidates = [path for path in sorted((stage/'result').glob('run_*'))
                  if (path/'manifest.json').exists()
                  and read(path/'manifest.json').get('probe') == 'steering_validation']
    validation_path = candidates[-1]
    validation = read(validation_path/'metrics.json')
    case_table = table(['Validation case', 'Target (rad)', 'Physical (rad)', 'Turn error (%)',
                        'Odom yaw error (rad)', 'Passed'],
        [[case['name'], f"{case['expected_rotation_rad']:.4f}",
          f"{case['physical_rotation_rad']:.4f}", f"{100*case['turn_relative_error']:.2f}",
          f"{case['odometry_yaw_error_rad']:.4f}", case['passed']] for case in validation['cases']])
    before_after = table(['Item', 'Original', 'Corrected'], [
        ['Target rotation (rad)', '1.200000', '1.200000'],
        ['Physical rotation (rad)', f'{before:.6f}',
         f"{metrics['checks']['positive_rotation_ground_truth_rad']:.6f}"],
        ['Relative turn error (%)', f'{100*abs(before/1.2-1):.2f}',
         f"{100*metrics['checks']['turn_relative_error']:.2f}"]])
    text = f'''## Steering correction and independent checks

{before_after}

The original fixture used steering efficiency 1.0. The measured steady track-to-body yaw gain was approximately 0.62370 in both +/-0.4 rad/s calibration turns. The frozen efficiency and odometry yaw scale are now **{config['steering_efficiency']}** in [drive.json](testing_code/src/well_e_bunker_sim/config/drive.json).

The [diagnostic run](result/{calibration.name}/metrics.json) retains the original and compensated left/right turns and arcs. Parameters were fitted from its two efficiency=1 in-place cases. The following validation uses different turn rates (+/-0.2 and +/-0.6 rad/s) and wider arcs (0.2 m/s, +/-0.2 rad/s), without retuning:

{case_table}

Validation status: **{validation['status']}**. Limits were set before execution: turn error <10%, odometry yaw error <0.05 rad and accumulated planar odometry error <0.05 m. Maximum planar error was {max(c['odometry_position_error_m'] for c in validation['cases']):.4f} m. Evidence: [validation metrics](result/{validation_path.name}/metrics.json) and [run manifest](result/{validation_path.name}/manifest.json).

The default smoke test is also stricter: its former 0.4-1.8 rad turning check is replaced by <5% error from 1.2 rad and <0.05 rad odometry turn error. Earlier result files are preserved.

Public `/odom` and `odom → base_link` now come from [odometry_adapter.py](testing_code/src/well_e_bunker_sim/scripts/odometry_adapter.py), which integrates raw track velocity with the calibrated yaw scale. It does **not** subscribe to simulator ground truth. Raw odometry/TF remain on `/simulation/track_odom` and `/simulation/track_tf`; the independent body pose remains available for evaluation only. Odometry covariance is not calibrated.

The correction is specific to this flat-ground simulated contact model. It is not a real Bunker Mini calibration or evidence for other terrain. The upstream [TrackedVehicle implementation](https://github.com/gazebosim/gz-sim/blob/gz-sim8/src/systems/tracked_vehicle/TrackedVehicle.cc) applies steering efficiency to track commands but integrates uncorrected virtual-wheel odometry; this is why adjusting the drive parameter alone is insufficient.
'''
    return text, calibration.name, validation_path.name


def drive_report(stage):
    latest = read(stage/'result/latest.json')
    run = stage/'result'/latest['run']
    metrics, manifest, verification = [read(run/name) for name in
        ('metrics.json', 'manifest.json', 'verification.json')]
    result_table = table(['Check', 'Result', 'Unit', 'Acceptance criterion'],
        [(label, f"{metrics['checks'][key]:.6f}", unit, criterion)
         for key, label, unit, criterion in ROWS])
    link = 'result/'+run.name
    steering, calibration_name, validation_name = steering_report(stage, manifest, metrics)
    trajectory_figure(run)
    description = f'''# BunkerMini Drive Test

## Objective and method

Verify the manufacturer's Mini mesh, Gazebo tracked drive, `/cmd_vel`, `/odom`, TF and command timeout. Environment: ROS 2 {manifest['ros_distro']} / Gazebo {manifest['gazebo_version']}, isolated domain 71.

Fixed flat-ground sequence: stop for 1 s → drive forward at 0.2 m/s for 3 s → stop for 1 s → turn at 0.4 rad/s for 3 s → stop for 1 s → reverse at -0.15 m/s for 2 s → stop publishing commands, wait 2 s, then observe for 1 s. Motion phases use simulation time; the watchdog expires commands after 0.5 s of wall time.

## Results

Status: **{manifest['status']}**. Start time (UTC): `{manifest['started_utc']}`. Six saved-artifact checks: **{verification['status']}**. GUI enabled: `{manifest['gui']}`.

{result_table}

![Trajectory and yaw]({link}/trajectory.png)

The plot compares physical pose, corrected track odometry and the retained raw yaw estimate. Flat-ground steering and odometry corrections have been validated on the maneuvers below; hardware dynamics remain uncalibrated.

{steering}

## Files and execution

| Location | Contents |
|---|---|
| [test_report.ipynb](test_report.ipynb) | Executed result replay, artifact/source integrity checks and trajectory plot; does not rerun ROS |
| [testing_code](testing_code) | ROS package, launchers, motion checks, log capture and report generation |
| [{link}]({link}) | Metrics, trajectory.csv, manifest, verification, observed topics, console/launch logs, SDF/URDF and plot |
| [result/initial_run](result/initial_run) | Original passing result and launch log from before reorganization; no trajectory was recorded |

On Windows, double-click `C:\\Ubuntu\\Test-Bunker-Mini.cmd` to run a new test or `C:\\Ubuntu\\Bunker-Mini-Sim.cmd` to open the simulation. Each test creates a new result directory and preserves earlier runs. `result/latest.json` points to the latest record.

```powershell
.\\testing_code\\Start-BunkerMini.ps1 -Action test
.\\testing_code\\Start-BunkerMini.ps1 -Action sim -Headless
```

From this test directory inside the container: `bash testing_code/run_test.sh [--gui] [--output <new-result-directory>]`. Use `--probe steering_validation` for the six fixed-gain validation maneuvers, or `--probe turning_test` for the diagnostic parameter comparison. Diagnostic runs do not replace the latest smoke-test pointer. Source: [model.py](testing_code/src/well_e_bunker_sim/launch/model.py), [sim.launch.py](testing_code/src/well_e_bunker_sim/launch/sim.launch.py), [watchdog](testing_code/src/well_e_bunker_sim/scripts/cmd_watchdog.py), [smoke_test](testing_code/src/well_e_bunker_sim/scripts/smoke_test.py), [runner](testing_code/tools/run_smoke_test.py).

Asset caches, generated files, build artifacts and raw ROS logs remain in `C:\\Ubuntu\\ROS`. The notebook environment is `reporting-venv` under that directory.

## Scope and limitations

The Mini STL is pinned to manufacturer commit `27633a956c845903ee630538afeb17fe70afdd84`. Total mass, 40.953842 kg, comes from the upstream single-body URDF. Contact boxes, inertia distribution, friction and steering efficiency are simulation assumptions. Collision boxes do not cover the complete CAD envelope. The provisional limits of 0.35 m/s and 0.6 rad/s are experiment settings, not confirmed hardware limits.

This stage includes no Nav2, map localization, LiDAR/ZED/IMU simulation or temperature measurement. GP/GRG is unchanged. The measured steering correction covers this fixed scene only; narrow passages, complex terrain, hardware performance and navigation accuracy remain untested.

Sources: [manufacturer Mini assets](https://github.com/agilexrobotics/ugv_gazebo_sim/tree/27633a956c845903ee630538afeb17fe70afdd84/bunker/bunker_mini), [manufacturer ROS 2 driver](https://github.com/agilexrobotics/bunker_ros2/tree/humble), [delivery documentation](https://indrorobotics.notion.site/MCMU-CA-03-3ab1bb7999b28041a889c0d6b5fe8203). The upstream asset package.xml declares BSD.
'''
    cells = [md('# BunkerMini Drive Test\n\nReplay saved drive-simulation artifacts without starting ROS or rerunning the motion test.'),
             setup(stage.name), md('## 1. Configuration and run record'), code(f'''\
                run = root/'result'/{run.name!r}
                manifest = read(run/'manifest.json')
                metrics = read(run/'metrics.json')
                verification = read(run/'verification.json')
                show(['Configuration', 'Value'], [[key, manifest[key]] for key in
                    ['status', 'started_utc', 'mode', 'gui', 'ros_distro', 'gazebo_version',
                     'ros_domain_id', 'model_commit', 'mesh_sha256']])
                show(['Drive parameter', 'Value'], list(manifest['drive_config'].items()))
                '''), md('## 2. Recorded results\n\n'+result_table), code('''\
                display(Markdown('**Verification status: '+verification['status']+'**'))
                show(['Verification', 'Passed'], list(verification['checks'].items()))
                display(Image(filename=str(run/'trajectory.png')))
                with (run/'trajectory.csv').open(encoding='utf-8') as stream:
                    trajectory = list(csv.DictReader(stream))
                print('Saved physical-pose samples:', len(trajectory))
                print('Last simulation timestamp (s):', trajectory[-1]['time_s'])
                '''), md('## 3. Record integrity\n\nVerify the saved artifacts using SHA-256. Report source-code consistency separately.'), code('''\
                artifact_matches = {name: hashlib.sha256((run/name).read_bytes()).hexdigest()==digest
                    for name, digest in manifest['artifact_sha256'].items()}
                assert all(artifact_matches.values()), 'Saved artifact changed'
                source_matches = {name: (root/name).exists() and
                    hashlib.sha256((root/name).read_bytes()).hexdigest()==digest
                    for name, digest in manifest['source_sha256'].items()}
                show(['Integrity check', 'Result'], [['Captured artifacts match', all(artifact_matches.values())],
                    ['Current source matches captured source', all(source_matches.values())]])
                '''), md(steering), code(f'''\
                calibration_run = root/'result'/{calibration_name!r}
                validation_run = root/'result'/{validation_name!r}
                calibration = read(calibration_run/'metrics.json')
                validation = read(validation_run/'metrics.json')
                assert calibration['status'] == 'passed'
                assert validation['status'] == 'passed'
                assert len(validation['cases']) == 6
                assert all(case['passed'] for case in validation['cases'])
                for evidence in [calibration_run, validation_run]:
                    record = read(evidence/'manifest.json')
                    assert all(hashlib.sha256((evidence/name).read_bytes()).hexdigest()==digest
                        for name, digest in record['artifact_sha256'].items())
                print('Calibration and independent validation artifacts verified.')
                '''), md('## 4. Conclusions and next stage\n\nBasic motion, command-timeout stopping and odom/TF checks passed. Flat-ground steering compensation and corrected odometry passed the stricter smoke test and six additional maneuvers. This is simulation calibration only; covariance and hardware dynamics remain uncalibrated. Navigation is evaluated separately in [Nav2_test](../Nav2_test/README.md); GP/GRG ROS integration remains pending.\n\nSee the [README](README.md) and [smoke_test.py](testing_code/src/well_e_bunker_sim/scripts/smoke_test.py) for execution instructions and acceptance thresholds.')]
    return description, cells


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=STAGES)
    args = parser.parse_args()
    if args.stage == 'Integration_test':
        print('Integration_test has not started; no report is generated.')
        return
    runtime = Path(os.environ.get('BUNKER_REPORT_RUNTIME', 'C:/Ubuntu/ROS/reporting-cache'
                                  if os.name == 'nt' else '/ros-data/reporting-cache'))
    runtime.mkdir(parents=True, exist_ok=True)
    os.environ['MPLCONFIGDIR'] = str(runtime/'matplotlib')
    os.environ['IPYTHONDIR'] = str(runtime/'ipython')
    os.environ['JUPYTER_RUNTIME_DIR'] = str(runtime/'jupyter')
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    sys.dont_write_bytecode = True
    results = []
    for name in ([args.stage] if args.stage else STAGES[:2]):
        stage = ROOT/name
        if not (stage/'result/latest.json').exists():
            print(f'Skipping {name}: no experiment run exists.')
            continue
        if name == 'Nav2_test' and (stage/'result/latest.json').exists():
            spec = importlib.util.spec_from_file_location('nav2_report', stage/'testing_code/tools/report.py')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            text, cells = module.build_report(stage)
        else:
            text, cells = drive_report(stage)
        notebook = nbformat.v4.new_notebook(cells=cells, metadata={
            'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
            'language_info': {'name': 'python'}, 'report_mode': 'saved_artifacts_only'})
        manager = KernelManager(kernel_name='python3')
        manager.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
        try:
            NotebookClient(notebook, km=manager, timeout=90,
                           resources={'metadata': {'path': str(stage)}}).execute()
        finally:
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)
        nbformat.validate(notebook)
        (stage/'README.md').write_text(text, encoding='utf-8')
        nbformat.write(notebook, stage/'test_report.ipynb')
        state = (read(stage/'result/latest.json')['status'] if (stage/'result/latest.json').exists()
                 else read(stage/'result/status.json')['status'])
        results.append({'stage': name, 'experiment_status': state,
                        'notebook_executed': True, 'notebook_valid': True,
                        'readme_sha256': sha(stage/'README.md'),
                        'notebook_sha256': sha(stage/'test_report.ipynb')})
    for row in results:
        (ROOT/row['stage']/'result/report_verification.json').write_text(
            json.dumps(row, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()

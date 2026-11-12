"""English Nav2 report from saved artifacts; shared coordinator executes the notebook."""
import importlib.util
import json
from pathlib import Path
import sys

import nbformat

sys.dont_write_bytecode = True


def build_report(stage):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import csv
    from matplotlib.patches import Rectangle
    latest = json.loads((stage/'result/latest.json').read_text())
    run = stage/'result'/latest['run']
    manifest = json.loads((run/'manifest.json').read_text())
    metrics = json.loads((run/'metrics.json').read_text())
    recording = json.loads((run/'rosbag_summary.json').read_text()) if (run/'rosbag_summary.json').exists() else None
    replay = json.loads((run/'rosbag_replay_verification.json').read_text()) if (run/'rosbag_replay_verification.json').exists() else None
    scenario = manifest['scenario']
    with (run/'trajectory.csv').open() as stream:
        rows = [row for row in csv.DictReader(stream) if row['phase'] in ('navigating', 'settling')]
    planned = json.loads((run/'planned_path.json').read_text())
    center, size = scenario['obstacle']['center_m'], scenario['obstacle']['size_m']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].add_patch(Rectangle((center[0]-size[0]/2, center[1]-size[1]/2), *size,
                                color='firebrick', alpha=0.8, label='Known physical obstacle'))
    axes[0].plot([p[0] for p in planned], [p[1] for p in planned], '--', label='Initial Nav2 plan')
    axes[0].plot([float(r['x_m']) for r in rows], [float(r['y_m']) for r in rows], label='Physical trajectory')
    axes[0].plot([0, 6], [0, 0], ':', color='gray', label='Blocked direct route')
    axes[0].scatter([0, 6], [0, 0], c=['green', 'black'], marker='x', zorder=5)
    xmin, xmax, ymin, ymax = scenario['map_bounds_m']
    axes[0].set(xlim=(xmin, xmax), ylim=(ymin, ymax), xlabel='x (m)', ylabel='y (m)',
                aspect='equal', title='Point-to-point obstacle detour')
    axes[0].legend(fontsize=7)
    axes[1].plot([float(r['time_s'])-float(rows[0]['time_s']) for r in rows],
                 [float(r['clearance_m']) for r in rows])
    axes[1].axhline(scenario['minimum_clearance_m'], color='red', linestyle='--', label='Acceptance minimum')
    axes[1].set(xlabel='Elapsed simulation time (s)', ylabel='Footprint clearance (m)',
                title='Physical body to obstacle')
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.savefig(run/'navigation.png', dpi=150)
    plt.close(fig)
    metric_rows = [
        ('Final position error', 'final_position_error_m', 'm'),
        ('Final heading error', 'final_yaw_error_rad', 'rad'),
        ('Minimum full-body clearance', 'minimum_footprint_clearance_m', 'm'),
        ('Physical path length', 'actual_path_length_m', 'm'),
        ('Initial planned path length', 'planned_path_length_m', 'm'),
        ('Navigation duration', 'navigation_duration_s', 's'),
        ('Sampled footprint overlaps', 'sampled_footprint_overlap_count', 'samples'),
    ]
    table = '\n'.join(['| Metric | Result | Unit |', '|---|---:|---|'] +
        [f'| {label} | {metrics[key]:.6f} | {unit} |' for label, key, unit in metric_rows])
    link = 'result/'+run.name
    bag_section = ''
    video_section = ''
    if recording:
        spec = importlib.util.spec_from_file_location('nav2_video', Path(__file__).with_name('make_video.py'))
        renderer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(renderer)
        video, video_metadata = renderer.ensure_video(run)
        video_section = f'''## Embedded replay video

The [Notebook](test_report.ipynb) embeds a playable [MP4]({link}/replay/rosbag_replay.mp4) generated directly from the physical-pose messages in the MCAP recording. Duration: {video_metadata['duration_s']:.2f} s, at 1× simulation-time speed and {video_metadata['fps']} fps.

This is a top-down recorded-data visualization of the robot footprint, heading and route, **not camera footage or a Gazebo screen recording**. The bag contains no recorded image topics. Frames use the nearest recorded pose without interpolating positions. The scenario comes from the saved run configuration.

Provenance and full-video decode verification: [video_metadata.json]({link}/replay/video_metadata.json). Decoded bag poses: [bag_poses.csv]({link}/replay/bag_poses.csv). Creating the video does not rerun navigation or modify the original bag.
'''
    if recording:
        topic_table = '\n'.join(['| Recorded topic | Messages |', '|---|---:|']+
            [f'| `{topic}` | {count} |' for topic, count in recording['topics'].items()])
        bag_section = f'''## rosbag2 recording

The primary ROS recording is [rosbag/]({link}/rosbag), stored as MCAP with `metadata.yaml`. It contains **{recording['message_count']} messages across {len(recording['topics'])} topics**. Recording timestamps use simulation time. The recorder starts before the goal is sent and flushes before simulation shutdown.

{topic_table}

Readback status: **{recording['status']}**. The bag contains the map, static and dynamic TF, goal, action status and physical motion throughout navigation. The bag/analysis pose match fraction is {recording['analysis_pose_match_fraction']:.3f}; minimum footprint clearance independently recomputed from bag poses is {recording['minimum_clearance_from_bag_m']:.6f} m. CSV and JSON remain analysis products, checked against the authoritative ROS recording.

Actual ROS playback was also verified at 5× speed on isolated domain 73, without replaying velocity-command topics: **{replay['status']}**. See [bag info]({link}/rosbag_info.txt), [readback checks]({link}/rosbag_summary.json), and [playback checks]({link}/rosbag_replay_verification.json).

To inspect or replay this run inside the container:

```bash
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=73
bag=/workspace/Code/10.27_ROS_simulation/Nav2_test/{link}/rosbag
ros2 bag info "$bag"
ros2 bag play "$bag" --clock --topics /tf /tf_static /map /odom /plan /robot_description /simulation/ground_truth/odom /test/navigation_goal /global_costmap/costmap /local_costmap/costmap
```

Use the same domain and `use_sim_time:=true` for RViz; set its fixed frame to `map`. This replays ROS data for inspection; it does not rerun Gazebo physics or reissue the navigation action. The visualization replay excludes velocity commands. Transient-local QoS overrides preserve latched map, static TF and model data; see [rosbag_qos.yaml](testing_code/src/well_e_nav2_test/config/rosbag_qos.yaml). Reference: [ROS 2 rosbag2 recording and QoS](https://github.com/ros2/rosbag2/blob/rolling/README.md).
'''
    description = f'''# Nav2 Point-to-Point Obstacle Test

## Objective and setup

Send one `NavigateToPose` goal to the calibrated Bunker Mini. Nav2 chooses and executes the route; no intermediate waypoints or manual velocity commands are supplied.

The field is 10 × 8 m, from x=-2 to 8 m and y=-4 to 4 m. Start: (0,0,0); goal: (6,0,0). A 1.5 × 2 × 1 m physical box at (3,0) blocks the direct route. A 0.05 m occupancy map describes the same obstacle and field boundary. The footprint covers the full CAD planar envelope, with 0.07 m costmap padding.

**Controlled assumptions:** the static obstacle is known in advance; a simulation-only ground-truth `map→odom` correction provides ideal localization. No AMCL, LiDAR, camera perception or unknown-obstacle detection is tested. The map feeds Nav2 static and inflation layers; NavFn plans and Regulated Pure Pursuit controls motion with footprint collision checking enabled.

## Results

Status: **{manifest['status']}**. Start (UTC): `{manifest['started_utc']}`. ROS 2 {manifest['ros_distro']}; Nav2 {manifest['nav2_version']}; Gazebo {manifest['gazebo_version']}. GUI: `{manifest['gui']}`.

{table}

![Physical navigation and clearance]({link}/navigation.png)

Acceptance requires Nav2 success, actual position/heading errors ≤0.20 m/rad, minimum footprint clearance ≥0.05 m, no sampled body/obstacle overlap, remaining inside the field, a visible detour and stopping after success. Nav2's internal position/heading tolerances are tighter: 0.15 m/rad. All {len(metrics['checks'])} recorded checks passed.

Clearance uses oriented full-body footprint polygons and independent simulator body poses, rather than planned points or robot-center distance. Overlap counts are sampled geometric checks, not Gazebo contact-sensor events. One fixed scene does not establish continuous collision guarantees, statistical reliability or hardware performance.

{video_section}

## Files and execution

| Location | Contents |
|---|---|
| [test_report.ipynb](test_report.ipynb) | Executed replay, trajectory, criteria and artifact checks |
| [testing_code](testing_code/README.md) | Scenario, launch, truth-localization fixture, action client and verification |
| [{link}]({link}) | Metrics, physical trajectory, initial plan, map, world, actual Nav2 parameters, logs and hashes |

{bag_section}

From `Nav2_test/`:

```powershell
.\\testing_code\\Start-Nav2.ps1 -Action test -Gui
.\\testing_code\\Start-Nav2.ps1 -Action sim
```

Windows shortcuts: `C:\\Ubuntu\\Test-Nav2.cmd` runs the automatic GUI test; `C:\\Ubuntu\\Nav2-Sim.cmd` opens the scene. The existing container desktop is at <http://127.0.0.1:6080/vnc.html?autoconnect=1&resize=scale>. Tests use ROS domain 72 and Gazebo partition `well_e_nav2_test`. Each run is preserved separately. ROS logs, builds and caches remain under `C:\\Ubuntu\\ROS`.

## Coverage and next steps

Completed: a single goal, known static-obstacle detour, physical footprint clearance, arrival and stopping. Pending: unmapped-obstacle sensing, realistic localization, ordered waypoints, unreachable goals and explicit cancellation. GP/GRG integration remains in [Integration_test](../Integration_test/README.md), unimplemented.

Earlier failed setup attempts remain in their result directories, with failed status; their logs distinguish clock-QoS and action-acknowledgment setup issues from this successful run. This report displays the latest run without deleting those records.

Reference: [Nav2 Regulated Pure Pursuit](https://docs.nav2.org/rolling/configuration_and_development/configuration_guide/controller_plugins/configuring_regulated_pp/). The executed Jazzy parameter file is saved with the run; installed defaults are adapted to this fixture, not copied from rolling documentation.
'''
    md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
    cells = [md('# Nav2 Point-to-Point Obstacle Test\n\nReplay recorded execution artifacts. This notebook does not start ROS.'),
        md('## Setup\n\nKnown static obstacle and ground-truth localization. No live perception or AMCL is tested.'),
        code(f"""from pathlib import Path
import hashlib, importlib.util, json, sys
from IPython.display import Image, Markdown, display
name = 'Nav2_test'
locations = [p for parent in [Path.cwd(), *Path.cwd().parents]
    for p in [parent, parent/name, parent/'Code/10.27_ROS_simulation'/name]]
root = next(p for p in locations if p.name == name and (p/'result').is_dir())
run = root/'result'/{run.name!r}
manifest = json.loads((run/'manifest.json').read_text())
metrics = json.loads((run/'metrics.json').read_text())
print(json.dumps(manifest['scenario'], indent=2))"""),
        md('## Recorded results\n\n'+table),
        code("display(Image(filename=str(run/'navigation.png')))\nprint(json.dumps(metrics['checks'], indent=2))\nassert metrics['status'] == 'passed' and all(metrics['checks'].values())"),
        md('## Artifact and geometry verification\n\nVerify source/data hashes, analytic geometry fixtures and recomputed footprint clearances.'),
        code("""sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location('nav2_verification', root/'testing_code/tools/verify_results.py')
verification = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verification)
verification.verify()"""),
        md('## Interpretation\n\nNav2 executed a detour around a known obstacle and reached the goal with the simulated Bunker Mini. Ground-truth localization isolates planning/control from localization error. This is not evidence of detecting unmapped obstacles or navigating on hardware. See the [README](README.md) for coverage and commands.')]
    if recording:
        cells[-1:-1] = [md(bag_section), code("""recording = json.loads((run/'rosbag_summary.json').read_text())
replay = json.loads((run/'rosbag_replay_verification.json').read_text())
assert recording['status'] == 'passed' and all(recording['checks'].values())
assert replay['status'] == 'passed'
assert replay['ros_domain_id'] == 73 and not replay['command_topics_replayed']
assert (run/'rosbag/metadata.yaml').is_file()
assert list((run/'rosbag').glob('*.mcap'))
print('rosbag2 MCAP record, readback and isolated playback verified.')""")]
        cells[-1:-1] = [md(video_section), code("""from IPython.display import Video
video = run/'replay/rosbag_replay.mp4'
video_metadata = json.loads((run/'replay/video_metadata.json').read_text())
assert video_metadata['status'] == 'verified'
assert video_metadata['kind'] == 'rosbag_pose_visualization'
assert not video_metadata['camera_footage']
assert hashlib.sha256(video.read_bytes()).hexdigest() == video_metadata['video_sha256']
assert hashlib.sha256((run/'replay/bag_poses.csv').read_bytes()).hexdigest() == video_metadata['pose_csv_sha256']
assert all(hashlib.sha256((run/'rosbag'/name).read_bytes()).hexdigest() == digest
           for name, digest in video_metadata['bag_sha256'].items())
display(Video(filename=str(video), embed=True, width=900,
              html_attributes='controls preload="metadata"'))""")]
    return description, cells


if __name__ == '__main__':
    shared = Path(__file__).resolve().parents[3]/'BunkerMini_drive_test/testing_code/tools/report.py'
    spec = importlib.util.spec_from_file_location('shared_reports', shared)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.argv = [str(shared), '--stage', 'Nav2_test']
    module.main()

"""Render a clearly labelled MP4 from physical poses decoded directly from MCAP."""
import argparse
from bisect import bisect_left
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

TOPIC = '/simulation/ground_truth/odom'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ensure_video(run):
    import imageio_ffmpeg
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter
    from matplotlib.patches import Polygon, Rectangle
    sys.dont_write_bytecode = True
    scripts = Path(__file__).resolve().parents[1]/'src/well_e_nav2_test/scripts'
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from geometry import clearance, footprint, rectangle

    run = Path(run)
    target = run/'replay'
    target.mkdir(exist_ok=True)
    video, metadata_path = target/'rosbag_replay.mp4', target/'video_metadata.json'
    bag_files = sorted((run/'rosbag').glob('*.mcap'))
    if not bag_files:
        raise FileNotFoundError(f'No MCAP recording in {run}')
    bag_hashes = {path.name: sha(path) for path in bag_files}
    manifest = json.loads((run/'manifest.json').read_text())
    scenario = manifest['scenario']
    scenario_hash = hashlib.sha256(json.dumps(scenario, sort_keys=True).encode()).hexdigest()
    source_hash = sha(Path(__file__))
    if metadata_path.exists() and video.exists():
        cached = json.loads(metadata_path.read_text())
        if (cached.get('status') == 'verified' and cached.get('bag_sha256') == bag_hashes
                and cached.get('render_source_sha256') == source_hash
                and cached.get('scenario_sha256') == scenario_hash
                and cached.get('video_sha256') == sha(video)):
            return video, cached

    poses = []
    image_topics = set()
    for path in bag_files:
        with path.open('rb') as stream:
            reader = make_reader(stream, decoder_factories=[DecoderFactory()])
            summary = reader.get_summary()
            if summary:
                for channel in summary.channels.values():
                    schema = summary.schemas.get(channel.schema_id)
                    if schema and schema.name in ('sensor_msgs/msg/Image', 'sensor_msgs/msg/CompressedImage'):
                        image_topics.add(channel.topic)
            for _, _, _, msg in reader.iter_decoded_messages(topics=[TOPIC]):
                p, q, stamp = msg.pose.pose.position, msg.pose.pose.orientation, msg.header.stamp
                theta = math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))
                poses.append((stamp.sec+stamp.nanosec*1e-9, p.x, p.y, theta))
    poses.sort()
    # CSV supplies phase-window boundaries only; every rendered pose comes from MCAP.
    with (run/'trajectory.csv').open() as stream:
        window = [row for row in csv.DictReader(stream) if row['phase'] in ('navigating', 'settling')]
    first, last = float(window[0]['time_s']), float(window[-1]['time_s'])
    poses = [pose for pose in poses if first-1e-8 <= pose[0] <= last+1e-8]
    if len(poses) < 2:
        raise RuntimeError('Bag does not cover the navigation replay window')
    with (target/'bag_poses.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['simulation_time_s', 'x_m', 'y_m', 'yaw_rad'])
        writer.writerows(poses)

    center, size = scenario['obstacle']['center_m'], scenario['obstacle']['size_m']
    obstacle = rectangle(center, size)
    planned = json.loads((run/'planned_path.json').read_text())
    fps = 12
    duration = poses[-1][0]-poses[0][0]
    frame_count = math.ceil(duration*fps)+1
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.add_patch(Rectangle((center[0]-size[0]/2, center[1]-size[1]/2), *size,
                           facecolor='#b44842', alpha=0.85, label='Known obstacle'))
    ax.plot([p[0] for p in planned], [p[1] for p in planned], '--', color='#a2aab3',
            linewidth=1.2, label='Initial Nav2 plan')
    trace, = ax.plot([], [], color='#ed9226', linewidth=2, label='Recorded physical path')
    body = Polygon(footprint(poses[0][1:], scenario['footprint_m']), closed=True,
                   facecolor='#4284b6', edgecolor='#164267', alpha=0.9, label='Bunker Mini footprint')
    ax.add_patch(body)
    heading, = ax.plot([], [], color='white', linewidth=2.5)
    ax.scatter([scenario['start'][0]], [scenario['start'][1]], color='#348343', marker='o', s=45, label='Start')
    ax.scatter([scenario['goal'][0]], [scenario['goal'][1]], color='black', marker='x', s=65, label='Goal')
    xmin, xmax, ymin, ymax = scenario['map_bounds_m']
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax), aspect='equal', xlabel='x (m)', ylabel='y (m)',
           title='Nav2 obstacle test - replay from rosbag physical poses')
    ax.grid(alpha=0.18)
    ax.legend(loc='lower left', fontsize=8)
    status = ax.text(0.02, 0.97, '', va='top', transform=ax.transAxes, fontsize=10,
                     bbox={'facecolor': 'white', 'alpha': 0.9, 'edgecolor': '#cccccc'})
    ax.text(0.5, -0.13, '1x simulation time | Top-down data visualization, not camera footage',
            ha='center', transform=ax.transAxes, fontsize=10)
    times = [pose[0] for pose in poses]
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    matplotlib.rcParams['animation.ffmpeg_path'] = ffmpeg
    encoder = FFMpegWriter(fps=fps, codec='libx264',
        metadata={'title': 'Nav2 rosbag-based pose replay',
                  'comment': 'Generated from recorded odometry; not a camera or Gazebo screen recording.'},
        extra_args=['-pix_fmt', 'yuv420p', '-crf', '23', '-movflags', '+faststart'])
    temporary = target/'rosbag_replay.rendering.mp4'
    try:
        with encoder.saving(fig, str(temporary), dpi=100):
            for frame in range(frame_count):
                t = min(poses[-1][0], poses[0][0]+frame/fps)
                index = min(bisect_left(times, t), len(poses)-1)
                if index and abs(times[index-1]-t) < abs(times[index]-t):
                    index -= 1
                _, x, y, theta = poses[index]
                polygon = footprint((x, y, theta), scenario['footprint_m'])
                body.set_xy(polygon)
                heading.set_data([x, x+0.32*math.cos(theta)], [y, y+0.32*math.sin(theta)])
                trace.set_data([p[1] for p in poses[:index+1]], [p[2] for p in poses[:index+1]])
                error = math.dist((x, y), scenario['goal'][:2])
                status.set_text(f'Time: {t-poses[0][0]:5.1f} s\nGoal error: {error:.2f} m\n'
                                f'Body clearance: {clearance(polygon, obstacle):.2f} m')
                if frame == frame_count//2:
                    fig.savefig(target/'preview.png', dpi=100)
                encoder.grab_frame(facecolor='white')
                if frame % 120 == 0:
                    print(f'Rendering rosbag replay: {frame}/{frame_count} frames', flush=True)
        temporary.replace(video)
    finally:
        plt.close(fig)
    decoded = subprocess.run([ffmpeg, '-v', 'error', '-i', str(video), '-progress', 'pipe:1',
                              '-nostats', '-f', 'null', '-'], capture_output=True, text=True,
                             timeout=60, check=True)
    frames = [int(line.split('=', 1)[1]) for line in decoded.stdout.splitlines() if line.startswith('frame=')]
    if not frames or max(frames) != frame_count:
        raise RuntimeError(f'Encoded/decoded frame count mismatch: expected {frame_count}, got {frames}')
    metadata = {'status': 'verified', 'kind': 'rosbag_pose_visualization', 'camera_footage': False,
                'source_topic': TOPIC, 'source_pose_samples': len(poses),
                'source_simulation_time_range_s': [poses[0][0], poses[-1][0]],
                'recorded_image_topics': sorted(image_topics),
                'frame_selection': 'nearest recorded pose; no position interpolation',
                'fps': fps, 'frame_count': frame_count, 'duration_s': frame_count/fps,
                'playback_speed': 1.0, 'codec': 'H.264', 'width': 1000, 'height': 600,
                'decoded_frames': max(frames), 'video_sha256': sha(video),
                'pose_csv_sha256': sha(target/'bag_poses.csv'),
                'bag_sha256': bag_hashes, 'scenario_sha256': scenario_hash,
                'render_source_sha256': source_hash}
    (target/'decode_verification.log').write_text(decoded.stdout+decoded.stderr, encoding='utf-8')
    metadata_path.write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf-8')
    return video, metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    video, metadata = ensure_video(args.run)
    print(json.dumps({'video': str(video), **metadata}, indent=2))

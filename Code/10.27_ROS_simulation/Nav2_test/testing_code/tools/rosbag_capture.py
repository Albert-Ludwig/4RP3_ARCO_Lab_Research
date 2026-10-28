"""rosbag2 recording configuration and real read/replay verification for the Nav2 test."""
import argparse
from collections import Counter
import csv
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

TOPICS = [
    '/clock', '/tf', '/tf_static', '/map', '/robot_description', '/joint_states',
    '/odom', '/cmd_vel', '/simulation/cmd_vel',
    '/simulation/ground_truth/odom', '/simulation/track_odom', '/simulation/track_tf',
    '/test/navigation_goal', '/plan', '/local_plan', '/received_global_plan',
    '/lookahead_point', '/lookahead_arc',
    '/global_costmap/costmap', '/global_costmap/published_footprint',
    '/local_costmap/costmap', '/local_costmap/published_footprint',
    '/navigate_to_pose/_action/feedback', '/navigate_to_pose/_action/status',
]
REQUIRED = ['/clock', '/tf', '/tf_static', '/map', '/robot_description',
            '/odom', '/cmd_vel', '/simulation/cmd_vel',
            '/simulation/ground_truth/odom', '/test/navigation_goal', '/plan',
            '/navigate_to_pose/_action/status']
REPLAY_TOPICS = ['/tf', '/tf_static', '/map', '/robot_description', '/joint_states',
                 '/odom', '/simulation/ground_truth/odom',
                 '/test/navigation_goal', '/plan', '/global_costmap/costmap', '/local_costmap/costmap']


def save(path, data):
    path.write_text(json.dumps(data, indent=2)+'\n', encoding='utf-8')


def record_command(output, qos):
    return ['ros2', 'bag', 'record', '--storage', 'mcap', '--use-sim-time',
            '--disable-keyboard-controls', '--include-hidden-topics', '--polling-interval', '100',
            '--node-name', 'nav2_test_recorder', '--qos-profile-overrides-path', str(qos),
            '--output', str(output/'rosbag'), '--topics', *TOPICS]


def wait_recorder(process, output):
    deadline = time.monotonic()+15
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError('rosbag2 recorder exited during startup')
        if list((output/'rosbag').glob('*.mcap')):
            return
        time.sleep(0.05)
    raise TimeoutError('rosbag2 storage did not open')


def stop_recorder(process):
    if process is None or process.poll() is not None:
        return False
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=20)
        return process.returncode == 0
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)
        return False


def verify_recording(run):
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    scripts = Path(__file__).resolve().parents[1]/'src/well_e_nav2_test/scripts'
    sys.path.insert(0, str(scripts))
    from geometry import clearance, footprint, rectangle, yaw
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(run/'rosbag'), storage_id='mcap'),
                rosbag2_py.ConverterOptions('', ''))
    types = {topic.name: topic.type for topic in reader.get_all_topics_and_types()}
    counts, frames, truth, goals, statuses, clocks, models = Counter(), set(), [], [], [], [], []
    selected = {'/tf', '/tf_static', '/simulation/ground_truth/odom', '/test/navigation_goal',
                '/navigate_to_pose/_action/status', '/clock', '/robot_description'}
    classes = {topic: get_message(types[topic]) for topic in selected if topic in types}
    while reader.has_next():
        topic, data, _ = reader.read_next()
        counts[topic] += 1
        if topic not in selected:
            continue
        msg = deserialize_message(data, classes[topic])
        if topic in ('/tf', '/tf_static'):
            frames.update((item.header.frame_id, item.child_frame_id) for item in msg.transforms)
        elif topic == '/clock':
            clocks.append(msg.clock.sec+msg.clock.nanosec*1e-9)
        elif topic == '/simulation/ground_truth/odom':
            p = msg.pose.pose.position
            stamp = msg.header.stamp
            truth.append((stamp.sec+stamp.nanosec*1e-9, p.x, p.y, yaw(msg.pose.pose.orientation)))
        elif topic == '/test/navigation_goal':
            goals.append([msg.pose.position.x, msg.pose.position.y, yaw(msg.pose.orientation)])
        elif topic == '/robot_description':
            models.append(msg.data)
        else:
            statuses.extend(item.status for item in msg.status_list)
    metadata = rosbag2_py.Info().read_metadata(str(run/'rosbag'), 'mcap')
    manifest = json.loads((run/'manifest.json').read_text())
    scenario = manifest['scenario']
    metrics = json.loads((run/'metrics.json').read_text())
    with (run/'trajectory.csv').open() as stream:
        rows = [row for row in csv.DictReader(stream) if row['phase'] in ('navigating', 'settling')]
    first, last = float(rows[0]['time_s']), float(rows[-1]['time_s'])
    during = [p for p in truth if first-1e-8 <= p[0] <= last+1e-8]
    obstacle = rectangle(scenario['obstacle']['center_m'], scenario['obstacle']['size_m'])
    bag_clearance = min(clearance(footprint(p[1:], scenario['footprint_m']), obstacle) for p in during)
    by_time = {round(p[0]*1e9): p for p in during}
    matching = sum(round(float(row['time_s'])*1e9) in by_time and
                   math.dist(by_time[round(float(row['time_s'])*1e9)][1:3],
                             [float(row['x_m']), float(row['y_m'])]) < 1e-9 for row in rows)
    checks = {
        'all_required_topics_have_messages': all(counts[topic] > 0 for topic in REQUIRED),
        'message_count_matches_metadata': sum(counts.values()) == metadata.message_count,
        'map_odom_base_and_static_tf_present': {('map', 'odom'), ('odom', 'base_link'),
                                               ('base_link', 'left_track')}.issubset(frames),
        'navigation_goal_matches_scenario': any(math.dist(goal, scenario['goal']) < 1e-9 for goal in goals),
        'successful_action_status_recorded': 4 in statuses,
        'bunker_mini_description_recorded': any('bunker_mini' in model and 'base_link.STL' in model
                                                for model in models),
        'simulation_clock_monotonic': bool(clocks) and all(b >= a for a, b in zip(clocks, clocks[1:])),
        'navigation_interval_covered': during[0][0] <= first+0.05 and during[-1][0] >= last-0.05,
        'physical_pose_matches_analysis_csv': matching/len(rows) >= 0.95,
        'minimum_clearance_matches_analysis': abs(bag_clearance-metrics['minimum_footprint_clearance_m']) < 0.01,
    }
    summary = {'status': 'passed' if all(checks.values()) else 'failed', 'storage_id': 'mcap',
               'bag_directory': 'rosbag', 'message_count': sum(counts.values()),
               'topics': dict(sorted(counts.items())), 'tf_frames': sorted(frames),
               'simulation_clock_range_s': [min(clocks), max(clocks)],
               'navigation_pose_samples': len(during), 'analysis_pose_match_fraction': matching/len(rows),
               'minimum_clearance_from_bag_m': bag_clearance, 'checks': checks}
    save(run/'rosbag_summary.json', summary)
    info = subprocess.run(['ros2', 'bag', 'info', str(run/'rosbag')], capture_output=True, text=True, check=True)
    (run/'rosbag_info.txt').write_text(info.stdout+info.stderr)
    assert all(checks.values()), f'rosbag readback checks failed: {checks}'
    return summary


def verify_playback(run):
    import rclpy
    from nav_msgs.msg import OccupancyGrid, Odometry
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
    from tf2_msgs.msg import TFMessage
    from rosgraph_msgs.msg import Clock
    from std_msgs.msg import String
    from rclpy.qos import qos_profile_sensor_data
    rclpy.init()
    node = Node('nav2_bag_replay_probe')
    observed = Counter()
    frames, last_pose = set(), []
    latched = QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         reliability=ReliabilityPolicy.RELIABLE)
    def receive_truth(msg):
        observed['truth'] += 1
        last_pose[:] = [msg.pose.pose.position.x, msg.pose.pose.position.y]
    def receive_tf(msg):
        observed['tf'] += 1
        frames.update((item.header.frame_id, item.child_frame_id) for item in msg.transforms)
    node.create_subscription(Odometry, '/simulation/ground_truth/odom', receive_truth, 10)
    node.create_subscription(OccupancyGrid, '/map', lambda _: observed.update(['map']), latched)
    node.create_subscription(String, '/robot_description', lambda _: observed.update(['model']), latched)
    node.create_subscription(Clock, '/clock', lambda _: observed.update(['clock']), qos_profile_sensor_data)
    node.create_subscription(TFMessage, '/tf', receive_tf, 10)
    node.create_subscription(TFMessage, '/tf_static', receive_tf, latched)
    command = ['ros2', 'bag', 'play', str(run/'rosbag'), '--rate', '5', '--clock', '100',
               '--delay', '1.0', '--disable-keyboard-controls', '--topics', *REPLAY_TOPICS]
    with (run/'rosbag_replay.log').open('w') as log:
        player = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic()+35
        try:
            while player.poll() is None:
                if time.monotonic() > deadline:
                    raise TimeoutError('rosbag playback timed out')
                rclpy.spin_once(node, timeout_sec=0.02)
            until = time.monotonic()+0.3
            while time.monotonic() < until:
                rclpy.spin_once(node, timeout_sec=0.02)
            final = json.loads((run/'metrics.json').read_text())['final_pose'][:2]
            passed = (player.returncode == 0 and observed['truth'] > 20 and observed['map'] > 0
                      and observed['model'] > 0
                      and observed['clock'] > 0 and {('map', 'odom'), ('odom', 'base_link'),
                          ('base_link', 'left_track')}.issubset(frames)
                      and math.dist(last_pose, final) < 0.03)
            summary = {'status': 'passed' if passed else 'failed', 'ros_domain_id': 73,
                       'playback_rate': 5, 'command_topics_replayed': False,
                       'received_messages': dict(observed), 'final_xy': last_pose,
                       'tf_frames': sorted(frames)}
            save(run/'rosbag_replay_verification.json', summary)
            assert passed, f'rosbag playback check failed: {summary}'
        finally:
            if player.poll() is None:
                os.killpg(player.pid, signal.SIGINT)
                player.wait(timeout=5)
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('--replay-only', action='store_true')
    args = parser.parse_args()
    if args.replay_only:
        verify_playback(args.run)
    else:
        print(json.dumps(verify_recording(args.run), indent=2))
        env = dict(os.environ, ROS_DOMAIN_ID='73')
        subprocess.run(['python3', str(Path(__file__).resolve()), str(args.run), '--replay-only'],
                       env=env, timeout=40, check=True)

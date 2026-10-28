#!/usr/bin/env python3
"""Send one goal to Nav2; evaluate independently recorded physical footprint motion."""
import csv
import json
import math
import os
from pathlib import Path
import time

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, Twist
from lifecycle_msgs.srv import GetState
import rclpy
from nav2_msgs.action import ComputePathToPose, NavigateToPose
from nav_msgs.msg import Odometry
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data, QoSProfile, DurabilityPolicy, ReliabilityPolicy
from rclpy.signals import SignalHandlerOptions
from rosgraph_msgs.msg import Clock

from geometry import clearance, footprint, rectangle, yaw


class NavigationProbe(Node):
    def __init__(self, scenario):
        super().__init__('nav2_obstacle_probe')
        self.set_parameters([Parameter('use_sim_time', value=True)])
        self.scenario = scenario
        self.obstacle = rectangle(scenario['obstacle']['center_m'], scenario['obstacle']['size_m'])
        self.truth = None
        self.sim_time = None
        self.command = Twist()
        self.samples = []
        self.phase = 'startup'
        self.create_subscription(Clock, '/clock', self.clock_cb, qos_profile_sensor_data)
        self.create_subscription(Odometry, '/simulation/ground_truth/odom', self.truth_cb, 10)
        self.create_subscription(Twist, '/simulation/cmd_vel', self.command_cb, 10)
        self.navigator = ActionClient(self, NavigateToPose, '/navigate_to_pose')
        self.planner = ActionClient(self, ComputePathToPose, '/compute_path_to_pose')
        self.goal_record = self.create_publisher(PoseStamped, '/test/navigation_goal',
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                       reliability=ReliabilityPolicy.RELIABLE))

    def clock_cb(self, msg):
        self.sim_time = msg.clock.sec+msg.clock.nanosec*1e-9

    def command_cb(self, msg):
        self.command = msg

    def truth_cb(self, msg):
        self.truth = msg
        p, theta = msg.pose.pose.position, yaw(msg.pose.pose.orientation)
        body = footprint((p.x, p.y, theta), self.scenario['footprint_m'])
        xmin, xmax, ymin, ymax = self.scenario['map_bounds_m']
        stamp = msg.header.stamp
        self.samples.append({
            'time_s': stamp.sec+stamp.nanosec*1e-9, 'phase': self.phase,
            'x_m': p.x, 'y_m': p.y, 'yaw_rad': theta,
            'clearance_m': clearance(body, self.obstacle),
            'body_in_bounds': all(xmin <= x <= xmax and ymin <= y <= ymax for x, y in body),
            'command_linear_m_s': self.command.linear.x,
            'command_angular_rad_s': self.command.angular.z,
        })

    def wait(self, future, seconds=20):
        deadline = time.monotonic()+seconds
        while not future.done():
            if time.monotonic() > deadline:
                raise TimeoutError('ROS request timed out')
            rclpy.spin_once(self, timeout_sec=0.02)
        return future.result()

    def ready(self):
        deadline = time.monotonic()+60
        while self.truth is None or self.sim_time is None:
            if time.monotonic() > deadline:
                raise TimeoutError('Simulation not ready')
            rclpy.spin_once(self, timeout_sec=0.05)
        for name in ('map_server', 'planner_server', 'controller_server', 'bt_navigator'):
            client = self.create_client(GetState, f'/{name}/get_state')
            while time.monotonic() < deadline:
                if client.wait_for_service(timeout_sec=0.2):
                    if self.wait(client.call_async(GetState.Request()), 5).current_state.id == 3:
                        break
                rclpy.spin_once(self, timeout_sec=0.1)
            else:
                raise TimeoutError(f'{name} did not become active')
        assert self.navigator.wait_for_server(timeout_sec=5)
        assert self.planner.wait_for_server(timeout_sec=5)
        if os.environ.get('NAV2_REQUIRE_ROSBAG') == '1':
            deadline = time.monotonic()+10
            while self.goal_record.get_subscription_count() == 0:
                if time.monotonic() > deadline:
                    raise TimeoutError('rosbag recorder did not subscribe to the goal record')
                rclpy.spin_once(self, timeout_sec=0.05)

    def pose(self, values):
        pose = PoseStamped()
        pose.header.frame_id, pose.header.stamp = 'map', self.get_clock().now().to_msg()
        pose.pose.position.x, pose.pose.position.y = float(values[0]), float(values[1])
        pose.pose.orientation.z = math.sin(values[2]/2)
        pose.pose.orientation.w = math.cos(values[2]/2)
        return pose


def main():
    output = Path(os.environ['NAV2_TEST_OUTPUT_DIR'])
    scenario = json.loads(Path(os.environ['NAV2_TEST_SCENARIO']).read_text())
    report = {'status': 'failed', 'case': scenario['case'],
              'assumptions': {'known_static_obstacle_map': True, 'ground_truth_localization': True},
              'criteria': {'position_error_max_m': 0.20, 'yaw_error_max_rad': 0.20,
                           'minimum_clearance_m': scenario['minimum_clearance_m']}}
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    probe = NavigationProbe(scenario)
    handle = None
    try:
        probe.ready()
        probe.phase = 'planning'
        plan = ComputePathToPose.Goal()
        plan.start, plan.goal = probe.pose(scenario['start']), probe.pose(scenario['goal'])
        plan.use_start, plan.planner_id = True, 'GridBased'
        plan_handle = probe.wait(probe.planner.send_goal_async(plan))
        assert plan_handle.accepted, 'Planner rejected the request'
        planned = probe.wait(plan_handle.get_result_async())
        assert planned.status == GoalStatus.STATUS_SUCCEEDED, f'Planning status: {planned.status}'
        points = [[pose.pose.position.x, pose.pose.position.y,
                   yaw(pose.pose.orientation)] for pose in planned.result.path.poses]
        assert len(points) >= 2, 'Planner returned no usable path'
        (output/'planned_path.json').write_text(json.dumps(points, indent=2)+'\n')
        request = NavigateToPose.Goal()
        request.pose = probe.pose(scenario['goal'])
        probe.goal_record.publish(request.pose)
        handle = probe.wait(probe.navigator.send_goal_async(request))
        assert handle.accepted, 'Nav2 rejected the navigation goal'
        probe.phase = 'navigating'
        start = probe.sim_time
        deadline = time.monotonic()+220
        future = handle.get_result_async()
        while not future.done():
            rclpy.spin_once(probe, timeout_sec=0.02)
            if probe.sim_time-start > scenario['max_navigation_sim_s'] or time.monotonic() > deadline:
                probe.wait(handle.cancel_goal_async(), 5)
                raise TimeoutError('Navigation deadline exceeded')
            if probe.samples[-1]['clearance_m'] <= 0:
                probe.wait(handle.cancel_goal_async(), 5)
                raise RuntimeError('Physical footprint overlaps the obstacle')
        result = future.result()
        report['nav2_action_status'] = result.status
        report['nav2_error_code'] = getattr(result.result, 'error_code', None)
        report['navigation_duration_s'] = probe.sim_time-start
        assert result.status == GoalStatus.STATUS_SUCCEEDED, f'Navigation status: {result.status}'
        probe.phase = 'settling'
        finish = probe.sim_time+1.5
        while probe.sim_time < finish:
            rclpy.spin_once(probe, timeout_sec=0.02)
        position = probe.truth.pose.pose.position
        final_yaw = yaw(probe.truth.pose.pose.orientation)
        target = scenario['goal']
        error = math.hypot(position.x-target[0], position.y-target[1])
        yaw_error = abs(math.atan2(math.sin(final_yaw-target[2]), math.cos(final_yaw-target[2])))
        samples = [row for row in probe.samples if row['phase'] in ('navigating', 'settling')]
        length = sum(math.hypot(b['x_m']-a['x_m'], b['y_m']-a['y_m'])
                     for a, b in zip(samples, samples[1:]))
        min_clearance = min(row['clearance_m'] for row in samples)
        overlap_count = sum(row['clearance_m'] <= 0 for row in samples)
        stopped = (math.hypot(probe.truth.twist.twist.linear.x,
                             probe.truth.twist.twist.linear.y) < 0.03
                   and abs(probe.truth.twist.twist.angular.z) < 0.03)
        report.update(final_pose=[position.x, position.y, final_yaw],
                      final_position_error_m=error, final_yaw_error_rad=yaw_error,
                      minimum_footprint_clearance_m=min_clearance,
                      sampled_footprint_overlap_count=overlap_count,
                      actual_path_length_m=length,
                      planned_path_length_m=sum(math.dist(a[:2], b[:2]) for a, b in zip(points, points[1:])),
                      max_lateral_detour_m=max(abs(row['y_m']) for row in samples),
                      stopped_after_success=stopped, physical_pose_samples=len(samples))
        checks = {'nav2_succeeded': True, 'physical_goal_position': error <= 0.20,
                  'physical_goal_heading': yaw_error <= 0.20,
                  'no_sampled_body_overlap': overlap_count == 0,
                  'minimum_clearance': min_clearance >= scenario['minimum_clearance_m'],
                  'body_remained_in_field': all(row['body_in_bounds'] for row in samples),
                  'detour_executed': report['max_lateral_detour_m'] > 1.2,
                  'stopped_after_success': stopped}
        report['checks'] = checks
        assert all(checks.values()), f'Acceptance checks failed: {checks}'
        report['status'] = 'passed'
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        if handle is not None and report['status'] != 'passed':
            handle.cancel_goal_async()
        (output/'metrics.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
        if probe.samples:
            with (output/'trajectory.csv').open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(probe.samples[0]))
                writer.writeheader()
                writer.writerows(probe.samples)
        print(json.dumps(report, indent=2), flush=True)
        probe.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

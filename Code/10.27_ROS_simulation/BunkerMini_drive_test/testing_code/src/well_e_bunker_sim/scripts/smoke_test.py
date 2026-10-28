#!/usr/bin/env python3
"""Exercise ROS commands against physics motion, independently of drive odometry."""
import csv
import json
import math
import os
from pathlib import Path
import time

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from tf2_msgs.msg import TFMessage


def xy(msg):
    return (msg.pose.pose.position.x, msg.pose.pose.position.y)


def distance(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])


def yaw(msg):
    q = msg.pose.pose.orientation
    return math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))


class Probe(Node):
    def __init__(self):
        super().__init__('bunker_mini_motion_probe')
        self.sim_time = None
        self.odom = None
        self.raw_odom = None
        self.truth = None
        self.transform = None
        self.last_truth = 0.0
        self.last_odom = 0.0
        self.last_tf = 0.0
        self.phase = 'initializing'
        self.samples = []
        self.create_subscription(Clock, '/clock', self.clock_cb, qos_profile_sensor_data)
        self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        self.create_subscription(Odometry, '/simulation/track_odom', self.raw_odom_cb, 10)
        self.create_subscription(Odometry, '/simulation/ground_truth/odom', self.truth_cb, 10)
        self.create_subscription(TFMessage, '/tf', self.tf_cb, 10)
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 1)

    def clock_cb(self, msg):
        self.sim_time = msg.clock.sec + msg.clock.nanosec*1e-9

    def odom_cb(self, msg):
        self.odom, self.last_odom = msg, time.monotonic()

    def raw_odom_cb(self, msg):
        self.raw_odom = msg

    def truth_cb(self, msg):
        self.truth, self.last_truth = msg, time.monotonic()
        stamp = msg.header.stamp
        self.samples.append({
            'time_s': stamp.sec+stamp.nanosec*1e-9, 'phase': self.phase,
            'truth_x_m': xy(msg)[0], 'truth_y_m': xy(msg)[1], 'truth_yaw_rad': yaw(msg),
            'odom_x_m': xy(self.odom)[0] if self.odom else None,
            'odom_y_m': xy(self.odom)[1] if self.odom else None,
            'odom_yaw_rad': yaw(self.odom) if self.odom else None,
            'raw_odom_yaw_rad': yaw(self.raw_odom) if self.raw_odom else None,
        })

    def tf_cb(self, msg):
        for transform in msg.transforms:
            if transform.header.frame_id == 'odom' and transform.child_frame_id == 'base_link':
                self.transform, self.last_tf = transform, time.monotonic()

    def wait_ready(self):
        deadline = time.monotonic() + 40
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            if (self.sim_time is not None and self.odom is not None
                    and self.truth is not None and self.transform is not None
                    and self.publisher.get_subscription_count() > 0):
                return
        raise RuntimeError('Timed out waiting for /clock, /odom, ground truth, TF, and command subscriber')

    def run_for(self, seconds, linear=None, angular=0.0, phase='stop'):
        self.phase = phase
        start, wall_deadline = self.sim_time, time.monotonic() + max(20, seconds*10)
        while self.sim_time-start < seconds:
            if time.monotonic() > wall_deadline:
                raise RuntimeError('Simulation clock stalled or is too slow')
            if linear is not None:
                msg = Twist()
                msg.linear.x, msg.angular.z = linear, angular
                self.publisher.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.02)
        now = time.monotonic()
        if any(now-t > 1.0 for t in (self.last_odom, self.last_truth, self.last_tf)):
            raise RuntimeError('Odometry, ground truth or TF is stale')


def main():
    rclpy.init()
    probe = Probe()
    report = {'status': 'failed', 'stage': 'tracked-motion-interface', 'checks': {}}
    default_output = Path(os.environ.get('ROS_HOME', '/ros-data')) / 'reports/bunker_mini_smoke'
    directory = Path(os.environ.get('BUNKER_TEST_OUTPUT_DIR', str(default_output)))
    output = directory / 'metrics.json'
    try:
        probe.wait_ready()
        probe.run_for(1.0, 0.0, phase='settle')
        p0, o0 = xy(probe.truth), xy(probe.odom)
        probe.run_for(3.0, 0.2, phase='forward')
        displacement = distance(p0, xy(probe.truth))
        odom_displacement = distance(o0, xy(probe.odom))
        report['checks']['forward_ground_truth_m'] = displacement
        report['checks']['forward_odometry_m'] = odom_displacement
        assert 0.3 < displacement < 0.9, f'Unexpected physical forward motion: {displacement}'
        assert probe.truth.pose.pose.position.x-p0[0] > 0.3, 'Forward command has wrong axis/sign'
        assert 0.3 < odom_displacement < 0.9, 'Drive odometry does not reflect forward travel'
        probe.run_for(1.0, 0.0)
        initial_yaw = yaw(probe.truth)
        initial_odom_yaw = yaw(probe.odom)
        probe.run_for(3.0, 0.0, 0.4, phase='turn')
        rotation = math.atan2(math.sin(yaw(probe.truth)-initial_yaw),
                              math.cos(yaw(probe.truth)-initial_yaw))
        report['checks']['positive_rotation_ground_truth_rad'] = rotation
        relative_error = abs(rotation/1.2-1)
        odom_rotation = math.atan2(math.sin(yaw(probe.odom)-initial_odom_yaw),
                                   math.cos(yaw(probe.odom)-initial_odom_yaw))
        odom_yaw_error = abs(odom_rotation-rotation)
        report['checks']['turn_relative_error'] = relative_error
        report['checks']['turn_odometry_error_rad'] = odom_yaw_error
        assert relative_error < 0.05, f'Physical turn error exceeds 5%: {relative_error}'
        assert odom_yaw_error < 0.05, f'Odometry turn error exceeds 0.05 rad: {odom_yaw_error}'
        probe.run_for(1.0, 0.0)
        before_reverse = xy(probe.truth)
        heading = yaw(probe.truth)
        probe.run_for(2.0, -0.15, phase='reverse')
        delta = (xy(probe.truth)[0]-before_reverse[0], xy(probe.truth)[1]-before_reverse[1])
        reverse = delta[0]*math.cos(heading)+delta[1]*math.sin(heading)
        report['checks']['reverse_ground_truth_m'] = reverse
        assert -0.5 < reverse < -0.15, f'Unexpected reverse motion: {reverse}'
        # Stop publishing entirely: verify expiry using independent physical pose.
        probe.run_for(2.0, phase='watchdog_expiry')
        stopped = xy(probe.truth)
        probe.run_for(1.0, phase='watchdog_stopped')
        drift = distance(stopped, xy(probe.truth))
        report['checks']['watchdog_stopped_drift_m'] = drift
        assert drift < 0.03, f'Robot did not stop after command timeout: {drift}'
        assert probe.odom.header.frame_id == 'odom'
        assert probe.odom.child_frame_id == 'base_link'
        t = probe.transform.transform.translation
        tf_error = distance((t.x, t.y), xy(probe.odom))
        report['checks']['odom_tf_translation_error_m'] = tf_error
        assert tf_error < 0.03, f'Odometry and TF disagree: {tf_error}'
        report['status'] = 'passed'
        report['simulation_time_s'] = probe.sim_time
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        probe.publisher.publish(Twist())
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
        if probe.samples:
            with (directory / 'trajectory.csv').open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(probe.samples[0]))
                writer.writeheader()
                writer.writerows(probe.samples)
        print(json.dumps(report, indent=2), flush=True)
        print(f'Report: {output}', flush=True)
        probe.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

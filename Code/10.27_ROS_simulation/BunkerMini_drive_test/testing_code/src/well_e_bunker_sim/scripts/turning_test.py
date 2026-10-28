#!/usr/bin/env python3
"""Measure steering gain on independent maneuvers; retain raw physical observations."""
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import time

import rclpy
from geometry_msgs.msg import Twist

from smoke_test import Probe, xy, yaw


def change_efficiency(probe, value):
    process = subprocess.Popen(['gz', 'topic', '-t', '/model/bunker_mini/steering_efficiency',
                                '-m', 'gz.msgs.Double', '-p', f'data: {value}'],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    deadline = time.monotonic()+8
    while process.poll() is None:
        if time.monotonic() > deadline:
            process.kill()
            process.communicate()
            raise TimeoutError('Efficiency command timed out')
        rclpy.spin_once(probe, timeout_sec=0.02)
    stdout, stderr = process.communicate()
    if process.returncode:
        raise RuntimeError(stderr.decode() or stdout.decode())


def main():
    rclpy.init()
    probe = Probe()
    output = Path(os.environ['BUNKER_TEST_OUTPUT_DIR'])
    report = {'status': 'failed', 'stage': 'steering-diagnostics', 'cases': []}
    try:
        probe.wait_ready()
        for efficiency in (1.0, 0.624):
            probe.run_for(0.8, 0.0)
            change_efficiency(probe, efficiency)
            probe.run_for(0.4, 0.0)
            for name, linear, angular in [('left', 0.0, 0.4), ('right', 0.0, -0.4),
                                          ('arc_left', 0.15, 0.3), ('arc_right', 0.15, -0.3)]:
                probe.run_for(0.5, 0.0)
                start_time, start_yaw = probe.sim_time, yaw(probe.truth)
                start_odom_yaw = yaw(probe.raw_odom or probe.odom)
                start_position = xy(probe.truth)
                first_sample = len(probe.samples)
                probe.run_for(3.0, linear, angular, phase=f'{efficiency}_{name}')
                duration = probe.sim_time-start_time
                rotation = math.atan2(math.sin(yaw(probe.truth)-start_yaw),
                                      math.cos(yaw(probe.truth)-start_yaw))
                raw_odom_yaw = yaw(probe.raw_odom or probe.odom)
                odom_rotation = math.atan2(math.sin(raw_odom_yaw-start_odom_yaw),
                                           math.cos(raw_odom_yaw-start_odom_yaw))
                # Discard the first/last 0.5 s when estimating steady motion gain.
                middle = [row for row in probe.samples[first_sample:]
                          if start_time+0.5 <= row['time_s'] <= start_time+2.5]
                steady_rate = (middle[-1]['truth_yaw_rad']-middle[0]['truth_yaw_rad']) / (
                    middle[-1]['time_s']-middle[0]['time_s'])
                # Cases begin close to zero yaw here; reject a wrap before using a slope.
                assert abs(middle[-1]['truth_yaw_rad']-middle[0]['truth_yaw_rad']) < math.pi
                report['cases'].append({
                    'name': name, 'steering_efficiency': efficiency,
                    'linear_m_s': linear, 'angular_rad_s': angular, 'duration_s': duration,
                    'expected_rotation_rad': angular*duration,
                    'physical_rotation_rad': rotation, 'raw_odometry_rotation_rad': odom_rotation,
                    'relative_turn_error': abs(rotation/(angular*duration)-1),
                    'steady_yaw_rate_rad_s': steady_rate,
                    'measured_track_to_body_yaw_gain': efficiency*steady_rate/angular,
                    'physical_chord_m': math.dist(start_position, xy(probe.truth)),
                })
                print(json.dumps(report['cases'][-1]), flush=True)
        report['status'] = 'passed'
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        probe.publisher.publish(Twist())
        (output/'metrics.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
        if probe.samples:
            with (output/'trajectory.csv').open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(probe.samples[0]))
                writer.writeheader()
                writer.writerows(probe.samples)
        probe.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

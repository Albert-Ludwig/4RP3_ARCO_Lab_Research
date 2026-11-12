#!/usr/bin/env python3
"""Validate the frozen steering gain on commands not used to estimate it."""
import csv
import json
import math
import os
from pathlib import Path

import rclpy
from geometry_msgs.msg import Twist

from smoke_test import Probe, xy, yaw


def angle(value):
    return math.atan2(math.sin(value), math.cos(value))


def main():
    rclpy.init()
    probe = Probe()
    output = Path(os.environ['BUNKER_TEST_OUTPUT_DIR'])
    report = {'status': 'failed', 'stage': 'independent-steering-validation', 'cases': [],
              'criteria': {'turn_relative_error_max': 0.10, 'odometry_yaw_error_max_rad': 0.05,
                           'odometry_position_error_max_m': 0.05},
              'note': 'Fixed gain; no parameter changes or ground-truth feedback during validation.'}
    cases = [('slow_left', 0.0, 0.2, 2.5), ('slow_right', 0.0, -0.2, 2.5),
             ('fast_left', 0.0, 0.6, 2.0), ('fast_right', 0.0, -0.6, 2.0),
             ('wide_arc_left', 0.2, 0.2, 3.0), ('wide_arc_right', 0.2, -0.2, 3.0)]
    try:
        probe.wait_ready()
        probe.run_for(1.0, 0.0)
        assert probe.raw_odom is not None, 'Raw track odometry missing'
        for name, linear, angular, seconds in cases:
            probe.run_for(0.8, 0.0)
            start_time = probe.sim_time
            initial_truth, initial_odom, initial_raw = [yaw(msg) for msg in
                (probe.truth, probe.odom, probe.raw_odom)]
            probe.run_for(seconds, linear, angular, phase=name)
            duration = probe.sim_time-start_time
            physical = angle(yaw(probe.truth)-initial_truth)
            corrected = angle(yaw(probe.odom)-initial_odom)
            raw = angle(yaw(probe.raw_odom)-initial_raw)
            expected = angular*duration
            case = {'name': name, 'linear_m_s': linear, 'angular_rad_s': angular,
                    'duration_s': duration, 'expected_rotation_rad': expected,
                    'physical_rotation_rad': physical,
                    'corrected_odometry_rotation_rad': corrected, 'raw_odometry_rotation_rad': raw,
                    'turn_relative_error': abs(physical/expected-1),
                    'odometry_yaw_error_rad': abs(angle(corrected-physical)),
                    'odometry_position_error_m': math.dist(xy(probe.truth), xy(probe.odom))}
            case['passed'] = (case['turn_relative_error'] < 0.10
                              and case['odometry_yaw_error_rad'] < 0.05
                              and case['odometry_position_error_m'] < 0.05)
            report['cases'].append(case)
            print(json.dumps(case), flush=True)
        assert all(case['passed'] for case in report['cases']), 'Independent maneuver checks failed'
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

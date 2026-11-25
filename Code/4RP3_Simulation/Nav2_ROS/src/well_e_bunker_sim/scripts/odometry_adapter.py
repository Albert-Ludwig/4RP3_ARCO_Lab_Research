#!/usr/bin/env python3
"""Integrate calibrated track velocities; never consume simulator ground truth."""
import copy
import math

import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from tf2_ros import TransformBroadcaster


class TrackOdometry(Node):
    def __init__(self):
        super().__init__('bunker_mini_track_odometry')
        self.scale = self.declare_parameter('yaw_scale', 0.624).value
        if not math.isfinite(self.scale) or self.scale <= 0:
            raise ValueError('yaw_scale must be finite and positive')
        self.x = self.y = self.yaw = 0.0
        self.previous = None
        self.publisher = self.create_publisher(Odometry, '/odom', 10)
        self.broadcaster = TransformBroadcaster(self)
        self.create_subscription(Odometry, '/simulation/track_odom', self.receive, 10)

    def receive(self, raw):
        stamp = raw.header.stamp
        t = stamp.sec+stamp.nanosec*1e-9
        v, w = raw.twist.twist.linear.x, raw.twist.twist.angular.z*self.scale
        if not all(math.isfinite(value) for value in (t, v, w)):
            return
        if self.previous is not None:
            old_t, old_v, old_w = self.previous
            dt = t-old_t
            if dt < 0:  # Gazebo time reset starts a new odometry origin.
                self.x = self.y = self.yaw = 0.0
            elif dt > 0:
                mean_v, mean_w = (v+old_v)/2, (w+old_w)/2
                dtheta = mean_w*dt
                half = dtheta/2
                sinc = math.sin(half)/half if abs(half) > 1e-9 else 1.0
                ds = mean_v*dt*sinc
                self.x += ds*math.cos(self.yaw+half)
                self.y += ds*math.sin(self.yaw+half)
                self.yaw = math.atan2(math.sin(self.yaw+dtheta), math.cos(self.yaw+dtheta))
        self.previous = (t, v, w)
        msg = copy.deepcopy(raw)
        msg.header.frame_id, msg.child_frame_id = 'odom', 'base_link'
        msg.pose.pose.position.x, msg.pose.pose.position.y = self.x, self.y
        msg.pose.pose.position.z = 0.0
        q = msg.pose.pose.orientation
        q.x = q.y = 0.0
        q.z, q.w = math.sin(self.yaw/2), math.cos(self.yaw/2)
        msg.twist.twist.angular.z = w
        # Input covariance is unspecified. This adapter does not estimate uncertainty.
        self.publisher.publish(msg)
        transform = TransformStamped()
        transform.header, transform.child_frame_id = msg.header, msg.child_frame_id
        transform.transform.translation.x, transform.transform.translation.y = self.x, self.y
        transform.transform.rotation = q
        self.broadcaster.sendTransform(transform)


def main():
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = TrackOdometry()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

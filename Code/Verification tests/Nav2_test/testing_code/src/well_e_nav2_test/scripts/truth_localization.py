#!/usr/bin/env python3
"""Simulation oracle for map -> odom; explicitly replaces localization in this test."""
import math

import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from tf2_ros import TransformBroadcaster

from geometry import yaw


class TruthLocalization(Node):
    def __init__(self):
        super().__init__('simulation_truth_localization')
        self.odom = None
        self.broadcaster = TransformBroadcaster(self)
        self.create_subscription(Odometry, '/odom', self.receive_odom, 10)
        self.create_subscription(Odometry, '/simulation/ground_truth/odom', self.receive_truth, 10)

    def receive_odom(self, msg):
        self.odom = msg

    def receive_truth(self, msg):
        if self.odom is None:
            return
        odom = self.odom
        offset = yaw(msg.pose.pose.orientation)-yaw(odom.pose.pose.orientation)
        c, s = math.cos(offset), math.sin(offset)
        estimated, actual = odom.pose.pose.position, msg.pose.pose.position
        transform = TransformStamped()
        # AMCL similarly publishes a short future-dated correction for TF consumers.
        transform.header.stamp = (rclpy.time.Time.from_msg(msg.header.stamp)+
                                  Duration(seconds=0.15)).to_msg()
        transform.header.frame_id, transform.child_frame_id = 'map', 'odom'
        transform.transform.translation.x = actual.x-c*estimated.x+s*estimated.y
        transform.transform.translation.y = actual.y-s*estimated.x-c*estimated.y
        transform.transform.rotation.z = math.sin(offset/2)
        transform.transform.rotation.w = math.cos(offset/2)
        self.broadcaster.sendTransform(transform)


def main():
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = TruthLocalization()
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

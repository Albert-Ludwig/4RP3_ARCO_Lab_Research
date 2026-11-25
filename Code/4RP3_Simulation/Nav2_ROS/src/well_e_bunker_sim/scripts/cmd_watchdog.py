#!/usr/bin/env python3
"""Simulation-only Twist interface: bounded commands and wall-clock expiry."""
import math
import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.clock import Clock, ClockType
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions


class CommandWatchdog(Node):
    def __init__(self):
        super().__init__('bunker_mini_sim_watchdog')
        self.timeout = self.declare_parameter('command_timeout_s', 0.5).value
        self.linear_limit = self.declare_parameter('max_linear_m_s', 0.35).value
        self.angular_limit = self.declare_parameter('max_angular_rad_s', 0.6).value
        if self.timeout <= 0 or self.linear_limit <= 0 or self.angular_limit <= 0:
            raise ValueError('Timeout and velocity limits must be positive')
        self.command = Twist()
        self.received = float('-inf')
        self.publisher = self.create_publisher(Twist, '/simulation/cmd_vel', 1)
        self.create_subscription(Twist, '/cmd_vel', self.receive, 1)
        self.create_timer(0.05, self.publish, clock=Clock(clock_type=ClockType.STEADY_TIME))

    def receive(self, msg):
        self.command = Twist()
        self.received = time.monotonic()
        if not all(math.isfinite(v) for v in (msg.linear.x, msg.angular.z)):
            return
        self.command.linear.x = max(-self.linear_limit, min(self.linear_limit, msg.linear.x))
        self.command.angular.z = max(-self.angular_limit, min(self.angular_limit, msg.angular.z))

    def publish(self):
        cmd = self.command if time.monotonic() - self.received <= self.timeout else Twist()
        self.publisher.publish(cmd)


def main():
    # Let Python raise KeyboardInterrupt, then close the context in this thread.
    # The default ROS handler can otherwise invalidate a wait set during shutdown.
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = CommandWatchdog()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if rclpy.ok():
            node.publisher.publish(Twist())
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

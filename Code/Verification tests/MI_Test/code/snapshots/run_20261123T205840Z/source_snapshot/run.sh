#!/usr/bin/env bash
set -eo pipefail
source /opt/ros/jazzy/setup.bash
source /ros-data/build/10.27_ROS_simulation/Nav2_test/install/setup.bash
code='/workspace/Code/Verification tests/MI_Test/code'
colcon --log-base /tmp/ros_mi_closed_loop_logs build --base-paths "$code/src" --build-base /tmp/ros_mi_closed_loop_build --install-base /tmp/ros_mi_closed_loop_install --event-handlers console_direct+
source /tmp/ros_mi_closed_loop_install/setup.bash
python3 "$code/run.py"

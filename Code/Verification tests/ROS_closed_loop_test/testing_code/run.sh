#!/usr/bin/env bash
set -eo pipefail
source /opt/ros/jazzy/setup.bash
source /ros-data/build/10.27_ROS_simulation/Nav2_test/install/setup.bash
code="/workspace/Code/Verification tests/ROS_closed_loop_test/testing_code"
colcon --log-base /tmp/cl_logs build --base-paths "$code/src" --build-base /tmp/cl_build --install-base /tmp/cl_install --event-handlers console_direct+
source /tmp/cl_install/setup.bash
python3 "$code/run.py"
/home/ubuntu/.venvs/srf-verification/bin/python "$code/audit.py"

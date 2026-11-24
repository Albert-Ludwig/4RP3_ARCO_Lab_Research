#!/usr/bin/env bash
set -eo pipefail
source /opt/ros/jazzy/setup.bash
source /ros-data/build/10.27_ROS_simulation/Nav2_test/install/setup.bash
code='/workspace/Code/Verification tests/WIVR_test/code'
colcon --log-base /tmp/ros_wivr_sampling_logs build --base-paths "$code/src" --build-base /tmp/ros_wivr_sampling_build --install-base /tmp/ros_wivr_sampling_install --event-handlers console_direct+
source /tmp/ros_wivr_sampling_install/setup.bash
python3 "$code/run.py"

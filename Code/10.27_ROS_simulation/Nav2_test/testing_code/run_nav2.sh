#!/usr/bin/env bash
set -eo pipefail
export PYTHONDONTWRITEBYTECODE=1
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=72
export GZ_PARTITION=well_e_nav2_test
export ROS_HOME="${ROS_HOME:-/ros-data}"
export ROS_LOG_DIR="$ROS_HOME/log"
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
drive="$root/../../BunkerMini_drive_test/testing_code"
runtime="$ROS_HOME/build/10.27_ROS_simulation/Nav2_test"
python3 "$drive/tools/fetch_asset.py"
colcon --log-base "$ROS_HOME/log/colcon_nav2_test" build \
  --base-paths "$drive/src" "$root/src" --packages-select well_e_bunker_sim well_e_nav2_test \
  --build-base "$runtime/build" --install-base "$runtime/install" --event-handlers console_direct+
source "$runtime/install/setup.bash"
exec ros2 launch well_e_nav2_test nav2_test.launch.py "$@"

#!/usr/bin/env bash
set -eo pipefail
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=71
export GZ_PARTITION=well_e_bunker_mini
export ROS_HOME="${ROS_HOME:-/ros-data}"
export ROS_LOG_DIR="$ROS_HOME/log"
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
runtime="$ROS_HOME/build/10.27_ROS_simulation/BunkerMini_drive_test"
mkdir -p "$runtime" "$ROS_LOG_DIR"
python3 "$root/tools/fetch_asset.py"
colcon --log-base "$ROS_HOME/log/colcon_bunker_drive" build \
  --base-paths "$root/src" --packages-select well_e_bunker_sim \
  --build-base "$runtime/build" --install-base "$runtime/install" \
  --event-handlers console_direct+
source "$runtime/install/setup.bash"
exec ros2 launch well_e_bunker_sim sim.launch.py "$@"

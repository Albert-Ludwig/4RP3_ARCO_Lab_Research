#!/usr/bin/env bash
set -eo pipefail
source /opt/ros/jazzy/setup.bash
source /ros-data/build/10.27_ROS_simulation/Nav2_test/install/setup.bash
export ROS_HOME=/ros-data
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}"
exec /home/ubuntu/.venvs/policy-compare/bin/python "$@"

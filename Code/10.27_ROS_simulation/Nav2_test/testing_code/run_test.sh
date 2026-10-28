#!/usr/bin/env bash
set -eo pipefail
export PYTHONDONTWRITEBYTECODE=1
source /opt/ros/jazzy/setup.bash
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$root/tools/run_test.py" "$@"

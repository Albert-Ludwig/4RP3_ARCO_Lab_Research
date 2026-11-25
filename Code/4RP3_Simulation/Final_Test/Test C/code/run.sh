#!/usr/bin/env bash
set -euo pipefail

code_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON_BIN:-python}" "$code_dir/../../code/run.py" C "$@"

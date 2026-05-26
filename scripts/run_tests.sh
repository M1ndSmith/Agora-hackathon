#!/usr/bin/env bash
# Avoid ROS launch_testing pytest plugins on systems with /opt/ros installed.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
exec .venv/bin/python -m pytest -q tests/ "$@"

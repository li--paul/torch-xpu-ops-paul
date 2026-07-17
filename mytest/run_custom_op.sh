#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/../.venv/bin/activate"

exec python "${SCRIPT_DIR}/test_custom_op.py" "$@"

#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/../.venv/bin/activate"
exec python "${SCRIPT_DIR}/test_single_p2p.py" "$@"

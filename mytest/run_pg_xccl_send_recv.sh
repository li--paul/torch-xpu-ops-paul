#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

export LD_LIBRARY_PATH="/home/lm/paul2/torch-xpu-ops/.venv/lib/python3.12/site-packages/torch/lib:/opt/intel/oneapi/2026.0/lib"
export CCL_CONFIGURATION=cpu_gpu_dpcpp

rm -f /tmp/pg_demo

"${SCRIPT_DIR}/test_pg_xccl_send_recv" 0 2 &
PID0=$!
sleep 2
"${SCRIPT_DIR}/test_pg_xccl_send_recv" 1 2
PID1=$!
wait $PID0 $PID1

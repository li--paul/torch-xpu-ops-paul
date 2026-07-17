#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

export LD_LIBRARY_PATH="/opt/intel/oneapi/ccl/2022.0/lib:/opt/intel/oneapi/mpi/2021.18/lib:$LD_LIBRARY_PATH"
export CCL_CONFIGURATION=cpu_gpu_dpcpp

exec mpirun -np 2 --bind-to none "${SCRIPT_DIR}/test_single_p2p" "$@"

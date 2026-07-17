#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

export LD_LIBRARY_PATH="/opt/intel/oneapi/2026.0/lib:/opt/intel/oneapi/ccl/2022.0/lib:/opt/intel/oneapi/mpi/2021.18/lib"
export CCL_CONFIGURATION=cpu_gpu_dpcpp

exec /opt/intel/oneapi/mpi/2021.18/bin/mpirun -np 2 --bind-to none "${SCRIPT_DIR}/test_all_reduce_coalesced" "$@"

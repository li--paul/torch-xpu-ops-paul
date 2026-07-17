#!/bin/bash
set -euo pipefail
# Usage: ./run_all_reduce.sh [num_gpus] [m] [n] [k]

NUM_GPUS="${1:-4}"
M="${2:-4096}"
N="${3:-4096}"
K="${4:-4096}"

echo "=== Tensor Parallel All-Reduce Demo ==="
echo "  GPUs:      ${NUM_GPUS}"
echo "  Matrix:    [${M}, ${N}] x [${N}, ${K}]"
echo "  Precision: BF16"
echo "  Backend:   xccl (oneCCL)"
echo ""

cd "$(dirname "$0")"
source ../.venv/bin/activate

torchrun --nproc_per_node="${NUM_GPUS}" \
    all_reduce_demo.py \
    --m "${M}" --n "${N}" --k "${K}" \
    --warmup 5 --iter 20

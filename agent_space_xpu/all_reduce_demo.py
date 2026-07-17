"""
Tensor parallelism with all-reduce across multiple XPU cards (oneCCL/XCCL).

Row-parallel linear: weight W [n, k] split along dim 0 into shards W_i.
Each GPU: partial_i = X[:, i*n/p:(i+1)*n/p] @ W_i, then all-reduce sums
Y = sum_i partial_i = X @ W (verified against full matmul).

Usage:
  torchrun --nproc_per_node=N all_reduce_demo.py [--m M --n N --k K]
"""

import os, argparse, socket, torch
import torch.distributed as dist


def init_process(local_rank, world_size):
    # Use ephemeral port to avoid stale rendezvous conflicts
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        os.environ.setdefault('MASTER_PORT', str(s.getsockname()[1]))
    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    torch.xpu.set_device(local_rank)
    dist.init_process_group(backend='xccl', rank=local_rank, world_size=world_size)


def run(local_rank, world_size, m, n, k, num_warmup, num_iter):
    init_process(local_rank, world_size)
    rank = dist.get_rank()
    dtype = torch.bfloat16
    device = torch.device(f'xpu:{local_rank}')
    assert n % world_size == 0
    n_pr = n // world_size

    if rank == 0:
        print(f'X [{m},{n}] @ W [{n},{k}]  |  {world_size} GPUs  |  '
              f'{n_pr} rows/GPU  |  BF16  |  backend=xccl', flush=True)

    # Init data deterministically
    torch.manual_seed(42)
    inp = torch.randn(m, n, dtype=dtype, device=device)
    torch.manual_seed(99 + rank)
    W_shard = torch.randn(n_pr, k, dtype=dtype, device=device)

    # Verification: all-reduce result vs full matmul (for ≤4 GPUs)
    if world_size <= 4:
        W_shards = [torch.zeros(n_pr, k, dtype=dtype, device=device) for _ in range(world_size)]
        dist.all_gather(W_shards, W_shard)
        W_full = torch.cat(W_shards, dim=0)
        y = torch.matmul(inp[:, rank * n_pr:(rank + 1) * n_pr], W_shard)
        dist.all_reduce(y)
        y_ref = torch.matmul(inp, W_full)
        err = (y - y_ref).abs().max().item()
        rel = err / (y_ref.abs().max().item() + 1e-8)
        if rank == 0:
            print(f'  Verify: abs_err={err:.2e} rel_err={rel:.2e} '
                  f'({"PASS" if rel < 0.05 else "FAIL"})', flush=True)

    # Warmup
    for _ in range(num_warmup):
        y = torch.matmul(inp[:, rank * n_pr:(rank + 1) * n_pr], W_shard)
        dist.all_reduce(y)
    torch.xpu.synchronize(device)
    dist.barrier()

    if rank == 0:
        ev_start = torch.xpu.Event(enable_timing=True)
        ev_end = torch.xpu.Event(enable_timing=True)
        ev_start.record()
    for _ in range(num_iter):
        y = torch.matmul(inp[:, rank * n_pr:(rank + 1) * n_pr], W_shard)
        dist.all_reduce(y)
    if rank == 0:
        ev_end.record()
    torch.xpu.synchronize(device)

    if rank == 0:
        t = ev_start.elapsed_time(ev_end) / num_iter
        total_flops = 2.0 * m * n * k / (t / 1000)
        bw = 2.0 * m * k * 2 / (t / 1000) / 1e9
        print(f'  Latency: {t:.3f} ms  |  {total_flops/1e12:.2f} TFLOPS  |  '
              f'~{bw:.1f} GB/s', flush=True)

    dist.destroy_process_group()
    if rank == 0:
        print(f'  Done on {world_size} GPU(s).', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--m', type=int, default=4096)
    p.add_argument('--n', type=int, default=4096)
    p.add_argument('--k', type=int, default=4096)
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--iter', type=int, default=20)
    args = p.parse_args()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    run(local_rank, world_size, args.m, args.n, args.k, args.warmup, args.iter)

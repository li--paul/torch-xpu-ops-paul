"""Minimal XCCL all-reduce test."""
import os, socket, torch
import torch.distributed as dist

rank = int(os.environ['RANK'])
ws = int(os.environ['WORLD_SIZE'])
lr = int(os.environ['LOCAL_RANK'])

# Use an ephemeral port that's the SAME for all ranks
# (torchrun sets MASTER_PORT=29400, but if that's stale, use a fresh one)
port = int(os.environ.get('MY_MASTER_PORT', os.environ.get('MASTER_PORT', '29400')))
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = str(port)

torch.xpu.set_device(lr)
dist.init_process_group(backend='xccl', rank=rank, world_size=ws)

x = torch.ones(1, dtype=torch.float32, device=f'xpu:{lr}') * rank
dist.all_reduce(x)
expected = ws * (ws - 1) / 2.0

if rank == 0:
    print(f'all_reduce sum(0..{ws-1}) = {x.item():.0f} '
          f'(expected {expected:.0f}) {"PASS" if x.item() == expected else "FAIL"}', flush=True)

y = torch.randn(1024, 1024, dtype=torch.bfloat16, device=f'xpu:{lr}')
dist.all_reduce(y)

dist.destroy_process_group()
if rank == 0:
    print('XCCL basic communication OK', flush=True)

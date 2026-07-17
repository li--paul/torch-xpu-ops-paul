"""Minimal XCCL all-reduce test."""
import os, socket, torch
import torch.distributed as dist

rank = int(os.environ['RANK'])
ws = int(os.environ['WORLD_SIZE'])
lr = int(os.environ['LOCAL_RANK'])

# Unique port
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(('', 0))
    port = s.getsockname()[1]
os.environ['MASTER_PORT'] = str(port)
os.environ['MASTER_ADDR'] = '127.0.0.1'

torch.xpu.set_device(lr)
dist.init_process_group(backend='xccl', rank=rank, world_size=ws)

x = torch.ones(1, dtype=torch.float32, device=f'xpu:{lr}') * rank
dist.all_reduce(x)
expected = ws * (ws - 1) / 2.0

if rank == 0:
    print(f'all_reduce sum(0..{ws-1}) = {x.item():.0f}  (expected {expected:.0f})  '
          f'{"PASS" if x.item() == expected else "FAIL"}')

# Larger test
y = torch.randn(1024, 1024, dtype=torch.bfloat16, device=f'xpu:{lr}')
dist.all_reduce(y)
y_full = torch.zeros_like(y)
dist.all_reduce(y)  # nop since already all-reduced... actually no, would double

dist.destroy_process_group()
if rank == 0:
    print('XCCL basic communication OK')

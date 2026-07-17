import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def worker(rank, world_size, file_name):
    store = dist.FileStore(file_name, world_size)
    dist.init_process_group(
        backend="xccl", rank=rank, world_size=world_size, store=store,
    )
    device = torch.device("xpu", rank)

    tensors = [
        torch.full((60 + i,), float(rank + 1 + i), device=device, dtype=torch.float32)
        for i in range(5)
    ]

    # test_all_reduce_coalesced_xccl
    all_reduce_coalesced_outs = [t.clone() for t in tensors]
    dist.all_reduce_coalesced(all_reduce_coalesced_outs, group=dist.group.WORLD)

    ok_arc = True
    for i, t in enumerate(all_reduce_coalesced_outs):
        expected = world_size * (i + (world_size + 1.0) / 2.0)
        if not torch.allclose(t, torch.full_like(t, expected)):
            ok_arc = False
            print(f"[rank {rank}] all_reduce_coalesced FAIL at tensor {i}: "
                  f"got {t[0].item():.1f}, expected {expected:.1f}")

    # test_all_reduce_coalesced_manager_xccl
    manager_outs = [t.clone() for t in tensors]
    with dist._coalescing_manager(group=dist.group.WORLD, device=device, async_ops=True) as cm:
        for tensor in manager_outs:
            dist.all_reduce(tensor)
    cm.wait()

    ok_cm = True
    for i, t in enumerate(manager_outs):
        expected = world_size * (i + (world_size + 1.0) / 2.0)
        if not torch.allclose(t, torch.full_like(t, expected)):
            ok_cm = False
            print(f"[rank {rank}] coalescing_manager FAIL at tensor {i}: "
                  f"got {t[0].item():.1f}, expected {expected:.1f}")

    if ok_arc:
        print(f"[rank {rank}] PASS all_reduce_coalesced")
    if ok_cm:
        print(f"[rank {rank}] PASS coalescing_manager")

    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 2
    file_name = "/tmp/test_xccl_arc"
    mp.spawn(worker, args=(world_size, file_name), nprocs=world_size)

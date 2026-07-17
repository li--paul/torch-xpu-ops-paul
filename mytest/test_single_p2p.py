import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def worker(rank, world_size, file_name):
    store = dist.FileStore(file_name, world_size)
    dist.init_process_group(
        backend="xccl",
        rank=rank,
        world_size=world_size,
        store=store,
    )
    torch.manual_seed(42)
    device = f"xpu:{rank}"
    send_tensor = torch.rand(10, 10, device=device)
    if rank == 0:
        dist.send(send_tensor, dst=1)
        print(f"[rank0] sent:\n{send_tensor}")
    elif rank == 1:
        recv_tensor = torch.rand(10, 10, device=device)
        dist.recv(recv_tensor, src=0)
        print(f"[rank1] received:\n{recv_tensor}")
        # same seed → send_tensor on both ranks should be identical
        if torch.equal(send_tensor, recv_tensor):
            print("[rank1] PASS: send_tensor == recv_tensor")
        else:
            print("[rank1] FAIL: mismatch")
    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 2
    file_name = "/tmp/test_xccl_p2p"
    mp.spawn(worker, args=(world_size, file_name), nprocs=world_size)

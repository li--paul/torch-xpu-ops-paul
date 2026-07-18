"""
Demonstrate ProcessGroupXCCL::send/recv via torch.distributed.

C++ equivalent:
    auto pg = c10::make_intrusive<ProcessGroupXCCL>(store, rank, size);
    std::vector<at::Tensor> tensors = {tensor};
    auto work = pg->send(tensors, dst, tag);   // or pg->recv(tensors, src, tag)
    work->wait();
"""
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def worker(rank, world_size, file_name):
    # C++: auto store = c10::make_intrusive<FileStore>(path, world_size);
    store = dist.FileStore(file_name, world_size)

    # C++: auto pg = c10::make_intrusive<ProcessGroupXCCL>(store, rank, size);
    dist.init_process_group(backend="xccl", rank=rank, world_size=world_size, store=store)
    pg = dist.distributed_c10d._get_default_group()
    device = f"xpu:{rank}"

    if rank == 0:
        # C++: pg->send({tensor}, dstRank=1, tag=0)->wait();
        tensor = torch.full((10,), 42.0, device=device)
        dist.send(tensor, dst=1)
        print(f"[rank {rank}] sent: {tensor}")
    else:
        # C++: pg->recv({tensor}, srcRank=0, tag=0)->wait();
        tensor = torch.empty((10,), device=device)
        dist.recv(tensor, src=0)

        expected = torch.full((10,), 42.0, device=device)
        if tensor.equal(expected):
            print(f"[rank {rank}] PASS: received tensor matches")
        else:
            print(f"[rank {rank}] FAIL: got {tensor}, expected {expected}")

    # C++: dist->destroy_process_group();
    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = min(torch.xpu.device_count(), 2)
    file_name = "/tmp/test_pg_xccl_send_recv"
    mp.spawn(worker, args=(world_size, file_name), nprocs=world_size)
    try:
        os.remove(file_name)
    except OSError:
        pass

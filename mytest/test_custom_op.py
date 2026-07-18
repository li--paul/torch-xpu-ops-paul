import os
import torch
import torch.multiprocessing as mp

_script_dir = os.path.dirname(os.path.abspath(__file__))

def worker(rank):
    with torch.xpu.device(rank):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "custom_op", os.path.join(_script_dir, "custom_op.so"))
        custom = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(custom)

        t = torch.tensor([1.0, 2.0, 3.0], device=f"xpu:{rank}")
        r = custom.whoami(t)
        print(f"[rank {rank}] {r}")

        out = custom.add_one(t)
        expected = t + 1
        if torch.equal(out, expected):
            print(f"[rank {rank}] PASS add_one")
        else:
            print(f"[rank {rank}] FAIL add_one: {out} != {expected}")

if __name__ == "__main__":
    nprocs = min(torch.xpu.device_count(), 2)
    mp.spawn(worker, args=(), nprocs=nprocs)

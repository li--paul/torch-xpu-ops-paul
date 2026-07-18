import os
import sys
import time
import signal
import tempfile
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

_script_dir = os.path.dirname(os.path.abspath(__file__))
failures = []

_TEST_FUNCTIONS = []

def _reg(fn):
    _TEST_FUNCTIONS.append(fn)
    return fn

def ok(msg=""):
    print(f"  PASS {msg}")

def fail(msg):
    print(f"  FAIL {msg}")
    failures.append(msg)

# ----------------------------------------------------------------
def _run_worker(rank, file_name, world_size, test_idx):
    test_fn = _TEST_FUNCTIONS[test_idx]
    store = dist.FileStore(file_name, world_size)
    with torch.xpu.device(rank):
        test_fn(rank, world_size, store)
    try:
        os.remove(file_name)
    except OSError:
        pass

def _run_test(test_idx, world_size):
    name = _TEST_FUNCTIONS[test_idx].__name__
    file_name = os.path.join(tempfile.gettempdir(), f"test_pg_xccl_{os.getpid()}_{test_idx}")
    try:
        mp.spawn(_run_worker, args=(file_name, world_size, test_idx), nprocs=world_size)
    except Exception as e:
        print(f"  Exception: {e}")
        return False
    finally:
        try:
            os.remove(file_name)
        except OSError:
            pass
    return True

# ----------------------------------------------------------------
@_reg
def test_file_store_check(rank, world_size, store):
    dist.init_process_group(backend="xccl", rank=rank, world_size=world_size, store=store)
    pg = dist.distributed_c10d._get_default_group()
    assert pg.rank() == rank and pg.size() == world_size
    ok(f"rank={rank} size={world_size}")
    dist.destroy_process_group()

@_reg
def test_send_recv_non_dense_tensor(rank, world_size, store):
    dist.init_process_group(backend="xccl", rank=rank, world_size=world_size, store=store)
    device = f"xpu:{rank}"
    full = torch.empty((64, 64), device=device).fill_(rank)
    block = full[:, 16:32]
    try:
        if rank == 0:
            dist.send(block, dst=1)
        else:
            dist.recv(block, src=0)
        fail("expected ValueError for non-dense tensor")
    except (ValueError, RuntimeError):
        ok("non-dense tensor correctly rejected")
    dist.destroy_process_group()

@_reg
def test_set_process_group_desc(rank, world_size, store):
    dist.init_process_group(backend="xccl", rank=rank, world_size=world_size, store=store,
                            device_id=torch.device(f"xpu:{rank}"))
    pg = dist.distributed_c10d._get_default_group()
    assert pg.group_desc == "default_pg", f"got {pg.group_desc}"
    pg1 = dist.new_group([0, 1], group_desc="test_purpose")
    assert pg1.group_desc == "test_purpose", f"got {pg1.group_desc}"
    pg2 = dist.new_group([0, 1])
    assert pg2.group_desc == "undefined", f"got {pg2.group_desc}"
    ok()
    dist.destroy_process_group()

@_reg
def test_nan_check_non_float(rank, world_size, store):
    os.environ["TORCH_XCCL_NAN_CHECK"] = "1"
    dist.init_process_group(backend="xccl", rank=rank, world_size=world_size, store=store)
    device = f"xpu:{rank}"
    for dtype in [torch.uint8, torch.int8, torch.int32, torch.int64, torch.bool]:
        t = torch.ones(3, 4, dtype=dtype, device=device)
        if dtype != torch.bool:
            t = t * rank
        try:
            dist.broadcast(t, 0)
            dist.all_gather_into_tensor(
                torch.empty(world_size, *t.shape, dtype=dtype, device=device), t)
            if rank == 0:
                dist.send(t, 1)
            else:
                dist.recv(t, 0)
            dist.barrier()
        except Exception as e:
            fail(f"dtype {dtype} raised {e}")
    ok("non-float dtypes all passed")
    dist.destroy_process_group()
    os.environ["TORCH_XCCL_NAN_CHECK"] = "0"

@_reg
def test_nan_check_empty_tensor(rank, world_size, store):
    os.environ["TORCH_XCCL_NAN_CHECK"] = "1"
    dist.init_process_group(backend="xccl", rank=rank, world_size=world_size, store=store)
    t = torch.empty((0,), dtype=torch.float32, device=f"xpu:{rank}")
    try:
        dist.broadcast(t, 0)
        dist.barrier()
        ok()
    except Exception as e:
        fail(f"empty tensor raised {e}")
    dist.destroy_process_group()
    os.environ["TORCH_XCCL_NAN_CHECK"] = "0"

@_reg
def test_close_multi_pg_unordered(rank, world_size, store):
    dist.init_process_group(backend="xccl", rank=rank, world_size=world_size, store=store)
    device = f"xpu:{rank}"
    pg = dist.distributed_c10d._get_default_group()
    t = torch.rand(10, 10, device=device)
    pg.allreduce(t).wait()
    new_pg1 = dist.new_group([0, 1])
    new_pg2 = dist.new_group([0, 1])
    t1 = torch.rand(10, 10, device=device)
    t2 = torch.rand(10, 10, device=device)
    new_pg1.allreduce(t1).wait()
    new_pg2.allreduce(t2).wait()
    if rank == 0:
        dist.destroy_process_group(new_pg2); del new_pg2
        dist.destroy_process_group(new_pg1); del new_pg1
    else:
        dist.destroy_process_group(new_pg1); del new_pg1
        dist.destroy_process_group(new_pg2); del new_pg2
    dist.destroy_process_group()
    ok()

# test_nan_assert_float32 skipped:
# TORCH_XCCL_NAN_CHECK triggers SIGABRT on detecting NaN (expected behavior),
# which kills the process before a Python exception can be caught.

@_reg
def test_oom(rank, world_size, store):
    dist.init_process_group(backend="xccl", rank=rank, world_size=world_size, store=store)
    device = f"xpu:{rank}"
    torch.xpu.set_device(device)
    shape = (16384 * 2, 16384 * 2)
    weight = torch.ones(shape, device=device).half()
    gradient = torch.zeros(shape, device=device).half()
    ret = torch.randn(shape, device=device).half()
    for _ in range(50):
        output = torch.empty_like(ret)
        output = ret + weight + gradient
        ret = torch.nn.functional.linear(output, weight=ret)
        dist.all_reduce(ret, op=dist.ReduceOp.SUM)
    torch.xpu.synchronize()
    assert torch.xpu.max_memory_allocated() < torch.xpu.max_memory_reserved() * 2
    ok("memory usage within bounds")
    dist.destroy_process_group()

# ----------------------------------------------------------------
if __name__ == "__main__":
    world_size = min(torch.xpu.device_count(), 2)

    for idx, fn in enumerate(_TEST_FUNCTIONS):
        print(f"\n--- {fn.__name__} ---")
        passed = _run_test(idx, world_size)
        if not passed:
            print(f"  FAIL (process exited with error)")
            failures.append(fn.__name__)

    if failures:
        print(f"\n{len(failures)} test(s) FAILED: {failures}")
        sys.exit(1)
    else:
        print("\nAll tests PASSED")

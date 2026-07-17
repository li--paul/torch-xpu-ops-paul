# test_c10d_xccl.py 分析

**文件**: `test/xpu/distributed/test_c10d_xccl.py` (1785 行)

## 总体结构

6 个测试类 + 1 个枚举:

### 1. `RendezvousEnvTest` (L115–212)
- 单进程, 测试 `c10d.rendezvous("env://")` 对环境变量的依赖
- 验证缺失 `WORLD_SIZE` / `RANK` / `MASTER_ADDR` / `MASTER_PORT` 时的异常

### 2. `ProcessGroupXCCLTest` (L215–481)
- 继承 `MultiProcessTestCase`, 自动派生 `world_size` 个子进程
- 使用 `FileStore`(临时文件) 做 rendezvous
- 测试内容:
  - `test_send_recv_non_dense_tensor`: 非连续 tensor 的 P2P 报 `ValueError`
  - `test_close_multi_pg_unordered`: 多 PG 乱序销毁不 crash
  - `test_file_store_check`: FileStore 方式初始化
  - `test_nan_assert`: `TORCH_XCCL_NAN_CHECK=1` 时 NaN 触发 SIGABRT
  - `test_nan_check_non_float`: 非浮点类型 no-op
  - `test_oom`: 重复 all_reduce 后验证内存分配正常

### 3. `CommTest` (L483–849)
- 通信操作测试, world_size=2
- `test_single_p2p`: send/recv 基本 P2P
- `test_xccl_barrier`: barrier + sub-group
- `test_all_reduce_coalesced_*`: 合并 all_reduce
- `test_reduce_scatter_*`: reduce_scatter
- `test_tensor_dtype_complex`: complex dtype all_gather
- `test_all_gather_into_tensor`: FP8 + FP32 all_gather
- `test_unwaited` / `test_wait_tensor`: async collective 生命周期
- `test_pass_xccl_options_high_priority_stream`: 高优先级 stream 选项

### 4. `XCCLTraceTest` (L948–1430)
- 通过 `_dump_xccl_trace()` 获取飞行记录 pickle
- 验证 trace 格式: `profiling_name`, `input_sizes`, `collective_seq_id` 等

### 5. `SymmetricMemoryTest` (L1450–1659)
- SYCL IPC 对称内存测试
- `test_rendezvous_basic`, `test_get_signal_pad`, `test_subgroup`, `test_put_wait_signal`
- 融合 op: `fused_all_gather_matmul`, `fused_matmul_reduce_scatter`

### 6. `MicroPipelineTPXpuTest` (L1668–1772)
- 单进程, `FakeStore` + `backend="fake"`, 不实际通信
- 验证 `torch.compile` 能否融合 `all_gather + matmul` / `matmul + reduce_scatter`

## 关键工作机制

```python
# 多进程启动 (MultiProcessTestCase)
store = c10d.FileStore(self.file_name, self.world_size)
c10d.init_process_group("xccl", store=store, rank=self.rank, world_size=self.world_size)

# GPU 分配: rank i → GPU i (当 world_size ≤ nGPUs 时)
def init_multigpu_helper(world_size, backend):
    return {i: [i] for i in range(world_size)}
```

## 本地测试结果

### `test_single_p2p.py`

基于 `CommTest.test_single_p2p` 的独立测试程序，使用 `mp.spawn` + `FileStore` 启动 2 个进程分别在 xpu:0 和 xpu:1 上做 send/recv。

**运行命令**:
```bash
python mytest/test_single_p2p.py
```

**结果**: ✅ PASS

```
[rank0] sent: tensor([[0.6130, ...]], device='xpu:0')
[rank1] received: tensor([[0.6130, ...]], device='xpu:1')
[rank1] PASS: send_tensor == recv_tensor
```

### 关键发现

| 启动方式 | Store | 结果 |
|---------|-------|------|
| `mp.spawn` + `FileStore` | ✅ FileStore | ✅ 正常工作 |
| `torchrun` + `init_process_group(backend="xccl")` | ❌ env:// TCPStore | ❌ hang |

`mp.spawn` 方式避开了 `torchrun` 引入的代理进程与 oneCCL ATL/OFI 传输层之间的初始化冲突。`FileStore`（本地文件）替代 `TCPStore`（网络）做 rendezvous 也能正常工作。

## SYCL C++ 独立 P2P 测试

### `test_single_p2p.cpp`

自包含的 SYCL C++ 程序，使用 oneCCL C++ API (`ccl::send` / `ccl::recv`) 直接在 XPU 设备上做 GPU P2P 传输，不依赖 Python 或 PyTorch。

**核心流程**:
```
ccl::init() → MPI_Init → sycl::queue(devices[rank]) → KVS rendezvous
  → ccl::create_communicator → ccl::create_stream
  → ccl::send / ccl::recv → sycl::free → MPI_Finalize
```

**构建**:
```bash
make -C mytest
# icpx -fsycl test_single_p2p.cpp -I$CCL_ROOT/include -I$MPI_ROOT/include \
#       -L$CCL_ROOT/lib -L$MPI_ROOT/lib -lccl -lmpi -o test_single_p2p
```

**运行**:
```bash
./mytest/run_sycl_p2p.sh
# mpirun -np 2 ./test_single_p2p
```

**输出**:
```
[rank 0] sending from GPU Intel(R) Arc(TM) Pro B60 Graphics
[rank 1] sending from GPU Intel(R) Arc(TM) Pro B60 Graphics
[rank 0] PASS
[rank 1] PASS
```

### 关键发现

| 角度 | Python + PyTorch | SYCL C++ 独立 |
|------|-----------------|--------------|
| 启动方式 | `mp.spawn` + `FileStore` | `mpirun` + `MPI_Bcast` KVS |
| 通信 API | `dist.send/recv` | `ccl::send/recv` |
| 设备绑定 | 通过 `torch.xpu.set_device()` | `sycl::device::get_devices()[rank]` |
| GPU 内存 | PyTorch Tensor (USM 自动管理) | `sycl::malloc_device` |
| 结果 | ✅ PASS | ✅ PASS |

两种方式最终都调用同一个底层路径：**oneCCL C++ API (`ccl::send`)** → `libccl.so` → Level Zero → Intel GPU。

## 自定义 XPU 算子示例

### 问题: 算子实现如何获取 Tensor 对应的 SYCL queue？

当 PyTorch 算子收到一个 `device='xpu:1'` 的 tensor 时，需要通过以下链路拿到正确的 `sycl::queue`:

```
Tensor (device=xpu:1)
  │  tensor.device().index() → 1
  ▼
c10::xpu::getCurrentXPUStream(dev_idx)
  │  streams[1][NORMAL][stream_id] → sycl::queue*
  ▼
sycl::queue::submit(cgh)  →  GPU 设备 1 上执行
```

### `custom_op.cpp` — 自定义算子演示

两个 demo op:

**`whoami`** — 打印 tensor 所在设备和 SYCL queue 信息:
```cpp
std::string whoami(const at::Tensor& input) {
  auto device = input.device();
  int dev_idx = device.index();
  auto& queue = c10::xpu::getCurrentXPUStream(dev_idx).queue();
}
```

**`add_one`** — 用 SYCL kernel 对 tensor 逐元素加 1:
```cpp
at::Tensor add_one(const at::Tensor& input) {
  int dev_idx = input.device().index();
  auto& queue = c10::xpu::getCurrentXPUStream(dev_idx).queue();
  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
      out_ptr[i] = in_ptr[i] + 1.0f;
    });
  });
}
```

**构建**:
```bash
make -f mytest/Makefile.custom_op -C mytest
```

**运行**:
```bash
./mytest/run_custom_op.sh
```

**输出**:
```
[rank 0] device=xpu:0, sycl::queue=0x284e1900, backend=Level Zero
[rank 0] PASS add_one
[rank 1] device=xpu:1, sycl::queue=0x2bb12010, backend=Level Zero
[rank 1] PASS add_one
```

### 关键点

- **`c10::xpu::getCurrentXPUStream(device_index)`** 是标准入口，定义在 `c10/xpu/XPUStream.h`
- `queue()` 返回预创建的 `sycl::queue`，已绑定到对应设备的 `sycl::device` 和 `sycl::context`
- 使用 `icpx -fsycl` 编译，链接 `libtorch_xpu.so` + `libc10_xpu.so`
- 由于本机主进程 Level Zero 驱动状态可能退化，测试通过 `mp.spawn` 在子进程中运行

## SYCL C++ all_reduce_coalesced 测试

### `test_all_reduce_coalesced.cpp`

使用 `ccl::group_start()/group_end()` 将 5 个不同 size 的 `ccl::allreduce` 合并为一次批量 kernel launch。

**核心逻辑**:
```cpp
// 创建 5 个 USM buffer, size 分别为 [60, 61, 62, 63, 64]
// 每个 buffer 填充值: rank + 1 + i

// 合并调用: group_start/end 包裹
ccl::group_start();
for (int i = 0; i < 5; ++i)
    ccl::allreduce(bufs[i], bufs[i], sizes[i], ccl::datatype::int32,
                   ccl::reduction::sum, comm, stream);
ccl::group_end();

// 验证: 每个元素 = world_size * (i + (world_size + 1.0) / 2.0)
```

**构建与运行**:
```bash
make -C mytest test_all_reduce_coalesced
./mytest/run_sycl_all_reduce_coalesced.sh
```

**输出**:
```
[rank 0] PASS coalesced allreduce
[rank 1] PASS coalesced allreduce
```

### 与 Python 测试的对应关系

| Python | SYCL C++ |
|--------|----------|
| `dist.all_reduce_coalesced(tensors)` | `ccl::group_start()` + N×`ccl::allreduce` + `ccl::group_end()` |
| `_coalescing_manager` context manager | 同上（Python 侧收集 op，最终也是 group_start/end） |

两种方式最终都调用 `ccl::allreduce` 在 GPU 上执行 SUM 规约。oneCCL 的 `ccl::group_start/end` 确保多个 allreduce 作为一次批量操作提交，在底层合并 kernel launch，减少驱动开销。

## `dist.send()` 底层调用链分析

从 Python 到 GPU 驱动的完整调用链：

### 1. Python → c10d 调度

```
dist.send(tensor, dst=1)
  └─ distributed_c10d.py:send() → isend() → group.send([tensor], dst, tag)
        │
    c10d operator dispatch (XPU)
        │
    Register.cpp:send_XPU() → ProcessGroup::getBackend(XPU)->send()
        │
    ProcessGroupXCCL::send()     [src/xccl/ProcessGroupXCCL.cpp:1071]
        │
    pointToPoint(tensor, lambda, dst, ...)
        ├─ initXCCLComm()       创建/获取 P2P 通信子
        │   key = "0:1", p2pRank = (self_rank < peer ? 0 : 1)
        ├─ syncStream()         同步当前 stream → xccl stream
        └─ OnecclGroupGuard (onecclGroupStart/End)
              └─ fn → xccl::onecclSend()     [src/xccl/xccl.cpp:112]
```

### 2. `xccl::onecclSend()` — torch-xpu-ops 封装

```cpp
// src/xccl/xccl.cpp
void onecclSend(at::Tensor& input, onecclComm_t& comm,
                const int dstRank, at::xpu::XPUStream& stream) {
    auto xcclDataType = getXcclDataType(input.scalar_type());  // kFloat → onecclFloat32
    onecclSend(input.data_ptr(),                // SYCL USM 设备指针
               (size_t)input.numel(),
               xcclDataType,
               dstRank,
               comm,
               &stream.queue());                // sycl::queue*
}
```

### 3. oneCCL V2 C API — 插件调度层

```c
// src/api.cpp (uxlfoundation/oneCCL)
onecclResult_t onecclSend(const void *sendbuff, size_t count,
                          onecclDataType_t datatype, int peer,
                          onecclComm_t comm, void *stream) {
    INIT_CCL;              // dlopen 加载 libccl_legacy.so
    return comm->send(sendbuff, count, datatype, peer, comm, stream);
    //       ↑ 函数指针，由 init_communicator 赋值
}
```

**插件架构**：V2 C API 是**薄封装**，运行时通过 `dlopen` 按平台评分自动选择插件：

| 插件 | 库名 | 用途 |
|------|------|------|
| `onecclLegacy` | `libccl_legacy.so` | GPU + CPU 完整实现 |
| `onecclLegacyCPU` | `libccl_legacy_cpu.so` | 仅 CPU |
| `onecclNull` | `libccl_null.so` | 空操作（测试用） |

插件选择由 `oneccl_platform_score_impl()` 按 `sycl::platform::get_platforms()` 中 Level Zero 设备数打分决定。

### 4. Legacy Plugin 实现 — 核心逻辑

```cpp
// plugins/legacy/ccl_legacy.cpp
onecclResult_t oneccl_send_impl(const void *sendbuff, size_t count,
                                onecclDataType_t datatype, int peer,
                                onecclComm_t comm, void *stream) {
    return execute_collective(
        sendbuff, comm, stream,
        // ★ GPU 路径: stream 就是 sycl::queue*
        [=](CommunicatorLegacy *comm_legacy, sycl::queue *sycl_ext) {
            ccl::send(sendbuff, count, convert(datatype), peer,
                      comm_legacy->comm,
                      get_stream(sycl_ext),  // sycl::queue → ccl::stream (thread_local 缓存)
                      attrs, get_deps(sycl_ext));
        },
        // CPU 路径: stream == nullptr 时的 fallback
        [=](CommunicatorLegacy *comm_legacy) {
            return ccl::send(sendbuff, count, convert(datatype),
                             peer, comm_legacy->comm);
        });
}
```

**`execute_collective` 分发逻辑**：

```cpp
auto cpu_buffer = is_host_pointer(buf);  // 用 msync 系统调用检测
auto *queue = static_cast<sycl::queue *>(stream);

if (!cpu_buffer) {
    gpu_collective(comm_legacy, queue);   // GPU 路径: 提交到 SYCL queue
} else {
    queue->wait();
    auto event = cpu_collective(comm_legacy);  // CPU 路径
    queue->submit([=](sycl::handler &cgh) {
        cgh.host_task([=]() { event->wait(); });
    });
}
```

### 5. `ccl::send` — oneCCL C++ API

声明在 `include/oneapi/ccl/api_functions.hpp`:

```cpp
event CCL_API send(void* buf, size_t count, datatype dtype, int peer,
                   const communicator& comm, const stream& stream,
                   const pt2pt_attr& attr = default_pt2pt_attr,
                   const vector_class<event>& deps = {});
```

- `stream` 包装了 `sycl::queue`，通过 `get_stream(sycl_ext)` 从 `thread_local` map 缓存获取
- 实际实现在编译好的 `libccl.so` 中

### 6. 通信子（P2P Communicator）的初始化

在 `pointToPoint()` 中，每对 rank 创建一个 2-rank oneCCL 通信子:

```cpp
// key = "lowRank:highRank" 如 "0:1"
int p2pRank = self_rank <= peer ? 0 : 1;

initXCCLComm(key, device, opType, p2pRank, isSendRecvSelf) 内部:
  1. onecclGetUniqueId(&xcclID)       // 生成唯一 ID
  2. store->set/get(xcclID)           // 通过 KV store 广播 ID
  3. onecclSetDevice(device.index())  // 绑定 XPU 设备
  4. onecclCommInitRank(&comm, 2, xcclID, rank)  // 2-rank 通信子
```

`onecclSetDevice()` 内部通过 `sycl::platform::get_platforms()` 查找 Level Zero 后端，创建 `sycl::queue`:

```cpp
onecclResult_t oneccl_set_device_impl(uint32_t index) {
    for (auto &p : sycl::platform::get_platforms())
        if (p.get_backend() == sycl::backend::ext_oneapi_level_zero)
            l0_platform = p;
    selected_device = l0_platform.get_devices()[index];
    selected_context = l0_platform.ext_oneapi_get_default_context();
    default_stream = ccl::create_stream(
        sycl::queue(*selected_context, *selected_device, in_order));
}
```

### 7. GPU 数据传输

`ccl::send` 内部（`libccl.so`）的实际数据传输机制：

```
             sycl::queue* (from XPUStream)
                    │
        ccl::send internally:
                    │
          ┌─────────┴──────────┐
          │ 同 device          │ 同 node 异 device       │ 异 node
          │ sycl::memcpy       │ Level Zero IPC P2P     │ ATL/OFI 传输层
          │ (设备内拷贝)        │ zeMemGetIpcHandle       │ libfabric / MPI
          │                    │ zeCommandListMemCopy    │
          └────────────────────┴─────────────────────────┘
```

### 全栈总览

```
┌────────────────────────────────────────────────┐
│ Python: torch.distributed.send()               │
├────────────────────────────────────────────────┤
│ torch-xpu-ops:                                 │
│   src/xccl/Register.cpp        - 算子调度      │
│   src/xccl/ProcessGroupXCCL.cpp - PG + P2P     │
│   src/xccl/xccl.cpp           - onecclSend 封装 │
├────────────────────────────────────────────────┤
│ oneCCL V2 C API (libccl.so.2.0)               │
│   src/api.cpp                 - 插件调度入口    │
├────────────────────────────────────────────────┤
│ oneCCL Legacy Plugin (libccl_legacy.so)        │
│   plugins/legacy/ccl_legacy.cpp                │
│     ├─ execute_collective()   - GPU/CPU 分发   │
│     └─ ccl::send()           - oneCCL C++ API  │
├────────────────────────────────────────────────┤
│ oneCCL Core (libccl.so)                        │
│   内部实现: SYCL queue / Level Zero IPC        │
├────────────────────────────────────────────────┤
│ Level Zero Driver (zeDriver)                   │
│   zeCommandListAppendMemoryCopy                │
├────────────────────────────────────────────────┤
│ Intel GPU Kernel Driver (xe)                   │
└────────────────────────────────────────────────┘
```

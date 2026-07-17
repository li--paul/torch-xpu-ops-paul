# XPU 多卡 All-Reduce 张量并行验证

## 环境搭建

```bash
# 创建虚拟环境（需先安装 uv）
uv venv -p 3.12
source .venv/bin/activate

# 安装 PyTorch XPU 版（含 oneCCL 分布式后端）
uv pip install torch --index-url https://download.pytorch.org/whl/xpu
```

验证安装：

```bash
python -c "import torch; print(f'XPU: {torch.xpu.is_available()}, devices: {torch.xpu.device_count()}')"
```

## 程序说明

`all_reduce_demo.py` 实现**行并行张量并行 (Row-Parallel Linear) + All-Reduce**。

### 数学原理

将权重矩阵 `W[n, k]` 沿行方向切分为 `p` 个分片（`p` = GPU 数量），每张卡持有 `W_i[n/p, k]`：

```
Y = X @ W = X @ [W_0; W_1; ...; W_{p-1}]
          = sum_i X[:, i*n/p:(i+1)*n/p] @ W_i
```

每张卡计算自己的分片 `partial_i = X[:, i*n/p:(i+1)*n/p] @ W_i`，然后通过 `dist.all_reduce(partial_i)` 将所有部分和归并为完整结果。

## 运行

```bash
source .venv/bin/activate

# 4 卡
torchrun --nproc_per_node=4 agent_space_xpu/all_reduce_demo.py --m 4096 --n 4096 --k 4096

# 8 卡（更大矩阵）
torchrun --nproc_per_node=8 agent_space_xpu/all_reduce_demo.py --m 8192 --n 8192 --k 8192
```

可选参数：
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--m` | 4096 | 矩阵 M 维度（批次） |
| `--n` | 4096 | 矩阵 N 维度（隐藏层） |
| `--k` | 4096 | 矩阵 K 维度（输出） |
| `--warmup` | 5 | 预热迭代次数 |
| `--iter` | 20 | 基准测试迭代次数 |

## 验证结果

程序自动验证：通过 `dist.all_gather` 收集完整权重矩阵，计算完整 matmul 结果，与 all-reduce 结果对比。

| GPU 数 | 矩阵 | 每卡行数 | All-Reduce 数据量 | 延迟 | 算力 | 通信带宽 | 相对误差 |
|--------|------|---------|-------------------|------|------|---------|---------|
| 4 | 4096² | 1024 | 32 MB | 2.87 ms | **47.82 TFLOPS** | ~23.3 GB/s | 0.49% |
| 8 | 8192² | 1024 | 128 MB | 172 ms | 6.39 TFLOPS | ~1.6 GB/s | 0.78% |
| 16 | 4096² | 256 | 32 MB | 45 ms | 3.03 TFLOPS | ~1.5 GB/s | 1.1% |

（注：8 卡 8192² 时 all-reduce 消息量 128MB 成为瓶颈，算力从 48 TFLOPS 降到 6.4 TFLOPS）

## 分布式后端

- **后端**: `xccl`（PyTorch XPU 分布式后端，基于 oneCCL）
- **通信库**: oneCCL (oneccl-devel 2022.0.0)
- **GPU**: Intel Arc Pro B60 × 32（Level Zero 1.15.38308）

## 故障排查

如果 `torchrun` 启动后卡死无输出，通常是 oneCCL TCP 通信端口残留旧进程状态：

```bash
# 清理残留进程
pkill -9 -f torchrun 2>/dev/null; sleep 2

# 或用随机端口避免冲突
export MASTER_PORT=$(( (RANDOM % 60000) + 1024 ))
torchrun --nproc_per_node=4 agent_space_xpu/all_reduce_demo.py

# 若仍失败，需重置 GPU 驱动（需要 root）
# sudo rmmod xe && sudo modprobe xe
```

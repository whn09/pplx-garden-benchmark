# pplx-garden 在 B200 + EFA 上的编译与性能测试

## 概述

本仓库记录了 [pplx-garden](https://github.com/perplexityai/pplx-garden)（Perplexity AI 的开源 MoE 通信库）在 AWS p6-b200 实例（B200 GPU + 400Gb/s EFA）上的编译过程和性能测试结果。

## 测试环境

| 项目 | 配置 |
|------|------|
| 实例类型 | p6-b200.48xlarge × 2 |
| GPU | NVIDIA B200 183GB × 8/节点 (compute capability 10.0, sm_100) |
| 网络 | EFA 400Gb/s × 8/节点 |
| NVLink | NVLink 18 (NV18) |
| CUDA | 13.0 (pip 安装) |
| PyTorch | 2.9.1+cu130 |
| 驱动 | 580.126.09 |
| 内核 | Linux 6.17.0-1007-aws |
| Python | 3.12.3 (/opt/pytorch venv) |
| pplx-garden | v0.1.dev5 (commit f84bc412e) |

## 编译步骤

### 1. 安装系统依赖

```bash
# Rust 编译器（pplx-garden 核心用 Rust 编写）
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source $HOME/.cargo/env

# clang-18（CUDA kernel 编译需要）
sudo apt-get update
sudo apt-get install -y clang-18
```

### 2. 配置 CUDA 环境

p6-b200 的 CUDA 是通过 pip 安装的，不在标准路径。需要创建符号链接：

```bash
# pplx-garden 的 build.rs 硬编码了 /usr/local/cuda 路径
CUDA_PIP=/opt/pytorch/lib/python3.12/site-packages/nvidia/cu13
sudo ln -sf $CUDA_PIP /usr/local/cuda
ln -sf $CUDA_PIP/lib $CUDA_PIP/lib64
```

pip 安装的 CUDA 缺少 `cuda_profiler_api.h`，需要手动创建桩文件：

```bash
sudo python3 -c "
header = '''#ifndef __CUDA_PROFILER_API_H__
#define __CUDA_PROFILER_API_H__
#include <driver_types.h>
cudaError_t cudaProfilerStart(void);
cudaError_t cudaProfilerStop(void);
#endif
'''
with open('/usr/local/cuda/include/cuda_profiler_api.h', 'w') as f:
    f.write(header)
"
```

### 3. 配置 EFA 环境

确认 EFA 相关库已安装：

```bash
ls /opt/amazon/efa/lib/libfabric.so   # libfabric
ls /usr/lib/x86_64-linux-gnu/libefa.so  # EFA provider
```

### 4. 克隆并编译

```bash
git clone https://github.com/perplexityai/pplx-garden.git
cd pplx-garden

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib:/opt/amazon/efa/lib:$LD_LIBRARY_PATH

/opt/pytorch/bin/python -m pip install build
/opt/pytorch/bin/python -m build --wheel
/opt/pytorch/bin/pip install dist/*.whl
```

编译大约需要 10-15 分钟，主要耗时在 Rust 编译和 CUDA kernel 编译。

### 5. 验证安装

```bash
/opt/pytorch/bin/python -c "from pplx_garden.native.p2p_all_to_all import AllToAllContext; print('OK')"
```

### 6. 运行前配置

pplx-garden 使用 `pidfd_getfd` 系统调用在进程间共享 GPU 内存句柄（NVLink 模式）。需要满足以下条件之一：

```bash
# 方式 1：设置 ptrace_scope=0（推荐用于测试）
echo 0 | sudo tee /proc/sys/kernel/yama/ptrace_scope

# 方式 2：在 Docker 中运行并添加 capabilities
# docker run --cap-add=SYS_PTRACE --cap-add=SYS_ADMIN ...
```

> **注意**：如果不设置，NVLink 模式会报错：
> `RuntimeError: Failed to deserialize CUMemExportHandle: OtherString("failed to open import handle: -1")`

## Benchmark 脚本

参见 [scripts/](scripts/) 目录：

**pplx-garden:**
- `run_ll.sh` — LL 模式，NVLink + RDMA（128 tokens，decode 场景）
- `run_normal.sh` — Normal 模式，NVLink + RDMA（4096 tokens，prefill 场景）
- `run_ll_rdma_only.sh` — LL 模式，纯 RDMA（不用 NVLink，用于拆分对比）
- `run_normal_rdma_only.sh` — Normal 模式，纯 RDMA（不用 NVLink，用于拆分对比）

**UCCL-EP（使用 pplx-style benchmark，apple-to-apple 对比）：**
- `run_uccl_pplx_ll.sh` — UCCL-EP LL 模式，使用 `test_low_latency_pplx.py`

用法：

```bash
# 在 bastion 机器上执行，会 SSH 到两个节点并行启动 benchmark
bash scripts/run_ll.sh
bash scripts/run_normal.sh
bash scripts/run_uccl_pplx_ll.sh
```

## 性能结果

### LL 模式 (16 EP, 128 tokens, 7168 hidden, top-8, 288 experts, FP8 dispatch + BF16 combine)

| 传输模式 | Dispatch (p50) | Dispatch BW | Combine (p50) | Combine BW | D+C 总延迟 |
|---------|---------------|-------------|--------------|------------|-----------|
| **NVLink + RDMA** | 145 μs | 52.4 GB/s | 221 μs | 66.7 GB/s | **~366 μs** |
| **纯 RDMA** | 220 μs | 34.4 GB/s | 364 μs | 40.4 GB/s | **~584 μs** |

NVLink 加速效果：D+C 从 584 μs 降至 366 μs，**节省 37%**。

### Normal 模式 (16 EP, 4096 tokens, 7168 hidden, top-8, 288 experts, FP8 dispatch + BF16 combine)

| 传输模式 | Dispatch (p50) | Dispatch BW | Combine (p50) | Combine BW | D+C 总延迟 |
|---------|---------------|-------------|--------------|------------|-----------|
| **NVLink + RDMA** | 2903 μs | 83.4 GB/s | 5187 μs | 90.6 GB/s | **~8090 μs** |
| **纯 RDMA** | 5148 μs | 47.1 GB/s | 9566 μs | 49.1 GB/s | **~14714 μs** |

NVLink 加速效果：D+C 从 14714 μs 降至 8090 μs，**节省 45%**。

### 与其他方案对比 (16 EP, LL 模式)

| 方案 | Dispatch | Combine | D+C 总延迟 | 硬件 |
|------|----------|---------|-----------|------|
| **pplx-garden** NVL+RDMA (实测) | 145 μs | 221 μs | **~366 μs** | B200 + 400G EFA |
| **pplx-garden** 纯 RDMA (实测) | 220 μs | 364 μs | **~584 μs** | B200 + 400G EFA |
| **pplx-garden** EFA (官方 README) | 215 μs | 242 μs | ~457 μs | H100 + EFA |
| **UCCL-EP** test_low_latency (实测) | — | — | ~504 μs | B200 + 400G EFA |
| **UCCL-EP** pplx-style (实测) | 183 μs | 335 μs | **~519 μs** | B200 + 400G EFA |
| **DeepEP** IBGDA (官方 README) | 118 μs | 195 μs | ~313 μs | H800 + CX7 IB |

### 与其他方案对比 (16 EP, Normal 模式)

| 方案 | Dispatch BW | Combine BW | D+C 总延迟 | 硬件 |
|------|------------|------------|-----------|------|
| **pplx-garden** NVL+RDMA (实测) | 83.4 GB/s | 90.6 GB/s | ~8090 μs | B200 + 400G EFA |
| **pplx-garden** EFA (官方 README) | — | — | ~8576 μs | H100 + EFA |
| **UCCL-EP** (实测) | 49.7 GB/s | 57.7 GB/s | — | B200 + 400G EFA |

## 带宽数据解读

pplx-garden Normal 模式报告的 83-91 GB/s 带宽远高于 UCCL-EP 的 50-58 GB/s，但这 **不是 EFA 吞吐量更高**，而是架构差异导致的：

```
pplx-garden (NVLink + RDMA 混合):
  节点内 8/16 GPU → NVLink (~900 GB/s)  → 几乎瞬间完成
  节点间 8/16 GPU → EFA RDMA            → 只传 ~50% 数据
  报告带宽 = 全部数据量 / 总时间 = 看起来很高

UCCL-EP (全 RDMA):
  所有 16/16 GPU → RDMA (ibverbs)       → EFA 搬运 100% 数据
  报告带宽 = 全部数据量 / 总时间 = EFA 实际吞吐量
```

反推 **实际 EFA 吞吐量**（仅计算节点间数据）：

| 方案 | 报告带宽 | 实际 EFA 吞吐量（估算） |
|------|---------|----------------------|
| pplx-garden (NVL+RDMA) | 83.4 GB/s | ~41.7 GB/s（50% 数据走 EFA）|
| pplx-garden (纯 RDMA，实测) | 47.1 GB/s | **47.1 GB/s**（100% 数据走 EFA）|
| UCCL-EP (全 RDMA) | 49.7 GB/s | **49.7 GB/s**（100% 数据走 EFA）|

纯 RDMA 实测验证了分析：pplx-garden 去掉 NVLink 后 EFA 吞吐量为 47-49 GB/s，与 UCCL-EP 的 50-58 GB/s 基本一致。pplx-garden 混合模式的高带宽数字主要来自 NVLink 卸载。

LL 模式差距较小（366 vs 504 μs），因为 **UCCL-EP 在 LL 模式也使用 NVLink**（`allow_nvlink_for_low_latency_mode=True`），差距主要来自 kernel/proxy 实现效率。

使用 UCCL-EP 的 pplx-style benchmark（`test_low_latency_pplx.py`）进行 apple-to-apple 对比，UCCL-EP D+C ~519 μs vs pplx-garden ~366 μs，差距 ~42%。UCCL-EP pplx-style benchmark 的 Dispatch BW 41.4 GB/s、Combine BW 43.8 GB/s，与 pplx-garden 纯 RDMA 模式的 34-40 GB/s 在同一量级。

## 已知问题

1. **ptrace_scope**：必须设为 0 或在 Docker 中使用 `--cap-add=SYS_PTRACE`，否则 NVLink 模式的 `pidfd_getfd` 会失败
2. **pip CUDA 缺少头文件**：`cuda_profiler_api.h` 需要手动创建
3. **Normal RDMA-only OOM**：不启用 NVLink 时，4096 tokens 的 buffer 超出 B200 183GB 显存
4. **B200 sm_100 支持**：pplx-garden 的 `build.rs` 已包含 `compute_100a` target，原生支持 B200

## 参考

- [pplx-garden](https://github.com/perplexityai/pplx-garden)
- [pplx-garden 论文: RDMA Point-to-Point Communication for LLM Systems](https://arxiv.org/abs/2510.27656)
- [UCCL-EP](https://github.com/uccl-project/uccl/tree/main/ep)
- [DeepEP](https://github.com/deepseek-ai/DeepEP)

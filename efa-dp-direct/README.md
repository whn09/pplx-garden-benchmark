# efa-dp-direct 在 B200 + EFA 上的验证

## 概述

[efa-dp-direct](https://github.com/amzn/efa-dp-direct) 是 Amazon 开源的 CUDA 库，允许 GPU kernel 直接操作 EFA 的发送队列（SQ）和完成队列（CQ），绕过 CPU proxy，实现 GPU-direct RDMA。这是在 EFA 上消除 CPU proxy 瓶颈、达到 IBGDA 级 LL 性能的关键技术路径。

本目录记录了在 p6-b200（B200 + 400Gb/s EFA）上的验证过程和结果。

## 测试环境

| 项目 | 配置 |
|------|------|
| 实例类型 | p6-b200.48xlarge × 2 |
| GPU | NVIDIA B200 183GB (SM 10.0) |
| 网络 | EFA 400Gb/s × 8/节点 |
| EFA 驱动 | 2.17.3g |
| rdma-core | 59.amzn0-1 |
| CUDA | 13.0 |
| efa-dp-direct | v0.0.1 |

## 验证结果

### 1. 基础 API 测试 (`test_efa_dp_direct.cu`)

| 测试项 | 结果 |
|--------|------|
| 库加载 + 版本检查 | OK (0.0.1) |
| GPU 上创建 CQ/QP | OK |
| Device-side 兼容性检查 | OK |
| WR 初始化 + SGE 设置 | OK |

### 2. 真实 EFA 硬件访问 (`test_efa_dp_real.cu`)

使用 `efadv_query_qp_wqs()` 获取 EFA QP 的底层硬件指针，注册到 GPU：

| 资源 | 注册方式 | 结果 |
|------|---------|------|
| SQ buffer (BAR-mapped LLQ) | `cudaHostRegister(IoMemory)` | OK (需要 sudo) |
| SQ doorbell (BAR-mapped) | `cudaHostRegister(IoMemory)` | OK (需要 sudo) |
| RQ buffer (DMA memory) | `cudaHostRegister(Default)` | OK |
| RQ doorbell (BAR-mapped) | `cudaHostRegister(IoMemory)` | OK (需要 sudo) |
| CQ buffer | 两种方式均失败 | 需要 dmabuf 方式 |

GPU kernel 成功写入真实 EFA SQ/RQ buffer 并验证 QP 状态更新。

### 3. 两节点 GPU-Direct RDMA Write (`test_efa_dp_e2e.cu`)

**GPU kernel 直接发起跨节点 RDMA Write，零 CPU 参与数据路径：**

```
Node 0 (Sender):
  GPU kernel → 写 WR 到 EFA SQ buffer → 写 doorbell (全部从 CUDA kernel)
  Completion: status=0 (success)

Node 1 (Receiver):
  验证: 512/512 words match (4KB 数据完整传输)
  *** GPU-DIRECT RDMA WRITE SUCCESS ***
```

这是首次在 B200 + EFA 上实现 GPU-direct RDMA write。

### 4. CPU 基线测试 (`test_efa_cpu_rdma.c`)

验证 EFA SRD QP 的 RDMA write 功能正常（使用 ibverbs extended QP API）。

## 关键发现

### 1. `efadv_query_qp_wqs()` 是关键 API

rdma-core 59 中的 `efadv_query_qp_wqs()` 返回 EFA QP 的底层硬件指针：
- `sq_buffer`：SQ 环形缓冲区（BAR-mapped）
- `sq_doorbell`：SQ 门铃寄存器（BAR-mapped）
- `rq_buffer`：RQ 环形缓冲区（DMA memory）
- `rq_doorbell`：RQ 门铃寄存器（BAR-mapped）

同样，`efadv_query_cq()` 返回 CQ 的硬件缓冲区指针。

### 2. GPU 访问 BAR 区域需要 `cudaHostRegister(IoMemory)` + sudo

SQ buffer 和 doorbell 是 EFA 设备的 BAR-mapped 区域。通过 `cudaHostRegister` 的 `cudaHostRegisterIoMemory` 标志可以注册到 GPU 地址空间，但需要 root 权限。

### 3. CQ buffer 注册是遗留问题

CQ buffer 无法通过 `cudaHostRegister` 注册（Default 和 IoMemory 均失败）。需要使用 `efadv_create_cq` 的 `ext_mem_dmabuf` 路径，将 CQ 创建在 GPU 可访问的内存中。

### 4. 不需要 nvidia-peermem

GPU-direct 访问 EFA SQ/doorbell 不需要 `nvidia-peermem` 内核模块。`cudaHostRegister(IoMemory)` 足以让 GPU 写入 EFA BAR 空间。

### 5. EFA SRD QP 必须使用 extended QP API

RDMA write 操作必须通过 `efadv_create_qp_ex()` + `IBV_QP_EX_WITH_RDMA_WRITE` 创建 QP。标准 `efadv_create_driver_qp()` 不支持。

### 6. GPU 侧和 CPU 侧 posting 不能混用

efa-dp-direct 的 GPU QP 和 ibverbs 的 CPU QP 共享物理 SQ buffer，但各自维护独立的 producer counter。混用会导致 SQ 状态不一致。必须在 QP 创建后选择一种方式使用。

## 编译

```bash
# 前提：CUDA toolkit, rdma-core, libibverbs-dev, libefa

# 1. 克隆 efa-dp-direct
git clone https://github.com/amzn/efa-dp-direct.git
cd efa-dp-direct/CUDA && make

# 2. 编译测试
cd ..
nvcc -o test_efa_dp_direct test_efa_dp_direct.cu \
  -LCUDA/build -lefacudadp -ICUDA/src \
  -Xlinker -rpath,$PWD/CUDA/build --gpu-architecture=sm_90

nvcc -o test_efa_dp_real test_efa_dp_real.cu \
  -LCUDA/build -lefacudadp -ICUDA/src \
  -L/usr/lib/x86_64-linux-gnu -libverbs -lefa \
  -Xlinker -rpath,$PWD/CUDA/build --gpu-architecture=sm_90

nvcc -o test_efa_dp_e2e test_efa_dp_e2e.cu \
  -LCUDA/build -lefacudadp -ICUDA/src \
  -L/usr/lib/x86_64-linux-gnu -libverbs -lefa \
  -Xlinker -rpath,$PWD/CUDA/build --gpu-architecture=sm_90

gcc -o test_efa_cpu_rdma test_efa_cpu_rdma.c -libverbs -lefa
```

## 运行

```bash
# 基础测试
./test_efa_dp_direct

# 真实硬件测试（需要 sudo）
sudo ./test_efa_dp_real

# 两节点 GPU-direct RDMA（需要 sudo）
# Node 0: sudo ./test_efa_dp_e2e 0 <node1_private_ip>
# Node 1: sudo ./test_efa_dp_e2e 1 <node0_private_ip>

# CPU 基线
# Node 0: ./test_efa_cpu_rdma 0 <node1_private_ip>
# Node 1: ./test_efa_cpu_rdma 1 <node0_private_ip>
```

## 对 EP 通信的意义

当前 EFA 上的 EP 通信（UCCL-EP、pplx-garden）都使用 CPU proxy 路径：
```
GPU kernel → CPU proxy → ibverbs → EFA NIC  (额外 25-30us 延迟)
```

efa-dp-direct 可以消除 CPU proxy：
```
GPU kernel → EFA SQ buffer + doorbell  (直接，零 CPU)
```

如果集成到 EP 通信库（如 UCCL-EP 或新框架），可能将 EFA LL 延迟从 ~150-230us 降低到接近 IBGDA 的 ~118us。

### 集成的剩余工作

1. **CQ GPU 访问**：需要通过 dmabuf 或其他方式让 GPU 能轮询 CQ
2. **多 QP 管理**：EP 通信需要大量 QP（每对 rank 一个），需要高效管理
3. **Memory registration**：GPU 内存需要注册到 EFA（当前 `ibv_reg_mr` 对 GPU memory 成功）
4. **性能调优**：PCIe P2P 写入延迟、doorbell batching、CQ polling 效率
5. **权限管理**：`cudaHostRegister(IoMemory)` 需要 root，生产环境需要解决

## 参考

- [efa-dp-direct](https://github.com/amzn/efa-dp-direct)
- [NVSHMEM issue #4](https://github.com/NVIDIA/nvshmem/issues/4) — efa-dp-direct 与 NVSHMEM 集成讨论
- [efadv_query_qp_wqs](https://github.com/linux-rdma/rdma-core) — rdma-core 中的 EFA direct verbs 扩展

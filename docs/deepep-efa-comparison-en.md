# Comparison of 4 Approaches to Run DeepEP on EFA

## Background

[DeepEP](https://github.com/deepseek-ai/DeepEP) is DeepSeek's open-source GPU-initiated expert-parallel (EP) communication library, a key component for efficient large-scale MoE inference and training. However, DeepEP is tightly coupled with NVIDIA IBGDA (InfiniBand GPU Direct Async) and can only run on InfiniBand networks, making it incompatible with AWS EFA (Elastic Fabric Adapter) and other RDMA networks.

This document compares 4 approaches to enable DeepEP-equivalent functionality on EFA:

1. **NCCL GIN** — DeepEP PR #521, replacing NVSHMEM with NCCL Device API
2. **UCCL-EP** — A fully independent cross-platform EP communication library
3. **Modified NVSHMEM** — Adding EFA transport support to NVSHMEM
4. **pplx-garden** — Perplexity AI's open-source RDMA communication solution

## Overview

| | NCCL GIN (PR #521) | UCCL-EP | Modified NVSHMEM | pplx-garden |
|--|---|---|---|---|
| **GPU-to-NIC path** | NCCL Device API → CPU proxy | GPU→FIFO→CPU proxy→ibverbs | NVSHMEM API → UCX proxy | GPU kernel → CPU (Rust) → libfabric |
| **GPU-direct on EFA** | No (GDAKI requires IBGDA) | No | Potentially (via efa-dp-direct) | No |
| **EFA validated** | No (known compat bugs) | Yes, multi-platform | No | Yes, with benchmarks |
| **AMD GPU support** | No (NCCL limitation) | Yes | No (NVSHMEM limitation) | No |
| **DeepEP API compatible** | Yes (same codebase) | Yes (compatible interface) | Yes (same codebase) | No (independent API) |
| **Intra-node transfer** | NVLink (via NCCL) | NVLink (always hybrid NVLink + RDMA) | NVLink (via NVSHMEM) | NVLink (always hybrid NVLink + RDMA) |
| **Language** | C++/CUDA | C++/CUDA | C++/CUDA | Rust + CUDA |
| **Open-source status** | PR not merged | Released | Requires custom work | Released |

## Performance Comparison

### Low-Latency Mode (16 EP, 128 tokens, 7168 hidden, top-8, FP8 dispatch + BF16 combine)

#### Official README Data

| Approach | Dispatch Latency | Dispatch BW | Combine Latency | Combine BW | D+C Total | Hardware |
|----------|-----------------|-------------|-----------------|------------|-----------|----------|
| **DeepEP** | 118 us | 63 GB/s | 195 us | 74 GB/s | ~313 us | H800 + CX7 IB 400Gb/s |
| **pplx-garden** (CX7) | 110 us | — | 186 us | — | ~296 us | H100 + CX7 |
| **pplx-garden** (EFA) | 215 us | — | 242 us | — | ~457 us | H100 + EFA |
| **UCCL-EP** (p6-b200) | 228 us | 33 GB/s | 318 us | 46 GB/s | ~546 us | B200 + 400G EFA |
| **UCCL-EP** (p5en) | 226 us | 36 GB/s | 293 us | 48 GB/s | ~519 us | H200 + 200G EFA x2 |

> Note: DeepEP uses DeepSeek-V3 config (256 experts); UCCL-EP uses 288 experts; pplx-garden official README does not specify expert count.

#### Measured Data (B200 + 400G EFA, 16 EP, 288 experts)

| Approach | Dispatch Latency | Dispatch BW | Combine Latency | Combine BW | D+C Total |
|----------|-----------------|-------------|-----------------|------------|-----------|
| **pplx-garden** (NVL+RDMA) | 145 us | 52.4 GB/s | 221 us | 66.7 GB/s | **~366 us** |
| **pplx-garden** (RDMA-only) | 220 us | 34.4 GB/s | 364 us | 40.4 GB/s | ~584 us |
| **UCCL-EP** (pplx-style) | 183 us | 41.4 GB/s | 335 us | 43.8 GB/s | ~519 us |
| **UCCL-EP** (test_low_latency) | — | — | — | — | ~504 us |

> Note: pplx-garden (NVL+RDMA) BW includes NVLink portion and cannot be directly compared with RDMA-only approaches. UCCL-EP pplx-style uses `test_low_latency_pplx.py` (288 experts) for apple-to-apple comparison with pplx-garden's benchmark. UCCL-EP test_low_latency only reports D+C total latency.

### Normal (High-Throughput) Mode (16 EP, 4096 tokens, 7168 hidden, top-4 groups, top-8, FP8 dispatch + BF16 combine)

#### Official README Data

| Approach | Dispatch Latency | Dispatch BW | Combine Latency | Combine BW | D+C Total | Hardware |
|----------|-----------------|-------------|-----------------|------------|-----------|----------|
| **DeepEP** | — | 43 GB/s (RDMA) | — | 43 GB/s (RDMA) | — | H800 + CX7 IB 400Gb/s |
| **pplx-garden** (CX7) | 2735 us | — | 1062 us | — | ~3797 us | H100 + CX7 |
| **pplx-garden** (EFA) | 3197 us | — | 5379 us | — | ~8576 us | H100 + EFA |
| **UCCL-EP** (p6-b200) | 1141 us | 53 GB/s (RDMA) | 1965 us | 60 GB/s (RDMA) | ~3106 us | B200 + 400G EFA |
| **UCCL-EP** (p5en) | 1196 us | 50 GB/s (RDMA) | 6379 us | 18 GB/s (RDMA) | ~7575 us | H200 + 200G EFA x2 |

> Note: DeepEP only reports RDMA bottleneck bandwidth, no latency data. pplx-garden official data is from H100; UCCL-EP p6-b200 data is from B200. UCCL-EP and DeepEP report RDMA bottleneck bandwidth.

#### Measured Data (B200 + 400G EFA, 16 EP, 288 experts)

| Approach | Dispatch Latency | Dispatch BW | Combine Latency | Combine BW | D+C Total |
|----------|-----------------|-------------|-----------------|------------|-----------|
| **pplx-garden** (NVL+RDMA) | 2903 us | 83.4 GB/s | 5187 us | 90.6 GB/s | ~8090 us |
| **pplx-garden** (RDMA-only) | 5148 us | 47.1 GB/s | 9566 us | 49.1 GB/s | ~14714 us |

> Note: pplx-garden (NVL+RDMA) BW includes NVLink portion. RDMA-only mode shows actual EFA throughput of 47-49 GB/s.

### Bandwidth Interpretation

All approaches use hybrid NVLink + RDMA architecture in Normal mode: NVLink for intra-node, RDMA for inter-node. pplx-garden reports 83-91 GB/s which includes the NVLink portion (formula: total_data / total_time), while DeepEP and UCCL-EP report RDMA bottleneck bandwidth.

pplx-garden RDMA-only measurement (47.1 GB/s) validates the actual EFA throughput, which is consistent with UCCL-EP's RDMA bandwidth (53 GB/s on B200).

In LL mode, pplx-garden (366 us) vs UCCL-EP (519 us) — both use hybrid NVLink + RDMA architecture, the gap is mainly from kernel/proxy implementation efficiency.

## Detailed Analysis

### Approach 1: NCCL GIN (DeepEP PR #521)

**Mechanism**: Introduces a `CommunicationBackend` abstraction layer in DeepEP, mapping NVSHMEM's PGAS memory model to NCCL's window-based model. GPU kernels issue network operations through the NCCL Device API (`put()`, `signal()`, `flush()`, etc.).

**NCCL GIN has two modes**:

| GIN Mode | Mechanism | EFA Support |
|----------|-----------|-------------|
| `NCCL_GIN_TYPE=3` (GDAKI) | GPU directly operates NIC doorbell | Not supported (requires IBGDA) |
| `NCCL_GIN_TYPE=2` (Proxy) | GPU → CPU proxy → NIC | Theoretically supported |

**Issues on EFA**:
- GIN Proxy mode has known compatibility issues on EFA ([NCCL #1913](https://github.com/NVIDIA/nccl/issues/1913), [#1921](https://github.com/NVIDIA/nccl/issues/1921)): EFA multi-rail topology inconsistencies cause initialization failures
- Requires `OFI_NCCL_FORCE_NUM_RAILS=4` workaround ([aws-ofi-nccl #1061](https://github.com/aws/aws-ofi-nccl/issues/1061))
- PR was only tested on H100 + InfiniBand; completely unvalidated on EFA
- Requires NCCL 2.28.9+

**Verdict**: Theoretically possible but practically problematic. Only Proxy mode works on EFA, which won't outperform UCCL-EP. Compatibility issues remain unresolved.

### Approach 2: UCCL-EP

**Mechanism**: A fully independent EP communication library. GPU kernels write RDMA commands to a FIFO; CPU proxy threads read and issue RDMA operations via ibverbs. The proxy path is deeply optimized for EP communication patterns.

**Strengths**:
- **Broadest hardware support**: EFA, CX7 InfiniBand, Broadcom Thor-2, AMD Pollara
- **Only solution supporting AMD GPUs** (CUDA + HIP)
- DeepEP-compatible API, drop-in replacement
- Validated on p5en (H200), p6-b200 (B200), MI300X, and more

**Limitations**:
- LL mode latency higher than pplx-garden (measured 519 us vs 366 us on B200), mainly due to kernel/proxy implementation efficiency differences
- LL mode latency ~1.6x higher than IBGDA due to CPU proxy overhead

**Verdict**: The most mature and reliable EFA solution today. Best cross-platform compatibility.

### Approach 3: Modified NVSHMEM for EFA

**Mechanism**: NVSHMEM includes a UCX transport (UCX → libfabric → EFA) and an ibrc transport alongside IBGDA. In theory, DeepEP could work on non-IBGDA networks through these transports.

**Possible paths**:

| Path | Description | Feasibility |
|------|-------------|-------------|
| **A. UCX/ibrc transport** | `nvshmem_put_nbi` calls from GPU kernels are routed through host proxy | Requires significant DeepEP kernel changes; performance degrades to CPU proxy level |
| **B. efa-dp-direct integration** | Amazon's [efa-dp-direct](https://github.com/amzn/efa-dp-direct) provides GPU-direct access to EFA queue pairs. If integrated into NVSHMEM as a new transport, it could enable true GPU-direct RDMA on EFA | efa-dp-direct is very early-stage (open-sourced Oct 2025, 3 stars) |

**Why efa-dp-direct matters**:
- Provides device-side APIs for CUDA kernels to directly post EFA work requests and poll completion queues
- If mature and integrated into NVSHMEM, it is the **only path that could match DeepEP's IBGDA-level LL performance on EFA**
- Discussed in NVSHMEM issue [#4](https://github.com/NVIDIA/nvshmem/issues/4)

**Verdict**: Not practical short-term (requires efa-dp-direct maturation + NVSHMEM integration + DeepEP adaptation), but **highest long-term potential**.

### Approach 4: pplx-garden (Perplexity AI)

**Mechanism**: Perplexity AI's open-source inference communication library with a Rust-based RDMA TransferEngine (supporting libfabric/EFA and libibverbs/CX7), paired with custom CUDA dispatch/combine kernels. **Always uses hybrid NVLink + RDMA architecture**.

**Repository**: [github.com/perplexityai/pplx-garden](https://github.com/perplexityai/pplx-garden)

**Strengths**:
- Native EFA and CX7 support with complete benchmark data
- Best end-to-end LL latency on EFA (measured 16EP D+C ~366 us on B200, official ~457 us on H100)
- Recommended by DeepEP maintainers as an EFA alternative ([issue #369](https://github.com/deepseek-ai/DeepEP/issues/369))
- Supports split send/recv and micro-batching; SM-free during RDMA transfers
- NVLink + RDMA hybrid architecture effectively reduces EFA load

**Limitations**:
- API incompatible with DeepEP; completely independent project
- Rust ecosystem requires additional integration work with existing Python/C++ inference frameworks
- No AMD GPU support
- Smaller community (370 stars)

**Verdict**: Lowest end-to-end LL latency on EFA among existing open-source solutions, but poor ecosystem compatibility.

## Root Cause: Why is LL Latency Higher on EFA?

All EFA-based approaches show higher LL mode latency than DeepEP with IBGDA. The fundamental reason is:

```
DeepEP (IBGDA):  GPU kernel → NIC doorbell  (118 us dispatch)
                 Zero CPU involvement

EFA approaches:  GPU kernel → CPU proxy → EFA NIC  (145-228 us dispatch)
                 Extra GPU↔CPU communication hop
```

EFA does not support IBGDA/GDAKI, so GPUs cannot directly operate the NIC. All EFA solutions must route through a CPU proxy, adding an inherent GPU→CPU→NIC latency overhead.

**The only potential breakthrough** is Amazon's `efa-dp-direct` — if it matures and is integrated into a communication library (NVSHMEM or otherwise), EFA could support true GPU-direct RDMA, eliminating the CPU proxy bottleneck.

## Recommendations

| Use Case | Recommended Approach | Rationale |
|----------|---------------------|-----------|
| Need EP on EFA now | **UCCL-EP** | Most mature, DeepEP-compatible API |
| Lowest end-to-end LL latency on EFA | **pplx-garden** | Measured D+C ~366 us vs UCCL-EP ~519 us (B200) |
| Need AMD GPU support | **UCCL-EP** | Only solution supporting AMD |
| Keep DeepEP code unchanged | **UCCL-EP** > NCCL GIN | UCCL-EP is API-compatible; NCCL GIN has compat issues |
| Long-term GPU-direct on EFA | Watch **efa-dp-direct** + NVSHMEM | Only path to IBGDA-level LL performance |

## References

- [DeepEP](https://github.com/deepseek-ai/DeepEP)
- [UCCL-EP](https://github.com/uccl-project/uccl/tree/main/ep)
- [DeepEP NCCL PR #521](https://github.com/deepseek-ai/DeepEP/pull/521)
- [pplx-garden](https://github.com/perplexityai/pplx-garden)
- [efa-dp-direct](https://github.com/amzn/efa-dp-direct)
- [NVSHMEM](https://github.com/NVIDIA/nvshmem)
- [NCCL GIN + EFA Compatibility Issues](https://github.com/NVIDIA/nccl/issues/1913)
- [DeepEP EFA Discussion](https://github.com/deepseek-ai/DeepEP/issues/369)
- [GPU-Initiated Networking for NCCL (Paper)](https://arxiv.org/abs/2511.15076)

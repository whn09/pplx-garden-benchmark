#!/bin/bash
# 在每个 B200 节点上执行的环境配置脚本
# 用法: ssh ubuntu@<node-ip> 'bash -s' < scripts/setup_node.sh
set -e

echo "=== 1. 安装 Rust ==="
if ! command -v rustc &>/dev/null; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
  source $HOME/.cargo/env
  echo "Rust $(rustc --version) installed"
else
  echo "Rust already installed: $(rustc --version)"
fi

echo "=== 2. 安装 clang-18 ==="
if ! command -v clang-18 &>/dev/null; then
  sudo apt-get update
  sudo apt-get install -y clang-18
  echo "clang-18 installed"
else
  echo "clang-18 already installed"
fi

echo "=== 3. 配置 CUDA 路径 ==="
CUDA_PIP=/opt/pytorch/lib/python3.12/site-packages/nvidia/cu13
if [ ! -L /usr/local/cuda ]; then
  sudo ln -sf $CUDA_PIP /usr/local/cuda
  echo "Created /usr/local/cuda -> $CUDA_PIP"
else
  echo "/usr/local/cuda already exists"
fi

if [ ! -L $CUDA_PIP/lib64 ]; then
  ln -sf $CUDA_PIP/lib $CUDA_PIP/lib64
  echo "Created lib64 symlink"
fi

echo "=== 4. 创建 cuda_profiler_api.h 桩文件 ==="
HEADER=/usr/local/cuda/include/cuda_profiler_api.h
if [ ! -f "$HEADER" ]; then
  sudo python3 -c "
header = '''#ifndef __CUDA_PROFILER_API_H__
#define __CUDA_PROFILER_API_H__
#include <driver_types.h>
cudaError_t cudaProfilerStart(void);
cudaError_t cudaProfilerStop(void);
#endif
'''
with open('$HEADER', 'w') as f:
    f.write(header)
"
  echo "Created $HEADER"
else
  echo "$HEADER already exists"
fi

echo "=== 5. 设置 ptrace_scope ==="
echo 0 | sudo tee /proc/sys/kernel/yama/ptrace_scope
echo "ptrace_scope set to 0"

echo "=== 6. 克隆并编译 pplx-garden ==="
if [ ! -d /home/ubuntu/pplx-garden ]; then
  git clone https://github.com/perplexityai/pplx-garden.git /home/ubuntu/pplx-garden
fi

cd /home/ubuntu/pplx-garden
source $HOME/.cargo/env
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib:/opt/amazon/efa/lib:$LD_LIBRARY_PATH

/opt/pytorch/bin/python -m pip install build 2>/dev/null
echo "Building wheel (this takes ~15 minutes)..."
/opt/pytorch/bin/python -m build --wheel
/opt/pytorch/bin/pip install dist/*.whl --force-reinstall

echo "=== 7. 验证安装 ==="
/opt/pytorch/bin/python -c "from pplx_garden.native.p2p_all_to_all import AllToAllContext; print('pplx-garden import OK')"

echo "=== Setup complete ==="

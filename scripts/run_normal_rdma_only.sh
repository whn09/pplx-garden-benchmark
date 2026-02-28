#!/bin/bash
# pplx-garden Normal benchmark: 纯 RDMA 模式（不用 NVLink，用于拆分对比）
# 16 EP (2 nodes × 8 GPUs), 4096 tokens, DeepSeek-V3 参数
# 注意：去掉 --nvlink=8 参数，所有数据走 RDMA
set -e

NODES=("3.140.151.17" "3.150.29.153")
KEY="/home/ubuntu/henanwan-us-east-2.pem"
MASTER_ADDR="172.31.23.209"
MASTER_PORT=29513
LOG="/tmp/pplx_normal_rdma_only.log"

> "$LOG"

for i in "${!NODES[@]}"; do
  ip=${NODES[$i]}
  echo "[$(date +%H:%M:%S)] Launching on $ip (rank $i)..." >> "$LOG"
  ssh -i "$KEY" -o StrictHostKeyChecking=no ubuntu@$ip "
    cd /home/ubuntu/pplx-garden
    export PATH=/usr/local/cuda/bin:\$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/lib:/opt/amazon/efa/lib:\$LD_LIBRARY_PATH
    /opt/pytorch/bin/python -m benchmarks.bench_all_to_all \
      --world-size 16 \
      --nets-per-gpu 1 \
      --init-method=tcp://$MASTER_ADDR:$MASTER_PORT \
      --node-rank=$i \
      --max-num-tokens=4096 \
      --hidden-dim=7168 \
      --num-experts=288 \
      --num-experts-per-token=8 \
      --num-warmup=100 \
      --num-repeats=1000
  " >> "$LOG" 2>&1 &
done

wait
echo "[$(date +%H:%M:%S)] All nodes finished." >> "$LOG"
echo "Results saved to $LOG"
grep "both time" "$LOG"

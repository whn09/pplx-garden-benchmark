#!/bin/bash
# pplx-garden LL benchmark: NVLink + RDMA 混合模式
# 16 EP (2 nodes × 8 GPUs), 128 tokens, DeepSeek-V3 参数
set -e

NODES=("3.140.151.17" "3.150.29.153")
KEY="/home/ubuntu/henanwan-us-east-2.pem"
MASTER_ADDR="172.31.23.209"
MASTER_PORT=29510
LOG="/tmp/pplx_ll_benchmark.log"

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
      --nvlink=8 \
      --max-num-tokens=128 \
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

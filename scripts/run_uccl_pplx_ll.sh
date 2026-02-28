#!/bin/bash
set -e
NODES=("3.140.151.17" "3.150.29.153")
KEY="/home/ubuntu/henanwan-us-east-2.pem"
MASTER_ADDR="172.31.23.209"
LOG="/tmp/uccl_pplx_ll.log"

> "$LOG"

for i in "${!NODES[@]}"; do
  ip=${NODES[$i]}
  echo "[$(date +%H:%M:%S)] Launching on $ip (rank $i)..." >> "$LOG"
  ssh -i "$KEY" -o StrictHostKeyChecking=no ubuntu@$ip "
    cd /home/ubuntu/uccl/ep/bench
    export PATH=/usr/local/cuda/bin:\$PATH
    export LD_LIBRARY_PATH=/opt/pytorch/lib/python3.12/site-packages/torch/lib:/usr/local/cuda/lib:/opt/amazon/efa/lib:\$LD_LIBRARY_PATH
    UCCL_SOCKET_IFNAME=enp71s0 \
    /opt/pytorch/bin/torchrun \
      --nnodes=2 --nproc_per_node=8 \
      --node_rank=$i \
      --master_addr=$MASTER_ADDR --master_port=12355 \
      test_low_latency_pplx.py \
      --num-tokens=128 --hidden=7168 --num-topk=8 --num-experts=288 \
      --dispatch-use-fp8 \
      --num-warmup=100 --num-repeats=1000
  " >> "$LOG" 2>&1 &
done

wait
echo "[$(date +%H:%M:%S)] All nodes finished." >> "$LOG"

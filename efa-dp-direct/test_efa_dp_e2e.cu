// test_efa_dp_e2e.cu — v4: Pure GPU-direct RDMA write
// Fix: create separate QP for GPU (no CPU posting to mix up SQ state)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <cuda_runtime.h>
#include <infiniband/verbs.h>
#include <infiniband/efadv.h>

#include "efa_cuda_dp.h"
#include "efa_cuda_dp.cuh"
#include "efa_cuda_dp_impl.cuh"

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while (0)

#define OOB_PORT 19876
#define DATA_SIZE 4096
#define MAGIC 0xDEADBEEFCAFEBABEULL

struct qp_info {
    uint32_t qp_num;
    union ibv_gid gid;
    uint32_t rkey;
    uint64_t remote_addr;
};

__global__ void gpu_rdma_write_kernel(
    efa_cuda_qp *qp,
    uint32_t rkey, uint64_t remote_addr,
    uint32_t lkey, uint64_t local_addr, uint32_t length,
    uint16_t ah, uint32_t remote_qpn, uint32_t qkey)
{
    if (threadIdx.x != 0) return;

    printf("[GPU] RDMA Write: raddr=0x%lx rkey=0x%x laddr=0x%lx lkey=0x%x len=%u ah=%u qpn=%u qkey=0x%x\n",
           remote_addr, rkey, local_addr, lkey, length, ah, remote_qpn, qkey);

    efa_io_tx_wqe wr_buf;
    memset(&wr_buf, 0, sizeof(wr_buf));
    efa_cuda_init_rdma_write_wr(&wr_buf, 42, rkey, remote_addr);
    efa_cuda_wr_set_sge(&wr_buf, lkey, local_addr, length);
    efa_cuda_wr_set_remote(&wr_buf, ah, remote_qpn, qkey);

    int ret = efa_cuda_start_sq_batch(qp, 1);
    if (ret) { printf("[GPU] start_sq_batch failed: %d\n", ret); return; }

    efa_cuda_sq_batch_place_wr(qp, 0, &wr_buf);
    printf("[GPU] Ringing doorbell from GPU...\n");
    efa_cuda_flush_sq_wrs(qp);
    printf("[GPU] Posted!\n");
}

int exchange(int rank, const char *peer, struct qp_info *l, struct qp_info *r) {
    if (rank == 0) {
        int s = socket(AF_INET, SOCK_STREAM, 0);
        int o = 1; setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &o, sizeof(o));
        struct sockaddr_in a = {}; a.sin_family = AF_INET; a.sin_port = htons(OOB_PORT); a.sin_addr.s_addr = INADDR_ANY;
        bind(s, (struct sockaddr*)&a, sizeof(a)); listen(s, 1);
        int c = accept(s, NULL, NULL);
        send(c, l, sizeof(*l), 0); recv(c, r, sizeof(*r), MSG_WAITALL);
        close(c); close(s);
    } else {
        int c = socket(AF_INET, SOCK_STREAM, 0);
        struct sockaddr_in a = {}; a.sin_family = AF_INET; a.sin_port = htons(OOB_PORT);
        inet_pton(AF_INET, peer, &a.sin_addr);
        for (int i = 0; i < 50; i++) { if (connect(c, (struct sockaddr*)&a, sizeof(a)) == 0) break; usleep(100000); }
        recv(c, r, sizeof(*r), MSG_WAITALL); send(c, l, sizeof(*l), 0);
        close(c);
    }
    return 0;
}

void barrier(int rank, const char *peer) {
    char b = 'X';
    if (rank == 0) {
        int s = socket(AF_INET, SOCK_STREAM, 0);
        int o = 1; setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &o, sizeof(o));
        struct sockaddr_in a = {}; a.sin_family = AF_INET; a.sin_port = htons(OOB_PORT+1); a.sin_addr.s_addr = INADDR_ANY;
        bind(s, (struct sockaddr*)&a, sizeof(a)); listen(s, 1);
        int c = accept(s, NULL, NULL);
        recv(c, &b, 1, MSG_WAITALL); send(c, &b, 1, 0);
        close(c); close(s);
    } else {
        int c = socket(AF_INET, SOCK_STREAM, 0);
        struct sockaddr_in a = {}; a.sin_family = AF_INET; a.sin_port = htons(OOB_PORT+1);
        inet_pton(AF_INET, peer, &a.sin_addr);
        for (int i = 0; i < 50; i++) { if (connect(c, (struct sockaddr*)&a, sizeof(a)) == 0) break; usleep(100000); }
        send(c, &b, 1, 0); recv(c, &b, 1, MSG_WAITALL);
        close(c);
    }
}

int main(int argc, char **argv) {
    if (argc != 3) { fprintf(stderr, "Usage: %s <0|1> <peer_ip>\n", argv[0]); return 1; }
    int rank = atoi(argv[1]);
    const char *peer = argv[2];

    printf("=== GPU-Direct RDMA E2E v4 (pure GPU) ===\n");
    printf("    Rank %d (%s)\n\n", rank, rank ? "RECEIVER" : "SENDER");
    CUDA_CHECK(cudaSetDevice(0));

    // Open EFA
    int nd;
    struct ibv_device **dl = ibv_get_device_list(&nd);
    struct ibv_device *efa = NULL;
    for (int i = 0; i < nd; i++) if (dl[i]->node_type == 7) { efa = dl[i]; break; }
    struct ibv_context *ctx = ibv_open_device(efa);
    printf("[1] EFA: %s\n", ibv_get_device_name(efa));

    union ibv_gid gid;
    ibv_query_gid(ctx, 1, 0, &gid);
    struct ibv_pd *pd = ibv_alloc_pd(ctx);
    struct ibv_cq *cq = ibv_create_cq(ctx, 256, NULL, NULL, 0);

    // QP with RDMA_WRITE flag
    struct ibv_qp_init_attr_ex ax = {};
    ax.qp_type = IBV_QPT_DRIVER;
    ax.send_cq = cq; ax.recv_cq = cq;
    ax.cap.max_send_wr = 256; ax.cap.max_recv_wr = 256;
    ax.cap.max_send_sge = 2; ax.cap.max_recv_sge = 2;
    ax.comp_mask = IBV_QP_INIT_ATTR_PD | IBV_QP_INIT_ATTR_SEND_OPS_FLAGS;
    ax.pd = pd;
    ax.send_ops_flags = IBV_QP_EX_WITH_RDMA_WRITE;
    struct efadv_qp_init_attr ei = {};
    ei.driver_qp_type = EFADV_QP_DRIVER_TYPE_SRD;
    struct ibv_qp *qp = efadv_create_qp_ex(ctx, &ax, &ei, sizeof(ei));
    if (!qp) { perror("QP"); return 1; }

    // QP → RTS
    {
        struct ibv_qp_attr a = {}; a.qp_state = IBV_QPS_INIT;
        a.pkey_index = 0; a.port_num = 1; a.qkey = 0x11111111;
        ibv_modify_qp(qp, &a, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_QKEY);
    }
    { struct ibv_qp_attr a = {}; a.qp_state = IBV_QPS_RTR; ibv_modify_qp(qp, &a, IBV_QP_STATE); }
    { struct ibv_qp_attr a = {}; a.qp_state = IBV_QPS_RTS; a.sq_psn = 0;
      ibv_modify_qp(qp, &a, IBV_QP_STATE | IBV_QP_SQ_PSN); }
    printf("    QP RTS (num=%u)\n", qp->qp_num);

    // Host pinned buffer
    void *buf;
    CUDA_CHECK(cudaMallocHost(&buf, DATA_SIZE));
    if (rank == 0) {
        uint64_t *p = (uint64_t*)buf;
        for (int i = 0; i < DATA_SIZE/8; i++) p[i] = MAGIC + i;
    } else {
        memset(buf, 0, DATA_SIZE);
    }
    struct ibv_mr *mr = ibv_reg_mr(pd, buf, DATA_SIZE,
        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
    printf("    MR: lkey=0x%x rkey=0x%x\n", mr->lkey, mr->rkey);

    // Exchange
    printf("[2] Exchange\n");
    struct qp_info li = {}, ri = {};
    li.qp_num = qp->qp_num; li.gid = gid; li.rkey = mr->rkey; li.remote_addr = (uint64_t)buf;
    exchange(rank, peer, &li, &ri);

    struct ibv_ah_attr aha = {};
    aha.is_global = 1; aha.grh.dgid = ri.gid; aha.port_num = 1;
    struct ibv_ah *ah = ibv_create_ah(pd, &aha);
    struct efadv_ah_attr eah = {};
    efadv_query_ah(ah, &eah, sizeof(eah));
    printf("    Remote: qp=%u rkey=0x%x AH=%u\n", ri.qp_num, ri.rkey, eah.ahn);

    barrier(rank, peer);

    // GPU-direct setup (NO CPU posting on this QP!)
    printf("[3] GPU-direct\n");
    struct efadv_wq_attr sq = {}, rq = {};
    efadv_query_qp_wqs(qp, &sq, &rq, sizeof(sq));
    printf("    SQ: buf=%p entry=%ux%u db=%p batch=%u\n",
           sq.buffer, sq.num_entries, sq.entry_size, sq.doorbell, sq.max_batch);

    cudaHostRegister(sq.buffer, (size_t)sq.entry_size * sq.num_entries, cudaHostRegisterIoMemory);
    cudaHostRegister(rq.buffer, (size_t)rq.entry_size * rq.num_entries, cudaHostRegisterDefault);
    void *sq_dp = (void*)((uintptr_t)sq.doorbell & ~0xFFF);
    void *rq_dp = (void*)((uintptr_t)rq.doorbell & ~0xFFF);
    cudaHostRegister(sq_dp, 4096, cudaHostRegisterIoMemory);
    cudaHostRegister(rq_dp, 4096, cudaHostRegisterIoMemory);

    struct efa_cuda_qp_attrs qa = {};
    qa.sq_buffer = sq.buffer; qa.rq_buffer = rq.buffer;
    qa.sq_doorbell = sq.doorbell; qa.rq_doorbell = rq.doorbell;
    qa.sq_num_entries = sq.num_entries; qa.sq_entry_size = sq.entry_size;
    qa.sq_max_batch = sq.max_batch;
    qa.rq_num_entries = rq.num_entries; qa.rq_entry_size = rq.entry_size;
    qa.reserved = 0;
    struct efa_cuda_qp *d_qp = efa_cuda_create_qp(&qa, sizeof(qa));
    printf("    GPU QP: %s\n", d_qp ? "OK" : "FAILED");

    // Transfer
    printf("[4] Transfer\n");
    if (rank == 0) {
        gpu_rdma_write_kernel<<<1, 1>>>(
            d_qp, ri.rkey, ri.remote_addr,
            mr->lkey, (uint64_t)buf, DATA_SIZE,
            eah.ahn, ri.qp_num, 0x11111111);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Poll CQ from CPU
        struct ibv_wc wc;
        int polls = 0;
        while (ibv_poll_cq(cq, 1, &wc) == 0 && polls < 10000000) polls++;
        if (polls < 10000000) {
            printf("    Completion: status=%d (%s) wr_id=%lu polls=%d\n",
                   wc.status, ibv_wc_status_str(wc.status), wc.wr_id, polls);
            if (wc.status == 0) {
                printf("    *** GPU-DIRECT RDMA WRITE SUCCESS (sender side) ***\n");
            }
        } else {
            printf("    No completion after %d polls\n", polls);
        }
    } else {
        usleep(2000000);  // 2 seconds
        uint64_t *p = (uint64_t*)buf;
        int ok = 0;
        for (int i = 0; i < DATA_SIZE/8; i++) if (p[i] == MAGIC + i) ok++;
        printf("    Verify: %d/%d match\n", ok, (int)(DATA_SIZE/8));
        if (ok == DATA_SIZE/8) {
            printf("    *** GPU-DIRECT RDMA WRITE SUCCESS (receiver verified) ***\n");
        } else {
            printf("    First 4 words: ");
            for (int i = 0; i < 4; i++) printf("0x%lx ", p[i]);
            printf("\n");
        }
    }

    barrier(rank, peer);

    efa_cuda_destroy_qp(d_qp);
    cudaHostUnregister(sq.buffer); cudaHostUnregister(rq.buffer);
    cudaHostUnregister(sq_dp); cudaHostUnregister(rq_dp);
    ibv_destroy_ah(ah); ibv_dereg_mr(mr); ibv_destroy_qp(qp);
    ibv_destroy_cq(cq); ibv_dealloc_pd(pd);
    ibv_close_device(ctx); ibv_free_device_list(dl);
    cudaFreeHost(buf);
    printf("[5] Done.\n");
    return 0;
}

// test_efa_cpu_rdma.c — v2: EFA SRD RDMA write using extended QP API
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <infiniband/verbs.h>
#include <infiniband/efadv.h>

#define OOB_PORT 19877
#define DATA_SIZE 4096
#define MAGIC 0xDEADBEEFCAFEBABEULL

struct info {
    uint32_t qp_num;
    union ibv_gid gid;
    uint32_t rkey;
    uint64_t addr;
};

int exchange(int rank, const char *peer, struct info *l, struct info *r) {
    if (rank == 0) {
        int s = socket(AF_INET, SOCK_STREAM, 0);
        int o = 1; setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &o, sizeof(o));
        struct sockaddr_in a = {};
        a.sin_family = AF_INET; a.sin_port = htons(OOB_PORT); a.sin_addr.s_addr = INADDR_ANY;
        bind(s, (struct sockaddr*)&a, sizeof(a)); listen(s, 1);
        int c = accept(s, NULL, NULL);
        send(c, l, sizeof(*l), 0); recv(c, r, sizeof(*r), MSG_WAITALL);
        close(c); close(s);
    } else {
        int c = socket(AF_INET, SOCK_STREAM, 0);
        struct sockaddr_in a = {};
        a.sin_family = AF_INET; a.sin_port = htons(OOB_PORT);
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
        struct sockaddr_in a = {};
        a.sin_family = AF_INET; a.sin_port = htons(OOB_PORT+1); a.sin_addr.s_addr = INADDR_ANY;
        bind(s, (struct sockaddr*)&a, sizeof(a)); listen(s, 1);
        int c = accept(s, NULL, NULL);
        recv(c, &b, 1, MSG_WAITALL); send(c, &b, 1, 0);
        close(c); close(s);
    } else {
        int c = socket(AF_INET, SOCK_STREAM, 0);
        struct sockaddr_in a = {};
        a.sin_family = AF_INET; a.sin_port = htons(OOB_PORT+1);
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

    printf("=== CPU RDMA Write Test v2, rank=%d ===\n", rank);

    int nd;
    struct ibv_device **dl = ibv_get_device_list(&nd);
    struct ibv_device *efa = NULL;
    for (int i = 0; i < nd; i++) if (dl[i]->node_type == 7) { efa = dl[i]; break; }
    struct ibv_context *ctx = ibv_open_device(efa);
    printf("Device: %s\n", ibv_get_device_name(efa));

    union ibv_gid gid;
    ibv_query_gid(ctx, 1, 0, &gid);

    struct ibv_pd *pd = ibv_alloc_pd(ctx);
    struct ibv_cq *cq = ibv_create_cq(ctx, 256, NULL, NULL, 0);

    // Create QP with extended API for RDMA write support
    struct ibv_qp_init_attr_ex attr_ex = {};
    attr_ex.qp_type = IBV_QPT_DRIVER;
    attr_ex.send_cq = cq;
    attr_ex.recv_cq = cq;
    attr_ex.cap.max_send_wr = 256;
    attr_ex.cap.max_recv_wr = 256;
    attr_ex.cap.max_send_sge = 2;
    attr_ex.cap.max_recv_sge = 2;
    attr_ex.comp_mask = IBV_QP_INIT_ATTR_PD | IBV_QP_INIT_ATTR_SEND_OPS_FLAGS;
    attr_ex.pd = pd;
    attr_ex.send_ops_flags = IBV_QP_EX_WITH_RDMA_WRITE;

    struct efadv_qp_init_attr efa_attr = {};
    efa_attr.driver_qp_type = EFADV_QP_DRIVER_TYPE_SRD;

    struct ibv_qp *qp = efadv_create_qp_ex(ctx, &attr_ex, &efa_attr, sizeof(efa_attr));
    if (!qp) { perror("efadv_create_qp_ex"); return 1; }
    printf("QP created: num=%u\n", qp->qp_num);

    // QP state transitions
    {
        struct ibv_qp_attr a = {};
        a.qp_state = IBV_QPS_INIT; a.pkey_index = 0; a.port_num = 1; a.qkey = 0x11111111;
        ibv_modify_qp(qp, &a, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_QKEY);
    }
    {
        struct ibv_qp_attr a = {}; a.qp_state = IBV_QPS_RTR;
        ibv_modify_qp(qp, &a, IBV_QP_STATE);
    }
    {
        struct ibv_qp_attr a = {}; a.qp_state = IBV_QPS_RTS; a.sq_psn = 0;
        ibv_modify_qp(qp, &a, IBV_QP_STATE | IBV_QP_SQ_PSN);
    }
    printf("QP state: RTS\n");

    // Buffer
    void *buf = malloc(DATA_SIZE);
    if (rank == 0) {
        uint64_t *p = (uint64_t*)buf;
        for (int i = 0; i < DATA_SIZE/8; i++) p[i] = MAGIC + i;
    } else {
        memset(buf, 0, DATA_SIZE);
    }
    struct ibv_mr *mr = ibv_reg_mr(pd, buf, DATA_SIZE,
        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
    printf("MR: lkey=0x%x rkey=0x%x\n", mr->lkey, mr->rkey);

    // Exchange
    struct info li = {}, ri = {};
    li.qp_num = qp->qp_num; li.gid = gid; li.rkey = mr->rkey; li.addr = (uint64_t)buf;
    exchange(rank, peer, &li, &ri);

    struct ibv_ah_attr aha = {};
    aha.is_global = 1; aha.grh.dgid = ri.gid; aha.port_num = 1;
    struct ibv_ah *ah = ibv_create_ah(pd, &aha);

    struct efadv_ah_attr eah = {};
    efadv_query_ah(ah, &eah, sizeof(eah));
    printf("Remote: qp=%u rkey=0x%x addr=0x%lx AH=%u\n",
           ri.qp_num, ri.rkey, ri.addr, eah.ahn);

    barrier(rank, peer);

    if (rank == 0) {
        // Use extended WR builder API for RDMA write on SRD
        struct ibv_qp_ex *qpx = ibv_qp_to_qp_ex(qp);

        ibv_wr_start(qpx);
        qpx->wr_id = 42;
        qpx->wr_flags = IBV_SEND_SIGNALED;

        ibv_wr_rdma_write(qpx, ri.rkey, ri.addr);
        ibv_wr_set_sge(qpx, mr->lkey, (uint64_t)buf, DATA_SIZE);
        ibv_wr_set_ud_addr(qpx, ah, ri.qp_num, 0x11111111);

        int ret = ibv_wr_complete(qpx);
        printf("ibv_wr_complete: %s (ret=%d)\n", ret ? strerror(errno) : "OK", ret);

        // Poll
        struct ibv_wc wc;
        int polls = 0;
        while (ibv_poll_cq(cq, 1, &wc) == 0 && polls < 10000000) polls++;
        if (polls < 10000000) {
            printf("Completion: status=%d (%s) wr_id=%lu polls=%d\n",
                   wc.status, ibv_wc_status_str(wc.status), wc.wr_id, polls);
        } else {
            printf("No completion after %d polls\n", polls);
        }
    } else {
        usleep(500000);
        uint64_t *p = (uint64_t*)buf;
        int ok = 0;
        for (int i = 0; i < DATA_SIZE/8; i++) if (p[i] == MAGIC + i) ok++;
        printf("Verify: %d/%d match\n", ok, (int)(DATA_SIZE/8));
        if (ok == DATA_SIZE/8) printf("*** SUCCESS ***\n");
        else { for (int i = 0; i < 4; i++) printf("[%d]: 0x%016lx\n", i, p[i]); }
    }

    barrier(rank, peer);
    ibv_destroy_ah(ah); ibv_dereg_mr(mr); ibv_destroy_qp(qp);
    ibv_destroy_cq(cq); ibv_dealloc_pd(pd); ibv_close_device(ctx);
    ibv_free_device_list(dl); free(buf);
    printf("Done.\n");
    return 0;
}

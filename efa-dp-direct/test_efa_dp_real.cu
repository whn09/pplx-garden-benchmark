// test_efa_dp_real.cu — v3: Test GPU-direct SQ write with real EFA QP
// CQ uses dummy GPU buffer since CQ BAR registration not yet working

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
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

// Kernel: write WR to real EFA SQ buffer from GPU
__global__ void test_gpu_sq_write(efa_cuda_qp *qp, efa_cuda_cq *cq) {
    if (threadIdx.x != 0) return;

    printf("[GPU] === GPU-Direct EFA Test ===\n");

    bool qp_ok = efa_cuda_is_qp_compatible(qp);
    bool cq_ok = efa_cuda_is_cq_compatible(cq);
    printf("[GPU] QP compatible: %s, CQ compatible: %s\n",
           qp_ok ? "YES" : "NO", cq_ok ? "YES" : "NO");

    if (!qp_ok) {
        printf("[GPU] QP compatibility failed, aborting\n");
        return;
    }

    // Prepare an RDMA write WR
    efa_io_tx_wqe wr_buf;
    memset(&wr_buf, 0, sizeof(wr_buf));

    int ret = efa_cuda_init_rdma_write_wr(&wr_buf, /*wr_id=*/1, /*rkey=*/0xDEAD, /*remote_addr=*/0xBEEF0000);
    printf("[GPU] init_rdma_write_wr: ret=%d\n", ret);

    ret = efa_cuda_wr_set_sge(&wr_buf, /*lkey=*/0x1234, /*addr=*/0x5000, /*length=*/4096);
    printf("[GPU] wr_set_sge: ret=%d\n", ret);

    // Start a batch of 1 — this writes to the real EFA SQ buffer!
    ret = efa_cuda_start_sq_batch(qp, 1);
    printf("[GPU] start_sq_batch(1): ret=%d\n", ret);

    if (ret == 0) {
        // Place WR into the real SQ buffer slot
        ret = efa_cuda_sq_batch_place_wr(qp, 0, &wr_buf);
        printf("[GPU] sq_batch_place_wr(0): ret=%d\n", ret);

        if (ret == 0) {
            printf("[GPU] *** SUCCESS: GPU wrote WR to real EFA SQ buffer! ***\n");
            printf("[GPU] This proves GPU-direct access to EFA hardware queues.\n");
        }

        // DO NOT flush — the WR has bogus addresses and would cause hw error
        // In a real scenario, flush_sq_wrs would ring the doorbell
        printf("[GPU] Skipping flush (no real connection setup)\n");
    }

    // Test: try posting a receive WR to RQ
    ret = efa_cuda_post_recv_wr(qp, /*addr=*/0x7000, /*length=*/2048, /*lkey=*/0xABCD);
    printf("[GPU] post_recv_wr: ret=%d\n", ret);
    if (ret == 0) {
        printf("[GPU] *** SUCCESS: GPU wrote recv WR to real EFA RQ buffer! ***\n");
    }
    // Don't flush RQ either

    // CQ poll on dummy buffer (will be NULL since no real completions)
    if (cq_ok) {
        void *wc = efa_cuda_cq_poll(cq, 0);
        printf("[GPU] CQ poll(0): %s\n", wc ? "non-NULL" : "NULL (expected)");
    }

    printf("[GPU] === Test Complete ===\n");
}

int try_gpu_register(void *ptr, size_t size, unsigned int flags, const char *name) {
    cudaError_t err = cudaHostRegister(ptr, size, flags);
    if (err == cudaSuccess) {
        const char *flag_name = (flags == cudaHostRegisterIoMemory) ? "IoMemory" : "Default";
        printf("    %s: OK (%s, ptr=%p, %zu bytes)\n", name, flag_name, ptr, size);
        return 0;
    }
    printf("    %s: FAILED (%s)\n", name, cudaGetErrorString(err));
    return -1;
}

int main() {
    printf("=== efa-dp-direct GPU-Direct SQ Write Test (v3) ===\n\n");

    // 1. Find EFA device
    printf("[1] Finding EFA device\n");
    int num_devices;
    struct ibv_device **dev_list = ibv_get_device_list(&num_devices);
    if (!dev_list) { fprintf(stderr, "No devices\n"); return 1; }

    struct ibv_device *efa_dev = NULL;
    for (int i = 0; i < num_devices; i++) {
        if (dev_list[i]->node_type == 7) {
            efa_dev = dev_list[i];
            break;
        }
    }
    if (!efa_dev) { fprintf(stderr, "No EFA device\n"); return 1; }
    printf("    EFA: %s\n", ibv_get_device_name(efa_dev));

    // 2. Open + query
    printf("\n[2] Opening device\n");
    struct ibv_context *ctx = ibv_open_device(efa_dev);
    if (!ctx) { perror("open"); return 1; }

    struct efadv_device_attr da = {};
    efadv_query_device(ctx, &da, sizeof(da));
    printf("    caps: 0x%x, max_sq_wr=%u, inline=%u, max_rdma=%u\n",
           da.device_caps, da.max_sq_wr, da.inline_buf_size, da.max_rdma_size);

    // 3. PD + CQ + QP
    struct ibv_pd *pd = ibv_alloc_pd(ctx);
    struct ibv_cq *cq = ibv_create_cq(ctx, 256, NULL, NULL, 0);
    if (!pd || !cq) { fprintf(stderr, "PD/CQ failed\n"); return 1; }

    struct ibv_qp_init_attr qi = {};
    qi.qp_type = IBV_QPT_DRIVER;
    qi.send_cq = cq;
    qi.recv_cq = cq;
    qi.cap.max_send_wr = 256;
    qi.cap.max_recv_wr = 256;
    qi.cap.max_send_sge = 1;
    qi.cap.max_recv_sge = 1;

    struct ibv_qp *qp = efadv_create_driver_qp(pd, &qi, EFADV_QP_DRIVER_TYPE_SRD);
    if (!qp) { perror("QP"); return 1; }
    printf("    PD+CQ+QP: OK (qp_num=%u)\n", qp->qp_num);

    // 4. Query hardware queue pointers
    printf("\n[3] Querying hardware queue pointers\n");
    struct efadv_wq_attr sq = {}, rq = {};
    int ret = efadv_query_qp_wqs(qp, &sq, &rq, sizeof(sq));
    if (ret) { fprintf(stderr, "query_qp_wqs failed: %d\n", ret); return 1; }

    printf("    SQ: buf=%p entry=%u×%u db=%p batch=%u\n",
           sq.buffer, sq.num_entries, sq.entry_size, sq.doorbell, sq.max_batch);
    printf("    RQ: buf=%p entry=%u×%u db=%p batch=%u\n",
           rq.buffer, rq.num_entries, rq.entry_size, rq.doorbell, rq.max_batch);

    struct efadv_cq_attr cqa = {};
    ret = efadv_query_cq(cq, &cqa, sizeof(cqa));
    if (ret == 0) {
        printf("    CQ: buf=%p entry=%u×%u\n", cqa.buffer, cqa.num_entries, cqa.entry_size);
    }

    // 5. Register for GPU
    printf("\n[4] GPU registration (requires sudo for IoMemory)\n");
    CUDA_CHECK(cudaSetDevice(0));

    size_t sq_sz = (size_t)sq.entry_size * sq.num_entries;
    size_t rq_sz = (size_t)rq.entry_size * rq.num_entries;

    int sq_ok = try_gpu_register(sq.buffer, sq_sz, cudaHostRegisterIoMemory, "SQ buffer");
    int rq_ok = try_gpu_register(rq.buffer, rq_sz, cudaHostRegisterDefault, "RQ buffer");

    void *sq_db_page = (void*)((uintptr_t)sq.doorbell & ~0xFFF);
    void *rq_db_page = (void*)((uintptr_t)rq.doorbell & ~0xFFF);
    int sq_db_ok = try_gpu_register(sq_db_page, 4096, cudaHostRegisterIoMemory, "SQ doorbell");
    int rq_db_ok = try_gpu_register(rq_db_page, 4096, cudaHostRegisterIoMemory, "RQ doorbell");

    if (sq_ok || rq_ok || sq_db_ok || rq_db_ok) {
        fprintf(stderr, "\n    Some registrations failed. Run with sudo.\n");
        goto cleanup;
    }

    // 6. Create efa-dp-direct GPU QP
    printf("\n[5] Creating efa-dp-direct GPU QP\n");
    {
        struct efa_cuda_qp_attrs a = {};
        a.sq_buffer = sq.buffer;
        a.rq_buffer = rq.buffer;
        a.sq_doorbell = sq.doorbell;
        a.rq_doorbell = rq.doorbell;
        a.sq_num_entries = sq.num_entries;
        a.sq_entry_size = sq.entry_size;
        a.sq_max_batch = sq.max_batch;
        a.rq_num_entries = rq.num_entries;
        a.rq_entry_size = rq.entry_size;
        a.reserved = 0;

        struct efa_cuda_qp *d_qp = efa_cuda_create_qp(&a, sizeof(a));
        if (!d_qp) { fprintf(stderr, "GPU QP failed\n"); goto cleanup; }
        printf("    GPU QP: OK\n");

        // Create dummy GPU CQ (not connected to real hardware, just for API completeness)
        uint8_t *d_cq_buf;
        uint32_t cq_entries = 256;
        uint32_t cq_entry_size = 32; // efa_io_rx_cdesc_ex size
        CUDA_CHECK(cudaMalloc(&d_cq_buf, cq_entries * cq_entry_size));
        CUDA_CHECK(cudaMemset(d_cq_buf, 0, cq_entries * cq_entry_size));

        struct efa_cuda_cq_attrs ca = {};
        ca.buffer = d_cq_buf;
        ca.num_entries = cq_entries;
        ca.entry_size = cq_entry_size;
        struct efa_cuda_cq *d_cq = efa_cuda_create_cq(&ca, sizeof(ca));
        printf("    GPU CQ (dummy): %s\n", d_cq ? "OK" : "FAILED");

        // 7. Run GPU kernel
        printf("\n[6] Running GPU-direct SQ write kernel\n");
        test_gpu_sq_write<<<1, 1>>>(d_qp, d_cq);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Read back QP state to verify writes happened
        struct efa_cuda_qp h_qp;
        CUDA_CHECK(cudaMemcpy(&h_qp, d_qp, sizeof(h_qp), cudaMemcpyDeviceToHost));
        printf("\n[7] QP state after GPU writes:\n");
        printf("    SQ: pc=%u wqes_pending=%u wqes_posted=%u\n",
               h_qp.sq.wq.pc, h_qp.sq.wq.wqes_pending, h_qp.sq.wq.wqes_posted);
        printf("    RQ: pc=%u wqes_pending=%u wqes_posted=%u\n",
               h_qp.rq.wq.pc, h_qp.rq.wq.wqes_pending, h_qp.rq.wq.wqes_posted);

        if (d_cq) efa_cuda_destroy_cq(d_cq);
        efa_cuda_destroy_qp(d_qp);
        cudaFree(d_cq_buf);
    }

cleanup:
    printf("\n[8] Cleanup\n");
    if (sq_ok == 0) cudaHostUnregister(sq.buffer);
    if (rq_ok == 0) cudaHostUnregister(rq.buffer);
    if (sq_db_ok == 0) cudaHostUnregister(sq_db_page);
    if (rq_db_ok == 0) cudaHostUnregister(rq_db_page);

    ibv_destroy_qp(qp);
    ibv_destroy_cq(cq);
    ibv_dealloc_pd(pd);
    ibv_close_device(ctx);
    ibv_free_device_list(dev_list);
    printf("    Done.\n\n");
    return 0;
}

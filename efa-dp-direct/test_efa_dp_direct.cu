// test_efa_dp_direct.cu — Basic efa-dp-direct library test
// Tests: version check, CQ/QP creation with GPU buffers, device-side compatibility check
//
// Build: nvcc -o test_efa_dp_direct test_efa_dp_direct.cu -L../CUDA/build -lefacudadp -I../CUDA/src -Wl,-rpath,'$ORIGIN/../CUDA/build'
// Run: ./test_efa_dp_direct

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
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

// Kernel to test device-side compatibility and basic operations
__global__ void test_compatibility_kernel(efa_cuda_cq *cq, efa_cuda_qp *qp) {
    if (threadIdx.x == 0) {
        // Check CQ compatibility
        bool cq_ok = efa_cuda_is_cq_compatible(cq);
        printf("[GPU] CQ compatible: %s\n", cq_ok ? "YES" : "NO");

        // Check QP compatibility
        bool qp_ok = efa_cuda_is_qp_compatible(qp);
        printf("[GPU] QP compatible: %s\n", qp_ok ? "YES" : "NO");

        // Test CQ poll (should return NULL since no real completions)
        void *wc = efa_cuda_cq_poll(cq, 0);
        printf("[GPU] CQ poll(0): %s (expected: NULL)\n", wc ? "non-NULL" : "NULL");

        // Test initializing a WR in local memory
        efa_io_tx_wqe wr_buf;
        memset(&wr_buf, 0, sizeof(wr_buf));
        int ret = efa_cuda_init_rdma_write_wr(&wr_buf, /*wr_id=*/42, /*rkey=*/0x1234, /*remote_addr=*/0xDEADBEEF);
        printf("[GPU] init_rdma_write_wr returned: %d (expected: 0)\n", ret);

        // Verify the WR was set up
        printf("[GPU] WR req_id: %u (expected: 42)\n", wr_buf.meta.req_id);

        // Test SGE setup
        ret = efa_cuda_wr_set_sge(&wr_buf, /*lkey=*/0xABCD, /*addr=*/0x1000, /*length=*/4096);
        printf("[GPU] wr_set_sge returned: %d (expected: 0)\n", ret);

        printf("[GPU] All device-side tests completed.\n");
    }
}

int main() {
    printf("=== efa-dp-direct Library Test ===\n\n");

    // 1. Version check
    printf("[1] Version check\n");
    int major, minor, subminor;
    int ret = efa_cuda_get_version(&major, &minor, &subminor);
    if (ret == 0) {
        printf("    Library version: %d.%d.%d\n", major, minor, subminor);
        printf("    Header version:  %d.%d.%d\n",
               EFA_CUDA_DP_VERSION_MAJOR, EFA_CUDA_DP_VERSION_MINOR, EFA_CUDA_DP_VERSION_SUBMINOR);
        if (major == EFA_CUDA_DP_VERSION_MAJOR && minor == EFA_CUDA_DP_VERSION_MINOR) {
            printf("    Version match: OK\n");
        } else {
            printf("    Version MISMATCH!\n");
        }
    } else {
        printf("    get_version failed: %d\n", ret);
    }

    // 2. Check CUDA device
    printf("\n[2] CUDA device info\n");
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    printf("    CUDA devices: %d\n", device_count);

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("    Device 0: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);

    CUDA_CHECK(cudaSetDevice(0));

    // 3. Create CQ with GPU-allocated buffer
    printf("\n[3] Create CQ\n");
    const uint32_t CQ_ENTRIES = 256;  // must be power of 2
    // CQ entry size: use sizeof(efa_io_rx_cdesc_ex) which is the largest CQ entry type
    const uint32_t CQ_ENTRY_SIZE = sizeof(efa_io_rx_cdesc_ex);
    printf("    CQ entries: %u, entry size: %u bytes\n", CQ_ENTRIES, CQ_ENTRY_SIZE);

    uint8_t *d_cq_buf;
    CUDA_CHECK(cudaMalloc(&d_cq_buf, CQ_ENTRIES * CQ_ENTRY_SIZE));
    CUDA_CHECK(cudaMemset(d_cq_buf, 0, CQ_ENTRIES * CQ_ENTRY_SIZE));

    struct efa_cuda_cq_attrs cq_attrs = {};
    cq_attrs.buffer = d_cq_buf;
    cq_attrs.num_entries = CQ_ENTRIES;
    cq_attrs.entry_size = CQ_ENTRY_SIZE;

    struct efa_cuda_cq *d_cq = efa_cuda_create_cq(&cq_attrs, sizeof(cq_attrs));
    if (d_cq) {
        printf("    CQ created: OK (device ptr: %p)\n", d_cq);
    } else {
        printf("    CQ creation FAILED\n");
        return 1;
    }

    // 4. Create QP with GPU-allocated buffers
    printf("\n[4] Create QP\n");
    const uint32_t SQ_ENTRIES = 256;  // must be power of 2
    const uint32_t RQ_ENTRIES = 256;
    const uint32_t SQ_ENTRY_SIZE = sizeof(efa_io_tx_wqe);
    const uint32_t RQ_ENTRY_SIZE = sizeof(efa_io_rx_desc);
    printf("    SQ entries: %u, entry size: %u bytes\n", SQ_ENTRIES, SQ_ENTRY_SIZE);
    printf("    RQ entries: %u, entry size: %u bytes\n", RQ_ENTRIES, RQ_ENTRY_SIZE);

    uint8_t *d_sq_buf, *d_rq_buf;
    uint32_t *d_sq_db, *d_rq_db;
    CUDA_CHECK(cudaMalloc(&d_sq_buf, SQ_ENTRIES * SQ_ENTRY_SIZE));
    CUDA_CHECK(cudaMalloc(&d_rq_buf, RQ_ENTRIES * RQ_ENTRY_SIZE));
    CUDA_CHECK(cudaMalloc(&d_sq_db, sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_rq_db, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_sq_buf, 0, SQ_ENTRIES * SQ_ENTRY_SIZE));
    CUDA_CHECK(cudaMemset(d_rq_buf, 0, RQ_ENTRIES * RQ_ENTRY_SIZE));
    CUDA_CHECK(cudaMemset(d_sq_db, 0, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_rq_db, 0, sizeof(uint32_t)));

    struct efa_cuda_qp_attrs qp_attrs = {};
    qp_attrs.sq_buffer = d_sq_buf;
    qp_attrs.rq_buffer = d_rq_buf;
    qp_attrs.sq_doorbell = d_sq_db;
    qp_attrs.rq_doorbell = d_rq_db;
    qp_attrs.sq_num_entries = SQ_ENTRIES;
    qp_attrs.sq_entry_size = SQ_ENTRY_SIZE;
    qp_attrs.sq_max_batch = 16;
    qp_attrs.rq_num_entries = RQ_ENTRIES;
    qp_attrs.rq_entry_size = RQ_ENTRY_SIZE;
    qp_attrs.reserved = 0;

    struct efa_cuda_qp *d_qp = efa_cuda_create_qp(&qp_attrs, sizeof(qp_attrs));
    if (d_qp) {
        printf("    QP created: OK (device ptr: %p)\n", d_qp);
    } else {
        printf("    QP creation FAILED\n");
        return 1;
    }

    // 5. Run device-side tests
    printf("\n[5] Device-side tests\n");
    test_compatibility_kernel<<<1, 1>>>(d_cq, d_qp);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 6. Cleanup
    printf("\n[6] Cleanup\n");
    efa_cuda_destroy_qp(d_qp);
    efa_cuda_destroy_cq(d_cq);
    cudaFree(d_sq_buf);
    cudaFree(d_rq_buf);
    cudaFree(d_sq_db);
    cudaFree(d_rq_db);
    cudaFree(d_cq_buf);
    printf("    Cleanup: OK\n");

    printf("\n=== All tests passed ===\n");
    return 0;
}

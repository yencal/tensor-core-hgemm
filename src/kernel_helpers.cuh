// kernel_helpers.cuh
// Reusable functions for WMMA HGEMM kernels (FP16)
//
// NOTE: All loadTileB functions expect B in standard layout B[K,N] row-major.
//       B uses stride N for global loads, and Bs is laid out as Bs[BK][BN] in shared memory.

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cuda_pipeline.h>

using namespace nvcuda;

// =========================================================================
// Tile Loading A: Scalar
// =========================================================================

template <int BM, int BK, int NUM_THREADS>
__device__ void loadTileA_scalar(
    const __half *A,
    __half *As,
    int K,
    uint tid)
{
    constexpr int ELEMS_PER_THREAD = (BM * BK) / NUM_THREADS;

    #pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
        uint idx = tid + i * NUM_THREADS;
        uint row = idx / BK;
        uint col = idx % BK;
        As[row * BK + col] = A[row * K + col];
    }
}

// =========================================================================
// Tile Loading B: Scalar (B is [K,N] row-major)
// =========================================================================

template <int BK, int BN, int NUM_THREADS>
__device__ void loadTileB_scalar(
    const __half *B,   // B[K,N] row-major, pointing to tile start
    __half *Bs,        // Bs[BK][BN]
    int N,
    uint tid)
{
    constexpr int ELEMS_PER_THREAD = (BK * BN) / NUM_THREADS;

    #pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
        uint idx = tid + i * NUM_THREADS;
        uint row = idx / BN;
        uint col = idx % BN;
        Bs[row * BN + col] = B[row * N + col];
    }
}

// =========================================================================
// Tile Loading A: Vectorized float4 = 8 halves
// =========================================================================

template <int BM, int BK, int NUM_THREADS>
__device__ void loadTileA_vec4(
    const __half *A,
    __half *As,
    int K,
    uint tid)
{
    constexpr int TOTAL_VEC = (BM * BK) / 8;
    constexpr int VEC_PER_THREAD = TOTAL_VEC / NUM_THREADS;

    static_assert((BM * BK) % 8 == 0, "Tile size must be divisible by 8");
    static_assert(TOTAL_VEC % NUM_THREADS == 0, "vec count must be divisible by NUM_THREADS");
    static_assert(BK % 8 == 0, "BK must be divisible by 8 for vectorized loads");

    #pragma unroll
    for (int i = 0; i < VEC_PER_THREAD; ++i) {
        uint idx = tid + i * NUM_THREADS;
        uint vec_per_row = BK / 8;
        uint row = idx / vec_per_row;
        uint col8 = idx % vec_per_row;

        float4 val = reinterpret_cast<const float4*>(&A[row * K + col8 * 8])[0];
        reinterpret_cast<float4*>(&As[row * BK + col8 * 8])[0] = val;
    }
}

// =========================================================================
// Tile Loading B: Vectorized float4 = 8 halves (B is [K,N] row-major)
// =========================================================================

template <int BK, int BN, int NUM_THREADS>
__device__ void loadTileB_vec4(
    const __half *B,   // B[K,N] row-major, pointing to tile start
    __half *Bs,        // Bs[BK][BN]
    int N,
    uint tid)
{
    constexpr int TOTAL_VEC = (BK * BN) / 8;
    constexpr int VEC_PER_THREAD = TOTAL_VEC / NUM_THREADS;

    static_assert((BK * BN) % 8 == 0, "Tile size must be divisible by 8");
    static_assert(TOTAL_VEC % NUM_THREADS == 0, "vec count must be divisible by NUM_THREADS");
    static_assert(BN % 8 == 0, "BN must be divisible by 8 for vectorized loads");

    #pragma unroll
    for (int i = 0; i < VEC_PER_THREAD; ++i) {
        uint idx = tid + i * NUM_THREADS;
        uint vec_per_row = BN / 8;
        uint row = idx / vec_per_row;
        uint col8 = idx % vec_per_row;

        float4 val = reinterpret_cast<const float4*>(&B[row * N + col8 * 8])[0];
        reinterpret_cast<float4*>(&Bs[row * BN + col8 * 8])[0] = val;
    }
}

// =========================================================================
// Tile Loading A: Async (cp.async, 16 bytes = 8 halves per copy)
// =========================================================================

template <int BM, int BK, int NUM_THREADS>
__device__ void loadTileA_async(
    const __half *A,
    __half *As,
    int K,
    uint tid)
{
    constexpr int TOTAL_VEC = (BM * BK) / 8;
    constexpr int VEC_PER_THREAD = TOTAL_VEC / NUM_THREADS;

    #pragma unroll
    for (int i = 0; i < VEC_PER_THREAD; ++i) {
        uint idx = tid + i * NUM_THREADS;
        uint row = idx / (BK / 8);
        uint col8 = idx % (BK / 8);
        __pipeline_memcpy_async(
            &As[row * BK + col8 * 8],
            &A[row * K + col8 * 8],
            sizeof(float4)
        );
    }
}

// =========================================================================
// Tile Loading B: Async (B is [K,N] row-major)
// =========================================================================

template <int BK, int BN, int NUM_THREADS>
__device__ void loadTileB_async(
    const __half *B,   // B[K,N] row-major, pointing to tile start
    __half *Bs,        // Bs[BK][BN]
    int N,
    uint tid)
{
    constexpr int TOTAL_VEC = (BK * BN) / 8;
    constexpr int VEC_PER_THREAD = TOTAL_VEC / NUM_THREADS;

    #pragma unroll
    for (int i = 0; i < VEC_PER_THREAD; ++i) {
        uint idx = tid + i * NUM_THREADS;
        uint row = idx / (BN / 8);
        uint col8 = idx % (BN / 8);
        __pipeline_memcpy_async(
            &Bs[row * BN + col8 * 8],
            &B[row * N + col8 * 8],
            sizeof(float4)
        );
    }
}

// =========================================================================
// Tile Loading A: Async with padding
// =========================================================================

template <int BM, int BK, int A_STRIDE, int NUM_THREADS>
__device__ void loadTileA_async_padded(
    const __half *A,
    __half *As,
    int K,
    uint tid)
{
    constexpr int TOTAL_VEC = (BM * BK) / 8;
    constexpr int VEC_PER_THREAD = TOTAL_VEC / NUM_THREADS;

    #pragma unroll
    for (int i = 0; i < VEC_PER_THREAD; ++i) {
        uint idx = tid + i * NUM_THREADS;
        uint row = idx / (BK / 8);
        uint col8 = idx % (BK / 8);

        __pipeline_memcpy_async(
            &As[row * A_STRIDE + col8 * 8],
            &A[row * K + col8 * 8],
            sizeof(float4)
        );
    }
}

// =========================================================================
// Tile Loading B: Async with padding (B is [K,N] row-major)
// =========================================================================

template <int BK, int BN, int B_STRIDE, int NUM_THREADS>
__device__ void loadTileB_async_padded(
    const __half *B,   // B[K,N] row-major, pointing to tile start
    __half *Bs,        // Bs[BK][B_STRIDE]
    int N,
    uint tid)
{
    constexpr int TOTAL_VEC = (BK * BN) / 8;
    constexpr int VEC_PER_THREAD = TOTAL_VEC / NUM_THREADS;

    #pragma unroll
    for (int i = 0; i < VEC_PER_THREAD; ++i) {
        uint idx = tid + i * NUM_THREADS;
        uint row = idx / (BN / 8);
        uint col8 = idx % (BN / 8);

        __pipeline_memcpy_async(
            &Bs[row * B_STRIDE + col8 * 8],
            &B[row * N + col8 * 8],
            sizeof(float4)
        );
    }
}

// =========================================================================
// Tile Loading A: Warp-based async (better locality, generalized BK)
// =========================================================================

template <int BM, int BK, int A_STRIDE, int NUM_THREADS>
__device__ void loadTileA_warp_based(
    const __half *A,
    __half *As,
    int K,
    uint tid)
{
    // Constants
    constexpr int WARPS = NUM_THREADS / 32;
    constexpr int ROWS_PER_WARP = BM / WARPS;
    constexpr int THREADS_PER_ROW = BK / 8;              // Each thread loads 8 halves (16 bytes)
    constexpr int ROWS_PER_ITER = 32 / THREADS_PER_ROW;  // Rows covered per iteration
    constexpr int ITERS = ROWS_PER_WARP / ROWS_PER_ITER; // Iterations per warp

    static_assert(BK % 8 == 0, "BK must be divisible by 8");
    static_assert(32 % THREADS_PER_ROW == 0, "Warp size must be divisible by THREADS_PER_ROW");
    static_assert(ROWS_PER_WARP % ROWS_PER_ITER == 0, "ROWS_PER_WARP must be divisible by ROWS_PER_ITER");

    // Thread mapping within warp
    const uint warp_id = tid / 32;
    const uint lane_id = tid % 32;
    const uint row_in_chunk = lane_id / THREADS_PER_ROW;
    const uint col_in_row = lane_id % THREADS_PER_ROW;

    // Global memory base for this warp
    const __half* A_warp_ptr = A + warp_id * ROWS_PER_WARP * K;

    // Shared memory base row for this warp
    uint smem_row = warp_id * ROWS_PER_WARP + row_in_chunk;

    #pragma unroll
    for (int i = 0; i < ITERS; ++i) {
        // Async copy 16 bytes (8 halves)
        __pipeline_memcpy_async(
            &As[smem_row * A_STRIDE + col_in_row * 8],
            A_warp_ptr + row_in_chunk * K + col_in_row * 8,
            sizeof(float4)
        );

        // Advance to next chunk of rows
        A_warp_ptr += ROWS_PER_ITER * K;
        smem_row += ROWS_PER_ITER;
    }
}

// =========================================================================
// Tile Loading B: Warp-based async (B is [K,N] row-major, generalized BN)
// =========================================================================

template <int BK, int BN, int B_STRIDE, int NUM_THREADS>
__device__ void loadTileB_warp_based(
    const __half *B,   // B[K,N] row-major, pointing to tile start
    __half *Bs,        // Bs[BK][B_STRIDE]
    int N,
    uint tid)
{
    // Constants
    constexpr int WARPS = NUM_THREADS / 32;
    constexpr int ROWS_PER_WARP = BK / WARPS;
    constexpr int THREADS_PER_ROW = BN / 8;              // Each thread loads 8 halves (16 bytes)
    constexpr int ROWS_PER_ITER = 32 / THREADS_PER_ROW;  // Rows covered per iteration
    constexpr int ITERS = ROWS_PER_WARP / ROWS_PER_ITER; // Iterations per warp

    static_assert(BN % 8 == 0, "BN must be divisible by 8");
    static_assert(32 % THREADS_PER_ROW == 0, "Warp size must be divisible by THREADS_PER_ROW");
    static_assert(ROWS_PER_WARP % ROWS_PER_ITER == 0, "ROWS_PER_WARP must be divisible by ROWS_PER_ITER");

    // Thread mapping within warp
    const uint warp_id = tid / 32;
    const uint lane_id = tid % 32;
    const uint row_in_chunk = lane_id / THREADS_PER_ROW;
    const uint col_in_row = lane_id % THREADS_PER_ROW;

    // Global memory base for this warp
    const __half* B_warp_ptr = B + warp_id * ROWS_PER_WARP * N;

    // Shared memory base row for this warp
    uint smem_row = warp_id * ROWS_PER_WARP + row_in_chunk;

    #pragma unroll
    for (int i = 0; i < ITERS; ++i) {
        // Async copy 16 bytes (8 halves)
        __pipeline_memcpy_async(
            &Bs[smem_row * B_STRIDE + col_in_row * 8],
            B_warp_ptr + row_in_chunk * N + col_in_row * 8,
            sizeof(float4)
        );

        // Advance to next chunk of rows
        B_warp_ptr += ROWS_PER_ITER * N;
        smem_row += ROWS_PER_ITER;
    }
}

// =========================================================================
// Epilogue: C = alpha * acc + beta * C (FP16 accumulator)
// =========================================================================

template <int MMA_M, int MMA_N, int MMA_K, int MMA_M_TILES, int MMA_N_TILES, int WM, int WN>
__device__ void epilogueAndStore(
    wmma::fragment<wmma::accumulator, MMA_M, MMA_N, MMA_K, __half> acc[MMA_M_TILES][MMA_N_TILES],
    __half *C,
    int N,
    __half alpha,
    __half beta,
    uint warpM,
    uint warpN)
{
    #pragma unroll
    for (int m = 0; m < MMA_M_TILES; ++m) {
        #pragma unroll
        for (int n = 0; n < MMA_N_TILES; ++n) {
            __half *C_ptr = C + (warpM * WM + m * MMA_M) * N
                              + (warpN * WN + n * MMA_N);

            #pragma unroll
            for (int i = 0; i < acc[m][n].num_elements; ++i) {
                acc[m][n].x[i] = __hmul(acc[m][n].x[i], alpha);
            }

            if (__heq(beta, __float2half(0.0f)) == false) {
                wmma::fragment<wmma::accumulator, MMA_M, MMA_N, MMA_K, __half> c_frag;
                wmma::load_matrix_sync(c_frag, C_ptr, N, wmma::mem_row_major);

                #pragma unroll
                for (int i = 0; i < acc[m][n].num_elements; ++i) {
                    acc[m][n].x[i] = __hadd(acc[m][n].x[i], __hmul(beta, c_frag.x[i]));
                }
            }

            wmma::store_matrix_sync(C_ptr, acc[m][n], N, wmma::mem_row_major);
        }
    }
}

// =========================================================================
// Vectorized Epilogue (reuses existing shared memory)
// =========================================================================

template <int BM, int BN, int WM, int WN,
          int MMA_M, int MMA_N, int MMA_K,
          int MMA_M_TILES, int MMA_N_TILES,
          int WARPS_M, int WARPS_N,
          int C_SMEM_STRIDE>
__device__ void epilogueAndStore_vec4(
    wmma::fragment<wmma::accumulator, MMA_M, MMA_N, MMA_K, __half> acc[MMA_M_TILES][MMA_N_TILES],
    __half *smem,
    __half *C,
    int N,
    __half alpha,
    __half beta,
    uint tid,
    uint warpM,
    uint warpN)
{
    __half *C_smem = smem;

    // Scale by alpha and handle beta
    #pragma unroll
    for (int m = 0; m < MMA_M_TILES; ++m) {
        #pragma unroll
        for (int n = 0; n < MMA_N_TILES; ++n) {
            #pragma unroll
            for (int i = 0; i < acc[m][n].num_elements; ++i) {
                acc[m][n].x[i] = __hmul(acc[m][n].x[i], alpha);
            }

            if (__heq(beta, __float2half(0.0f)) == false) {
                __half *C_ptr = C + (warpM * WM + m * MMA_M) * N
                                  + (warpN * WN + n * MMA_N);
                wmma::fragment<wmma::accumulator, MMA_M, MMA_N, MMA_K, __half> c_frag;
                wmma::load_matrix_sync(c_frag, C_ptr, N, wmma::mem_row_major);

                #pragma unroll
                for (int i = 0; i < acc[m][n].num_elements; ++i) {
                    acc[m][n].x[i] = __hadd(acc[m][n].x[i], __hmul(beta, c_frag.x[i]));
                }
            }
        }
    }

    // Store fragments to shared memory
    #pragma unroll
    for (int m = 0; m < MMA_M_TILES; ++m) {
        #pragma unroll
        for (int n = 0; n < MMA_N_TILES; ++n) {
            __half *C_smem_ptr = &C_smem[(warpM * WM + m * MMA_M) * C_SMEM_STRIDE
                                        + (warpN * WN + n * MMA_N)];
            wmma::store_matrix_sync(C_smem_ptr, acc[m][n], C_SMEM_STRIDE, wmma::mem_row_major);
        }
    }

    __syncthreads();

    // Vectorized copy from shared to global
    constexpr int NUM_THREADS = WARPS_M * WARPS_N * 32;
    constexpr int TOTAL_ELEMENTS = BM * BN;
    constexpr int ELEMENTS_PER_VEC = 8;
    constexpr int TOTAL_VECS = TOTAL_ELEMENTS / ELEMENTS_PER_VEC;
    constexpr int VECS_PER_THREAD = TOTAL_VECS / NUM_THREADS;

    static_assert(BN % 8 == 0, "BN must be divisible by 8 for vectorized stores");
    static_assert(TOTAL_VECS % NUM_THREADS == 0, "Vectors must divide evenly among threads");

    #pragma unroll
    for (int i = 0; i < VECS_PER_THREAD; ++i) {
        int vec_idx = tid + i * NUM_THREADS;

        int vecs_per_row = BN / ELEMENTS_PER_VEC;
        int row = vec_idx / vecs_per_row;
        int col8 = vec_idx % vecs_per_row;

        float4 val = *reinterpret_cast<float4*>(&C_smem[row * C_SMEM_STRIDE + col8 * 8]);
        *reinterpret_cast<float4*>(&C[row * N + col8 * 8]) = val;
    }
}

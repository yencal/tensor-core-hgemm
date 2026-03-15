// 07_wmma_final.cuh
// WMMA HGEMM with interleaved load/compute pipeline
//   - Fragment double buffering with async copy between k-steps
//   - Block swizzle for L2 cache locality
//   - Vectorized epilogue through shared memory
//
// NOTE: B is in standard layout B[K,N] row-major.
//       BK must be a multiple of MMA_K (16), with even K_STEPS = BK/16.

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_pipeline.h>
#include <mma.h>
#include "kernel_helpers.cuh"

using namespace nvcuda;

template <int BM, int BN, int BK, int WM, int WN, int STAGES,
          bool USE_SWIZZLE = false, int BLOCK_STRIDE = 16>
__global__ void wmma_final(
    int M, int N, int K, __half alpha,
    const __half *__restrict__ A, const __half *__restrict__ B,
    __half beta, __half *__restrict__ C)
{
    constexpr int MMA_M = 16, MMA_N = 16, MMA_K = 16;
    constexpr int K_STEPS = BK / MMA_K;

    // Compile-time validation
    static_assert(BK % MMA_K == 0, "BK must be multiple of MMA_K (16)");
    static_assert(K_STEPS >= 1 && K_STEPS <= 8, "K_STEPS must be 1-8");
    static_assert(K_STEPS % 2 == 0, "K_STEPS must be even for double-buffering (BK must be multiple of 32)");

    constexpr int MMA_M_TILES = WM / MMA_M;
    constexpr int MMA_N_TILES = WN / MMA_N;
    constexpr int WARPS_M = BM / WM;
    constexpr int WARPS_N = BN / WN;
    constexpr int NUM_THREADS = WARPS_M * WARPS_N * 32;

    constexpr int SMEM_PAD = 8;
    constexpr int A_STRIDE = BK + SMEM_PAD;
    constexpr int B_STRIDE = BN + SMEM_PAD;
    constexpr int A_STAGE_SIZE = BM * A_STRIDE;
    constexpr int B_STAGE_SIZE = BK * B_STRIDE;

    constexpr int C_SMEM_PAD = 8;
    constexpr int C_SMEM_STRIDE = BN + C_SMEM_PAD;

    extern __shared__ __half smem[];
    __half* As = smem;
    __half* Bs = smem + STAGES * A_STAGE_SIZE;

    const uint tid = threadIdx.x;
    const uint warpId = tid / 32;
    const uint warpM = warpId / WARPS_N;
    const uint warpN = warpId % WARPS_N;

    // Block mapping (swizzle or simple)
    uint blockM, blockN;
    if constexpr (USE_SWIZZLE) {
        const uint numBlocksM = (M + BM - 1) / BM;
        const uint numBlocksN = (N + BN - 1) / BN;
        const uint numBlocksInGroup = BLOCK_STRIDE * numBlocksN;
        const uint groupId = blockIdx.x / numBlocksInGroup;
        const uint firstBlockM = groupId * BLOCK_STRIDE;
        const uint groupSizeM = min(numBlocksM - firstBlockM, (uint)BLOCK_STRIDE);
        blockM = firstBlockM + (blockIdx.x % groupSizeM);
        blockN = (blockIdx.x % numBlocksInGroup) / groupSizeM;
    } else {
        blockM = blockIdx.y;
        blockN = blockIdx.x;
    }

    if (blockM * BM >= (uint)M || blockN * BN >= (uint)N) return;

    A += blockM * BM * K;
    B += blockN * BN;
    C += blockM * BM * N + blockN * BN;

    // Accumulators
    wmma::fragment<wmma::accumulator, MMA_M, MMA_N, MMA_K, __half>
        acc[MMA_M_TILES][MMA_N_TILES];
    #pragma unroll
    for (int m = 0; m < MMA_M_TILES; ++m)
        #pragma unroll
        for (int n = 0; n < MMA_N_TILES; ++n)
            wmma::fill_fragment(acc[m][n], __float2half(0.0f));

    // Double-buffered fragments (only 2 buffers needed regardless of K_STEPS)
    wmma::fragment<wmma::matrix_a, MMA_M, MMA_N, MMA_K, __half, wmma::row_major>
        a_frag[2][MMA_M_TILES];
    wmma::fragment<wmma::matrix_b, MMA_M, MMA_N, MMA_K, __half, wmma::row_major>
        b_frag[2][MMA_N_TILES];

    const int numTiles = K / BK;

    // ====== PROLOGUE: fill pipeline ======
    #pragma unroll
    for (int s = 0; s < STAGES - 1 && s < numTiles; ++s) {
        loadTileA_async_padded<BM, BK, A_STRIDE, NUM_THREADS>(
            A + s * BK, As + s * A_STAGE_SIZE, K, tid);
        loadTileB_async_padded<BK, BN, B_STRIDE, NUM_THREADS>(
            B + s * BK * N, Bs + s * B_STAGE_SIZE, N, tid);
        __pipeline_commit();
    }

    __pipeline_wait_prior(0);
    __syncthreads();

    // Load initial k=0 fragments from stage 0
    #pragma unroll
    for (int m = 0; m < MMA_M_TILES; ++m) {
        const __half *ptr = &As[(warpM * WM + m * MMA_M) * A_STRIDE];
        wmma::load_matrix_sync(a_frag[0][m], ptr, A_STRIDE);
    }
    #pragma unroll
    for (int n = 0; n < MMA_N_TILES; ++n) {
        const __half *ptr = &Bs[0 * B_STRIDE + warpN * WN + n * MMA_N];
        wmma::load_matrix_sync(b_frag[0][n], ptr, B_STRIDE);
    }

    int loadTile = STAGES - 1;

    // ====== MAIN LOOP ======
    for (int tile = 0; tile < numTiles; ++tile) {
        const int stage = tile % STAGES;
        const __half* As_tile = As + stage * A_STAGE_SIZE;
        const __half* Bs_tile = Bs + stage * B_STAGE_SIZE;

        // Phase 1: load k+1 from current stage, MMA k (k=0..K_STEPS-2)
        #pragma unroll
        for (int k = 1; k < K_STEPS; ++k) {
            #pragma unroll
            for (int m = 0; m < MMA_M_TILES; ++m) {
                const __half *ptr = &As_tile[(warpM * WM + m * MMA_M) * A_STRIDE + k * MMA_K];
                wmma::load_matrix_sync(a_frag[k % 2][m], ptr, A_STRIDE);
            }
            #pragma unroll
            for (int n = 0; n < MMA_N_TILES; ++n) {
                const __half *ptr = &Bs_tile[k * MMA_K * B_STRIDE + warpN * WN + n * MMA_N];
                wmma::load_matrix_sync(b_frag[k % 2][n], ptr, B_STRIDE);
            }

            #pragma unroll
            for (int m = 0; m < MMA_M_TILES; ++m)
                #pragma unroll
                for (int n = 0; n < MMA_N_TILES; ++n)
                    wmma::mma_sync(acc[m][n], a_frag[(k - 1) % 2][m],
                                   b_frag[(k - 1) % 2][n], acc[m][n]);

            // Issue async copy at midpoint
            constexpr int ASYNC_ISSUE_K = K_STEPS / 2;
            if (k == ASYNC_ISSUE_K && loadTile < numTiles) {
                int loadStage = loadTile % STAGES;
                loadTileA_async_padded<BM, BK, A_STRIDE, NUM_THREADS>(
                    A + loadTile * BK, As + loadStage * A_STAGE_SIZE, K, tid);
                loadTileB_async_padded<BK, BN, B_STRIDE, NUM_THREADS>(
                    B + loadTile * BK * N, Bs + loadStage * B_STAGE_SIZE, N, tid);
                __pipeline_commit();
                ++loadTile;
            }
        }

        // Barrier: ensure next tile's data is ready
        if (loadTile < numTiles) {
            __pipeline_wait_prior(STAGES - 2);
        } else {
            __pipeline_wait_prior(0);
        }
        __syncthreads();

        // Phase 2: load k=0 of next tile, MMA last k-step
        if (tile + 1 < numTiles) {
            int nextStage = (tile + 1) % STAGES;
            const __half* As_next = As + nextStage * A_STAGE_SIZE;
            const __half* Bs_next = Bs + nextStage * B_STAGE_SIZE;
            #pragma unroll
            for (int m = 0; m < MMA_M_TILES; ++m) {
                const __half *ptr = &As_next[(warpM * WM + m * MMA_M) * A_STRIDE];
                wmma::load_matrix_sync(a_frag[0][m], ptr, A_STRIDE);
            }
            #pragma unroll
            for (int n = 0; n < MMA_N_TILES; ++n) {
                const __half *ptr = &Bs_next[0 * B_STRIDE + warpN * WN + n * MMA_N];
                wmma::load_matrix_sync(b_frag[0][n], ptr, B_STRIDE);
            }
        }

        #pragma unroll
        for (int m = 0; m < MMA_M_TILES; ++m)
            #pragma unroll
            for (int n = 0; n < MMA_N_TILES; ++n)
                wmma::mma_sync(acc[m][n], a_frag[(K_STEPS - 1) % 2][m],
                               b_frag[(K_STEPS - 1) % 2][n], acc[m][n]);
    }

    // ====== EPILOGUE ======
    epilogueAndStore_vec4<BM, BN, WM, WN, MMA_M, MMA_N, MMA_K,
                          MMA_M_TILES, MMA_N_TILES, WARPS_M, WARPS_N, C_SMEM_STRIDE>(
        acc, smem, C, N, alpha, beta, tid, warpM, warpN);
}

template<int BM, int BN, int BK, int WM, int WN, int STAGES = 3,
         bool USE_SWIZZLE = false, int BLOCK_STRIDE = 16>
struct WMMAFinal {
    static constexpr int WARPS_M = BM / WM;
    static constexpr int WARPS_N = BN / WN;
    static constexpr int NUM_THREADS = WARPS_M * WARPS_N * 32;

    static constexpr int SMEM_PAD = 8;
    static constexpr int A_STRIDE = BK + SMEM_PAD;
    static constexpr int B_STRIDE = BN + SMEM_PAD;
    static constexpr int C_SMEM_STRIDE = BN + SMEM_PAD;
    static constexpr size_t AB_SMEM_SIZE = STAGES * (BM * A_STRIDE + BK * B_STRIDE) * sizeof(__half);
    static constexpr size_t C_SMEM_SIZE = BM * C_SMEM_STRIDE * sizeof(__half);
    static constexpr size_t SMEM_SIZE = AB_SMEM_SIZE > C_SMEM_SIZE ? AB_SMEM_SIZE : C_SMEM_SIZE;

    static void Run(int M, int N, int K, __half alpha,
                    const __half* A, const __half* B,
                    __half beta, __half* C) {
        static bool configured = false;
        if (!configured) {
            cudaFuncSetAttribute(
                wmma_final<BM, BN, BK, WM, WN, STAGES, USE_SWIZZLE, BLOCK_STRIDE>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                SMEM_SIZE
            );
            configured = true;
        }

        int numBlocksM = (M + BM - 1) / BM;
        int numBlocksN = (N + BN - 1) / BN;

        dim3 block(NUM_THREADS);
        dim3 grid;
        if constexpr (USE_SWIZZLE) {
            grid = dim3(numBlocksM * numBlocksN);
        } else {
            grid = dim3(numBlocksN, numBlocksM);
        }

        wmma_final<BM, BN, BK, WM, WN, STAGES, USE_SWIZZLE, BLOCK_STRIDE>
            <<<grid, block, SMEM_SIZE>>>(M, N, K, alpha, A, B, beta, C);
    }

    static void PrintConfig() {
        printf("WMMAFinal<%d, %d, %d, %d, %d, %d>\n", BM, BN, BK, WM, WN, STAGES);
        printf("  Block tile: %dx%dx%d\n", BM, BN, BK);
        printf("  Warp tile:  %dx%d\n", WM, WN);
        printf("  Stages:     %d\n", STAGES);
        printf("  Threads:    %d (%d warps)\n", NUM_THREADS, WARPS_M * WARPS_N);
        printf("  Shared mem: %.2f KB (dynamic)\n", SMEM_SIZE / 1024.0f);
    }
};

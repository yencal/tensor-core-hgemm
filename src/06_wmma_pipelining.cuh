// 06_wmma_pipelining.cuh
// WMMA HGEMM with interleaved load/compute software pipeline
//   - Fragment double buffering (smem->register overlap within k-loop)
//   - Async copy issued at k-loop midpoint (overlaps DMA with MMA)
//   - Cross-tile fragment prefetch (next tile k=0 overlaps with current tile last MMA)
//
// NOTE: B is in standard layout B[K,N] row-major.
//       BK must be a multiple of 32 (K_STEPS must be even for double-buffering).

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_pipeline.h>
#include <mma.h>
#include "kernel_helpers.cuh"

using namespace nvcuda;

template <int BM, int BN, int BK, int WM, int WN, int STAGES>
__global__ void wmma_pipelining(
    int M, int N, int K, __half alpha,
    const __half *__restrict__ A, const __half *__restrict__ B,
    __half beta, __half *__restrict__ C)
{
    constexpr int MMA_M = 16, MMA_N = 16, MMA_K = 16;
    constexpr int K_STEPS = BK / MMA_K;

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

    static_assert(BM % WM == 0, "BM must be divisible by WM");
    static_assert(BN % WN == 0, "BN must be divisible by WN");
    static_assert(WM % MMA_M == 0, "WM must be divisible by MMA_M (16)");
    static_assert(WN % MMA_N == 0, "WN must be divisible by MMA_N (16)");
    static_assert((BM * BK) % NUM_THREADS == 0, "A tile must be evenly divisible among threads");
    static_assert((BK * BN) % NUM_THREADS == 0, "B tile must be evenly divisible among threads");

    extern __shared__ __half smem[];
    __half* As = smem;
    __half* Bs = smem + STAGES * A_STAGE_SIZE;

    const uint tid = threadIdx.x;
    const uint warpId = tid / 32;
    const uint warpM = warpId / WARPS_N;
    const uint warpN = warpId % WARPS_N;

    A += blockIdx.y * BM * K;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * N + blockIdx.x * BN;

    // Accumulators
    wmma::fragment<wmma::accumulator, MMA_M, MMA_N, MMA_K, __half>
        acc[MMA_M_TILES][MMA_N_TILES];
    #pragma unroll
    for (int m = 0; m < MMA_M_TILES; ++m)
        #pragma unroll
        for (int n = 0; n < MMA_N_TILES; ++n)
            wmma::fill_fragment(acc[m][n], __float2half(0.0f));

    // Double-buffered fragments
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

        // Phase 1: load k+1 fragments, MMA k (k=0..K_STEPS-2)
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

        // Phase 2: load k=0 of next tile, MMA last k-step of current tile
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
    epilogueAndStore<MMA_M, MMA_N, MMA_K, MMA_M_TILES, MMA_N_TILES, WM, WN>(
        acc, C, N, alpha, beta, warpM, warpN);
}

template<int BM, int BN, int BK, int WM, int WN, int STAGES = 3>
struct WMMAPipelining {
    static constexpr int WARPS_M = BM / WM;
    static constexpr int WARPS_N = BN / WN;
    static constexpr int NUM_THREADS = WARPS_M * WARPS_N * 32;

    static constexpr int SMEM_PAD = 8;
    static constexpr int A_STRIDE = BK + SMEM_PAD;
    static constexpr int B_STRIDE = BN + SMEM_PAD;
    static constexpr size_t SMEM_SIZE = STAGES * (BM * A_STRIDE + BK * B_STRIDE) * sizeof(__half);

    static void Run(int M, int N, int K, __half alpha,
                    const __half* A, const __half* B,
                    __half beta, __half* C) {
        static bool configured = false;
        if (!configured) {
            cudaFuncSetAttribute(
                wmma_pipelining<BM, BN, BK, WM, WN, STAGES>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                SMEM_SIZE
            );
            configured = true;
        }
        dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
        dim3 block(NUM_THREADS);
        wmma_pipelining<BM, BN, BK, WM, WN, STAGES><<<grid, block, SMEM_SIZE>>>(
            M, N, K, alpha, A, B, beta, C);
    }

    static void PrintConfig() {
        printf("WMMAPipelining<%d, %d, %d, %d, %d, %d>\n", BM, BN, BK, WM, WN, STAGES);
        printf("  Block tile: %dx%dx%d\n", BM, BN, BK);
        printf("  Warp tile:  %dx%d\n", WM, WN);
        printf("  Stages:     %d\n", STAGES);
        printf("  Threads:    %d\n", NUM_THREADS);
        printf("  Shared mem: %.2f KB\n", SMEM_SIZE / 1024.0f);
    }
};

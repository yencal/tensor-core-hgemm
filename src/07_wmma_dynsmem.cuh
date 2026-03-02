// 07_wmma_dynsmem.cuh
// WMMA HGEMM with all optimizations + dynamic shared memory
// Allows exceeding 48KB static limit (A100 supports up to 164KB)
// Includes: padding, multi-stage pipeline, fragment double buffering

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_pipeline.h>
#include <mma.h>
#include "kernel_helpers.cuh"

using namespace nvcuda;

template <int BM, int BN, int BK, int WM, int WN, int STAGES>
__global__ void wmma_dynsmem(
    int M, int N, int K, __half alpha, 
    const __half *A, const __half *B, __half beta, __half *C)
{
    constexpr int MMA_M = 16;
    constexpr int MMA_N = 16;
    constexpr int MMA_K = 16;

    constexpr int MMA_M_TILES = WM / MMA_M;
    constexpr int MMA_N_TILES = WN / MMA_N;
    constexpr int WARPS_M = BM / WM;
    constexpr int WARPS_N = BN / WN;
    constexpr int NUM_THREADS = WARPS_M * WARPS_N * 32;

    // Padded strides
    constexpr int SMEM_PAD = 8;
    constexpr int A_STRIDE = BK + SMEM_PAD;
    constexpr int B_STRIDE = BN + SMEM_PAD;

    // Sizes per stage
    constexpr int A_STAGE_SIZE = BM * A_STRIDE;
    constexpr int B_STAGE_SIZE = BK * B_STRIDE;

    static_assert(BM % WM == 0, "BM must be divisible by WM");
    static_assert(BN % WN == 0, "BN must be divisible by WN");
    static_assert(BK % MMA_K == 0, "BK must be divisible by MMA_K (16)");
    static_assert(WM % MMA_M == 0, "WM must be divisible by MMA_M (16)");
    static_assert(WN % MMA_N == 0, "WN must be divisible by MMA_N (16)");
    static_assert((BM * BK) % NUM_THREADS == 0, "A tile must be evenly divisible among threads");
    static_assert((BK * BN) % NUM_THREADS == 0, "B tile must be evenly divisible among threads");

    // Dynamic shared memory
    extern __shared__ __half smem[];
    __half* As = smem;                          // As[STAGES][BM * A_STRIDE]
    __half* Bs = smem + STAGES * A_STAGE_SIZE;  // Bs[STAGES][BK * B_STRIDE]

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

    // ====== PROLOGUE: fill the pipeline ======
    #pragma unroll
    for (int s = 0; s < STAGES - 1 && s < numTiles; ++s) {
        __half* As_stage = As + s * A_STAGE_SIZE;
        __half* Bs_stage = Bs + s * B_STAGE_SIZE;
        loadTileA_async_padded<BM, BK, A_STRIDE, NUM_THREADS>(A + s * BK, As_stage, K, tid);
        loadTileB_async_padded<BK, BN, B_STRIDE, NUM_THREADS>(B + s * BK * N, Bs_stage, N, tid);
        __pipeline_commit();
    }

    // ====== MAIN LOOP ======
    int loadTile = STAGES - 1;

    for (int tile = 0; tile < numTiles; ++tile) {
        int computeStage = tile % STAGES;

        // --- Async load: prefetch future tile ---
        if (loadTile < numTiles) {
            int loadStage = loadTile % STAGES;
            __half* As_stage = As + loadStage * A_STAGE_SIZE;
            __half* Bs_stage = Bs + loadStage * B_STAGE_SIZE;
            loadTileA_async_padded<BM, BK, A_STRIDE, NUM_THREADS>(A + loadTile * BK, As_stage, K, tid);
            loadTileB_async_padded<BK, BN, B_STRIDE, NUM_THREADS>(B + loadTile * BK * N, Bs_stage, N, tid);
            __pipeline_commit();
            ++loadTile;
        }

        // Wait for compute stage
        __pipeline_wait_prior(STAGES - 1);
        __syncthreads();

        // Pointers to current tile in shared memory
        const __half* As_tile = As + computeStage * A_STAGE_SIZE;
        const __half* Bs_tile = Bs + computeStage * B_STAGE_SIZE;

        // --- Fragment double buffering within this tile ---
        int frag_load = 0;
        int frag_compute = 1;

        // Prologue: load first fragments (innerK = 0)
        #pragma unroll
        for (int m = 0; m < MMA_M_TILES; ++m) {
            const __half *As_ptr = &As_tile[(warpM * WM + m * MMA_M) * A_STRIDE + 0];
            wmma::load_matrix_sync(a_frag[frag_load][m], As_ptr, A_STRIDE);
        }
        #pragma unroll
        for (int n = 0; n < MMA_N_TILES; ++n) {
            const __half *Bs_ptr = &Bs_tile[0 * B_STRIDE + warpN * WN + n * MMA_N];
            wmma::load_matrix_sync(b_frag[frag_load][n], Bs_ptr, B_STRIDE);
        }

        // Main inner loop with double buffering
        #pragma unroll
        for (int innerK = 0; innerK < BK; innerK += MMA_K) {
            // Swap buffers
            frag_load ^= 1;
            frag_compute ^= 1;

            // Load NEXT fragments (if not last iteration)
            if (innerK + MMA_K < BK) {
                #pragma unroll
                for (int m = 0; m < MMA_M_TILES; ++m) {
                    const __half *As_ptr = &As_tile[(warpM * WM + m * MMA_M) * A_STRIDE + innerK + MMA_K];
                    wmma::load_matrix_sync(a_frag[frag_load][m], As_ptr, A_STRIDE);
                }
                #pragma unroll
                for (int n = 0; n < MMA_N_TILES; ++n) {
                    const __half *Bs_ptr = &Bs_tile[(innerK + MMA_K) * B_STRIDE + warpN * WN + n * MMA_N];
                    wmma::load_matrix_sync(b_frag[frag_load][n], Bs_ptr, B_STRIDE);
                }
            }

            // Compute with CURRENT fragments
            #pragma unroll
            for (int m = 0; m < MMA_M_TILES; ++m) {
                #pragma unroll
                for (int n = 0; n < MMA_N_TILES; ++n) {
                    wmma::mma_sync(acc[m][n], a_frag[frag_compute][m], b_frag[frag_compute][n], acc[m][n]);
                }
            }
        }

        __syncthreads();
    }

    epilogueAndStore<MMA_M, MMA_N, MMA_K, MMA_M_TILES, MMA_N_TILES, WM, WN>(
        acc, C, N, alpha, beta, warpM, warpN);
}

template<int BM, int BN, int BK, int WM, int WN, int STAGES = 2>
struct WMMADynSmem {
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
        
        // Configure max dynamic shared memory
        static bool configured = false;
        if (!configured) {
            cudaFuncSetAttribute(
                wmma_dynsmem<BM, BN, BK, WM, WN, STAGES>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                SMEM_SIZE
            );
            configured = true;
        }

        dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
        dim3 block(NUM_THREADS);
        wmma_dynsmem<BM, BN, BK, WM, WN, STAGES><<<grid, block, SMEM_SIZE>>>(
            M, N, K, alpha, A, B, beta, C);
    }

    static void PrintConfig() {
        printf("WMMADynSmem<%d, %d, %d, %d, %d, %d>\n", BM, BN, BK, WM, WN, STAGES);
        printf("  Block tile: %dx%dx%d\n", BM, BN, BK);
        printf("  Warp tile:  %dx%d\n", WM, WN);
        printf("  Stages:     %d\n", STAGES);
        printf("  Threads:    %d\n", NUM_THREADS);
        printf("  Shared mem: %.2f KB (dynamic)\n", SMEM_SIZE / 1024.0f);
    }
};

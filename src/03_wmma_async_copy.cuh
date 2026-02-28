// 03_wmma_async_copy.cuh
// WMMA HGEMM with async copy (cp.async)

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_pipeline.h>
#include <mma.h>
#include "kernel_helpers.cuh"

using namespace nvcuda;

template <int BM, int BN, int BK, int WM, int WN>
__global__ void wmma_async(
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

    static_assert(BM % WM == 0, "BM must be divisible by WM");
    static_assert(BN % WN == 0, "BN must be divisible by WN");
    static_assert(BK % MMA_K == 0, "BK must be divisible by MMA_K (16)");
    static_assert(WM % MMA_M == 0, "WM must be divisible by MMA_M (16)");
    static_assert(WN % MMA_N == 0, "WN must be divisible by MMA_N (16)");
    static_assert((BM * BK) % NUM_THREADS == 0, "A tile must be evenly divisible among threads");
    static_assert((BK * BN) % NUM_THREADS == 0, "B tile must be evenly divisible among threads");

    __shared__ __half As[BM * BK];
    __shared__ __half Bs[BK * BN];

    const uint tid = threadIdx.x;
    const uint warpId = tid / 32;
    const uint warpM = warpId / WARPS_N;
    const uint warpN = warpId % WARPS_N;

    A += blockIdx.y * BM * K;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * N + blockIdx.x * BN;

    wmma::fragment<wmma::accumulator, MMA_M, MMA_N, MMA_K, __half> 
        acc[MMA_M_TILES][MMA_N_TILES];

    #pragma unroll
    for (int m = 0; m < MMA_M_TILES; ++m)
        #pragma unroll
        for (int n = 0; n < MMA_N_TILES; ++n)
            wmma::fill_fragment(acc[m][n], __float2half(0.0f));

    for (int tileK = 0; tileK < K; tileK += BK) {
        loadTileA_async<BM, BK, NUM_THREADS>(A, As, K, tid);
        loadTileB_async<BK, BN, NUM_THREADS>(B, Bs, N, tid);
        __pipeline_commit();
        __pipeline_wait_prior(0);
        __syncthreads();

        #pragma unroll
        for (int innerK = 0; innerK < BK; innerK += MMA_K) {
            wmma::fragment<wmma::matrix_a, MMA_M, MMA_N, MMA_K, 
                __half, wmma::row_major> a_frag[MMA_M_TILES];

            #pragma unroll
            for (int m = 0; m < MMA_M_TILES; ++m) {
                const __half *As_ptr = &As[(warpM * WM + m * MMA_M) * BK + innerK];
                wmma::load_matrix_sync(a_frag[m], As_ptr, BK);
            }

            wmma::fragment<wmma::matrix_b, MMA_M, MMA_N, MMA_K, 
                __half, wmma::row_major> b_frag[MMA_N_TILES];

            #pragma unroll
            for (int n = 0; n < MMA_N_TILES; ++n) {
                const __half *Bs_ptr = &Bs[innerK * BN + warpN * WN + n * MMA_N];
                wmma::load_matrix_sync(b_frag[n], Bs_ptr, BN);
            }

            #pragma unroll
            for (int m = 0; m < MMA_M_TILES; ++m)
                #pragma unroll
                for (int n = 0; n < MMA_N_TILES; ++n)
                    wmma::mma_sync(acc[m][n], a_frag[m], b_frag[n], acc[m][n]);
        }

        __syncthreads();
        A += BK;
        B += BK * N;
    }

    epilogueAndStore<MMA_M, MMA_N, MMA_K, MMA_M_TILES, MMA_N_TILES, WM, WN>(
        acc, C, N, alpha, beta, warpM, warpN);
}

template<int BM, int BN, int BK, int WM, int WN>
struct WMMAAsync {
    static constexpr int WARPS_M = BM / WM;
    static constexpr int WARPS_N = BN / WN;
    static constexpr int NUM_THREADS = WARPS_M * WARPS_N * 32;

    static void Run(int M, int N, int K, __half alpha,
                    const __half* A, const __half* B,
                    __half beta, __half* C) {
        dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
        dim3 block(NUM_THREADS);
        wmma_async<BM, BN, BK, WM, WN><<<grid, block>>>(
            M, N, K, alpha, A, B, beta, C);
    }

    static void PrintConfig() {
        printf("WMMAAsync<%d, %d, %d, %d, %d>\n", BM, BN, BK, WM, WN);
        printf("  Block tile: %dx%dx%d\n", BM, BN, BK);
        printf("  Warp tile:  %dx%d\n", WM, WN);
        printf("  Threads:    %d\n", NUM_THREADS);
    }
};

using WMMAAsyncFixed = WMMAAsync<128, 128, 32, 64, 64>;
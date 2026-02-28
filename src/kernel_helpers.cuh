// kernel_helpers.cuh
// Reusable functions for WMMA HGEMM kernels (FP16)

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

// =========================================================================
// Tile Loading: Scalar (for 01_wmma_block_tiling)
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

template <int BK, int BN, int NUM_THREADS>
__device__ void loadTileB_scalar(
    const __half *B,
    __half *Bs,
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
// Tile Loading: Vectorized float4 = 8 halves (for 02_wmma_vectorized)
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

template <int BK, int BN, int NUM_THREADS>
__device__ void loadTileB_vec4(
    const __half *B,
    __half *Bs,
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
// Tile Loading: Async (cp.async, 16 bytes = 8 halves per copy)
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

template <int BK, int BN, int NUM_THREADS>
__device__ void loadTileB_async(
    const __half *B,
    __half *Bs,
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

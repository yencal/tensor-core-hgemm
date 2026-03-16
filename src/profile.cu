// profile.cu
// Minimal driver for NCU profiling - one kernel per run
// Includes correctness check against cuBLAS
// Usage: ./profile <kernel_num>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "utils.cuh"
#include "00_cublas.cuh"
#include "01_wmma_block_tiling.cuh"
#include "02_wmma_vectorized.cuh"
#include "03_wmma_async_copy.cuh"
#include "04_wmma_padded.cuh"
#include "05_wmma_multistage.cuh"
#include "06_wmma_pipelining.cuh"
#include "07_wmma_final.cuh"

int main(int argc, char** argv) {
    if (argc != 2) {
        printf("Usage: %s <kernel_num>\n", argv[0]);
        printf("  0  = cuBLAS\n");
        printf("  1  = WMMABlockTiling\n");
        printf("  2  = WMMAVectorized\n");
        printf("  3  = WMMAAsync\n");
        printf("  4  = WMMAPadded\n");
        printf("  5  = WMMAMultistage\n");
        printf("  6  = WMMAPipelining\n");
        printf("  7  = WMMAFinal\n");
        return 1;
    }

    const int N = 8192;
    const int M = N, K = N;
    __half alpha = __float2half(1.0f);
    __half beta  = __float2half(0.0f);
    const char* kernel = argv[1];

    __half *d_A, *d_B, *d_C, *d_C_ref;
    CHECK_CUDA(cudaMalloc(&d_A,     (size_t)M * K * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_B,     (size_t)K * N * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_C,     (size_t)M * N * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_C_ref, (size_t)M * N * sizeof(__half)));

    FillRandomDevice(d_A, (size_t)M * K);
    FillRandomDevice(d_B, (size_t)K * N);

    // Compute reference with cuBLAS
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    HGEMMCuBLAS::Run(handle, M, N, K, alpha, d_A, d_B, beta, d_C_ref);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Fixed configs per kernel (easy to tweak)
    if      (strcmp(kernel, "0") == 0)
        HGEMMCuBLAS::Run(handle, M, N, K, alpha, d_A, d_B, beta, d_C);
    else if (strcmp(kernel, "1") == 0)
        WMMABlockTiling<128,128,32, 32,32>::Run(M,N,K, alpha, d_A, d_B, beta, d_C);
    else if (strcmp(kernel, "2") == 0)
        WMMAVectorized<128,128,32, 32,32>::Run(M,N,K, alpha, d_A, d_B, beta, d_C);
    else if (strcmp(kernel, "3") == 0)
        WMMAAsync<128,128,32, 32,32>::Run(M,N,K, alpha, d_A, d_B, beta, d_C);
    else if (strcmp(kernel, "4") == 0)
        WMMAPadded<128,128,32, 32,32>::Run(M,N,K, alpha, d_A, d_B, beta, d_C);
    else if (strcmp(kernel, "5") == 0)
        WMMAMultistage<128,128,32, 32,32, 2>::Run(M,N,K, alpha, d_A, d_B, beta, d_C);
    else if (strcmp(kernel, "6") == 0)
        WMMAPipelining<256,128,32, 64,64, 3>::Run(M,N,K, alpha, d_A, d_B, beta, d_C);
    else if (strcmp(kernel, "7") == 0)
        WMMAFinal<256,128,32, 64,64, 3, true>::Run(M,N,K, alpha, d_A, d_B, beta, d_C);
    else { printf("Unknown kernel: %s\n", kernel); return 1; }

    CHECK_CUDA(cudaDeviceSynchronize());

    // Verify against cuBLAS
    if (strcmp(kernel, "0") != 0) {
        if (VerifyGEMM(d_C, d_C_ref, M * N)) {
            printf("Kernel %s on %dx%d: PASS\n", kernel, N, N);
        } else {
            printf("Kernel %s on %dx%d: FAIL\n", kernel, N, N);
        }
    } else {
        printf("Kernel %s launched on %dx%d\n", kernel, N, N);
    }

    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaFree(d_C_ref));
    return 0;
}

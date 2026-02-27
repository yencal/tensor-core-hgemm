// 00_cublas.cuh
// cuBLAS wrapper for HGEMM benchmarking (FP16)

#pragma once

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include "utils.cuh"

struct HGEMMCuBLAS {
    static void Run(cublasHandle_t handle, int M, int N, int K, __half alpha,
                    const __half* A, const __half* B,
                    __half beta, __half* C) {
        // Row-major: C = A*B becomes C^T = B^T * A^T in column-major
        cublasGemmEx(handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K,
                    &alpha,
                    B, CUDA_R_16F, N,
                    A, CUDA_R_16F, K,
                    &beta,
                    C, CUDA_R_16F, N,
                    CUBLAS_COMPUTE_16F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
};

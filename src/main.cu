// main.cu
// Benchmark runner for HGEMM implementations (FP16)
//
// NOTE: All kernels expect B in standard layout B[K,N] row-major.

#include <iostream>
#include <vector>

#include "utils.cuh"
#include "00_cublas.cuh"
#include "01_wmma_block_tiling.cuh"
#include "02_wmma_vectorized.cuh"
#include "03_wmma_async_copy.cuh"
#include "04_wmma_padded.cuh"
#include "05_wmma_multistage.cuh"
#include "06_wmma_double_buffer.cuh"
#include "07_wmma_final.cuh"
#include "autotune.cuh"

int main(int argc, char** argv)
{
    std::vector<int> sizes = {1024, 2048, 4096, 8192, 16384};
    std::vector<BenchmarkResult> results;

    __half alpha = __float2half(1.0f);
    __half beta = __float2half(0.0f);

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    // ========================================
    // Benchmark loop - autotune all kernels per size
    // ========================================
    for (int N : sizes) {
        int M = N, K = N;

        std::cout << "\n========================================" << std::endl;
        std::cout << "N = " << N << " (" << (2.0 * M * N * K / 1e9) << " GFLOPs)" << std::endl;
        std::cout << "========================================" << std::endl;

        __half *d_A, *d_B, *d_C, *d_C_ref;
        CHECK_CUDA(cudaMalloc(&d_A, (size_t)M * K * sizeof(__half)));
        CHECK_CUDA(cudaMalloc(&d_B, (size_t)K * N * sizeof(__half)));
        CHECK_CUDA(cudaMalloc(&d_C, (size_t)M * N * sizeof(__half)));
        CHECK_CUDA(cudaMalloc(&d_C_ref, (size_t)M * N * sizeof(__half)));

        FillRandomDevice(d_A, (size_t)M * K);
        FillRandomDevice(d_B, (size_t)K * N);

        // Generate reference
        HGEMMCuBLAS::Run(handle, M, N, K, alpha, d_A, d_B, beta, d_C_ref);
        CHECK_CUDA(cudaDeviceSynchronize());

        // 00: cuBLAS reference
        results.push_back(RunCuBLASBenchmark<HGEMMCuBLAS>(
            "00_cuBLAS", handle, M, N, K, alpha, d_A, d_B, beta, d_C));

        // 01: WMMABlockTiling
        printf("\nAutotuning 01_WMMABlockTiling for N=%d\n", N);
        RunAutotune<WMMABlockTilingTag>(GetWMMAVariants<WMMABlockTiling>(), N);
        CHECK_CUDA(cudaMemset(d_C, 0, (size_t)M * N * sizeof(__half)));
        results.push_back(RunBenchmark<Autotuned<WMMABlockTilingTag>>(
            "01_WMMABlockTiling", M, N, K, alpha, d_A, d_B, beta, d_C, d_C_ref));

        // 02: WMMAVectorized
        printf("\nAutotuning 02_WMMAVectorized for N=%d\n", N);
        RunAutotune<WMMAVectorizedTag>(GetWMMAVectorizedVariants<WMMAVectorized>(), N);
        CHECK_CUDA(cudaMemset(d_C, 0, (size_t)M * N * sizeof(__half)));
        results.push_back(RunBenchmark<Autotuned<WMMAVectorizedTag>>(
            "02_WMMAVectorized", M, N, K, alpha, d_A, d_B, beta, d_C, d_C_ref));

        // 03: WMMAAsync
        printf("\nAutotuning 03_WMMAAsync for N=%d\n", N);
        RunAutotune<WMMAAsyncTag>(GetWMMAVectorizedVariants<WMMAAsync>(), N);
        CHECK_CUDA(cudaMemset(d_C, 0, (size_t)M * N * sizeof(__half)));
        results.push_back(RunBenchmark<Autotuned<WMMAAsyncTag>>(
            "03_WMMAAsync", M, N, K, alpha, d_A, d_B, beta, d_C, d_C_ref));

        // 04: WMMAPadded
        printf("\nAutotuning 04_WMMAPadded for N=%d\n", N);
        RunAutotune<WMMAPaddedTag>(GetWMMAVectorizedVariants<WMMAPadded>(), N);
        CHECK_CUDA(cudaMemset(d_C, 0, (size_t)M * N * sizeof(__half)));
        results.push_back(RunBenchmark<Autotuned<WMMAPaddedTag>>(
            "04_WMMAPadded", M, N, K, alpha, d_A, d_B, beta, d_C, d_C_ref));

        // 05: WMMAMultistage
        printf("\nAutotuning 05_WMMAMultistage for N=%d\n", N);
        RunAutotune<WMMAMultistageTag>(GetWMMAMultistageVariants<WMMAMultistage>(), N);
        CHECK_CUDA(cudaMemset(d_C, 0, (size_t)M * N * sizeof(__half)));
        results.push_back(RunBenchmark<Autotuned<WMMAMultistageTag>>(
            "05_WMMAMultistage", M, N, K, alpha, d_A, d_B, beta, d_C, d_C_ref));

        // 06: WMMADoubleBuffer
        printf("\nAutotuning 06_WMMADoubleBuffer for N=%d\n", N);
        RunAutotune<WMMADoubleBufferTag>(GetWMMAMultistageVariants<WMMADoubleBuffer>(), N);
        CHECK_CUDA(cudaMemset(d_C, 0, (size_t)M * N * sizeof(__half)));
        results.push_back(RunBenchmark<Autotuned<WMMADoubleBufferTag>>(
            "06_WMMADoubleBuffer", M, N, K, alpha, d_A, d_B, beta, d_C, d_C_ref));

        // 07: WMMAFinal
        printf("\nAutotuning 07_WMMAFinal for N=%d\n", N);
        RunAutotune<WMMAFinalTag>(GetWMMAFinalVariants<WMMAFinal>(), N);
        CHECK_CUDA(cudaMemset(d_C, 0, (size_t)M * N * sizeof(__half)));
        results.push_back(RunBenchmark<Autotuned<WMMAFinalTag>>(
            "07_WMMAFinal", M, N, K, alpha, d_A, d_B, beta, d_C, d_C_ref));

        CHECK_CUDA(cudaFree(d_A));
        CHECK_CUDA(cudaFree(d_B));
        CHECK_CUDA(cudaFree(d_C));
        CHECK_CUDA(cudaFree(d_C_ref));
    }
    CHECK_CUBLAS(cublasDestroy(handle));

    WriteCSV(results, "hgemm_results.csv");
    std::cout << "\nResults saved to hgemm_results.csv" << std::endl;

    return 0;
}

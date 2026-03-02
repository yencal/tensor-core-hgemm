// main.cu
// Benchmark runner for HGEMM implementations (FP16)

#include <iostream>
#include <vector>

#include "utils.cuh"
#include "00_cublas.cuh"
#include "01_wmma_block_tiling.cuh"
#include "02_wmma_vectorized.cuh"
#include "03_wmma_async_copy.cuh"
#include "04_wmma_padded.cuh"
#include "05_wmma_multistage.cuh"
#include "autotune.cuh"

int main(int argc, char** argv)
{
    std::vector<int> sizes = {1024, 2048, 4096, 8192, 16384};
    std::vector<BenchmarkResult> results;

    __half alpha = __float2half(1.0f);
    __half beta = __float2half(0.0f);

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    // // Autotune both kernels once before the benchmark loop
    // printf("Autotuning 01_WMMABlockTiling\n");
    // RunAutotune<WMMABlockTilingTag>(GetWMMAVariants<WMMABlockTiling>());

    // printf("Autotuning 02_WMMAVectorized\n");
    // RunAutotune<WMMAVectorizedTag>(GetWMMAVectorizedVariants<WMMAVectorized>());

    // printf("Autotuning 03_WMMAAsync\n");
    // RunAutotune<WMMAAsyncTag>(GetWMMAVectorizedVariants<WMMAAsync>());

    printf("Autotuning 04_WMMAPadded\n");
    RunAutotune<WMMAPaddedTag>(GetWMMAVectorizedVariants<WMMAPadded>());

    printf("Autotuning 05_WMMAMultistage\n");
    RunAutotune<WMMAMultistageTag>(GetWMMAMultistageVariants<WMMAMultistage>());

    for (int N : sizes) {
        int M = N, K = N;

        std::cout << "\n========================================" << std::endl;
        std::cout << "N = " << N << " (" << (2.0 * M * N * K / 1e9) << " GFLOPs)" << std::endl;
        std::cout << "========================================" << std::endl;

        __half *d_A, *d_B, *d_C, *d_C_ref;
        CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(__half)));
        CHECK_CUDA(cudaMalloc(&d_B, K * N * sizeof(__half)));
        CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(__half)));
        CHECK_CUDA(cudaMalloc(&d_C_ref, M * N * sizeof(__half)));

        FillRandomDevice(d_A, M * K);
        FillRandomDevice(d_B, K * N);

        HGEMMCuBLAS::Run(handle, M, N, K, alpha, d_A, d_B, beta, d_C_ref);
        CHECK_CUDA(cudaDeviceSynchronize());

        // cuBLAS reference
        results.push_back(RunCuBLASBenchmark<HGEMMCuBLAS>(
            "00_cuBLAS", handle, M, N, K, alpha, d_A, d_B, beta, d_C));

        // 01: WMMABlockTiling
        CHECK_CUDA(cudaMemset(d_C, 0, M * N * sizeof(__half)));
        results.push_back(RunBenchmark<WMMABlockTiling<128, 128, 16, 32, 32>>(
            "01_WMMABlockTiling", M, N, K, alpha, d_A, d_B, beta, d_C, d_C_ref));

        // 02: WMMAVectorized
        CHECK_CUDA(cudaMemset(d_C, 0, M * N * sizeof(__half)));
        results.push_back(RunBenchmark<WMMAVectorized<128, 128, 32, 64, 64>>(
            "02_WMMAVectorized", M, N, K, alpha, d_A, d_B, beta, d_C, d_C_ref));

        // 03: WMMAAsync
        CHECK_CUDA(cudaMemset(d_C, 0, M * N * sizeof(__half)));
        results.push_back(RunBenchmark<WMMAAsync<128, 128, 32, 64, 64>>(
            "03_WMMAAsync", M, N, K, alpha, d_A, d_B, beta, d_C, d_C_ref));

        // 04: WMMAPadded
        CHECK_CUDA(cudaMemset(d_C, 0, M * N * sizeof(__half)));
        results.push_back(RunBenchmark<WMMAPadded<128, 128, 16, 64, 64>>(
            "04_WMMAPadded", M, N, K, alpha, d_A, d_B, beta, d_C, d_C_ref));

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

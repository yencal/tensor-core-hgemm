// main.cu
// Benchmark runner for HGEMM implementations (FP16)

#include <iostream>
#include <vector>

#include "utils.cuh"
#include "00_cublas.cuh"
#include "01_wmma_block_tiling.cuh"
#include "02_wmma_vectorized.cuh"
#include "autotune.cuh"

int main(int argc, char** argv)
{
    std::vector<int> sizes = {1024, 2048, 4096, 8192};
    std::vector<BenchmarkResult> results;

    __half alpha = __float2half(1.0f);
    __half beta = __float2half(0.0f);

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    // Autotune both kernels once before the benchmark loop
    printf("========================================\n");
    printf("Autotuning 01_WMMABlockTiling\n");
    printf("========================================\n");
    RunAutotune<WMMABlockTilingTag>(GetWMMAVariants<WMMABlockTiling>());

    printf("========================================\n");
    printf("Autotuning 02_WMMAVectorized\n");
    printf("========================================\n");
    RunAutotune<WMMAVectorizedTag>(GetWMMAVectorizedVariants<WMMAVectorized>());

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

        // 01: Autotuned WMMABlockTiling
        CHECK_CUDA(cudaMemset(d_C, 0, M * N * sizeof(__half)));
        results.push_back(RunBenchmark<Autotuned<WMMABlockTilingTag>>(
            "01_WMMABlockTiling_Autotuned", M, N, K, alpha, d_A, d_B, beta, d_C, d_C_ref));

        // 02: Autotuned WMMAVectorized
        CHECK_CUDA(cudaMemset(d_C, 0, M * N * sizeof(__half)));
        results.push_back(RunBenchmark<Autotuned<WMMAVectorizedTag>>(
            "02_WMMAVectorized_Autotuned", M, N, K, alpha, d_A, d_B, beta, d_C, d_C_ref));

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

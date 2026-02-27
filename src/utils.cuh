// utils.cuh
// Error checking, verification, and benchmark utilities for HGEMM (FP16)

#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <curand.h>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << ": " << cudaGetErrorString(err) << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CHECK_CUBLAS(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__ \
                      << ": " << status << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CHECK_CURAND(call) \
    do { \
        curandStatus_t status = call; \
        if (status != CURAND_STATUS_SUCCESS) { \
            std::cerr << "cuRAND error at " << __FILE__ << ":" << __LINE__ \
                      << ": " << status << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    } while(0)

struct BenchmarkResult {
    std::string label;
    int N;
    float time_ms;
    float tflops;
};

__global__ void float_to_half_kernel(const float* src, __half* dst, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = __float2half(src[idx]);
    }
}

inline void FillRandomDevice(__half* d_ptr, size_t n, unsigned long long seed = 42)
{
    float* d_tmp;
    CHECK_CUDA(cudaMalloc(&d_tmp, n * sizeof(float)));

    curandGenerator_t gen;
    CHECK_CURAND(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(gen, seed));
    CHECK_CURAND(curandGenerateUniform(gen, d_tmp, n));
    CHECK_CURAND(curandDestroyGenerator(gen));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    float_to_half_kernel<<<blocks, threads>>>(d_tmp, d_ptr, n);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaFree(d_tmp));
}

inline bool VerifyGEMM(const __half* d_C, const __half* d_C_ref, int size, float rtol = 2e-2f)
{
    std::vector<__half> h_C(size);
    std::vector<__half> h_C_ref(size);

    CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, size * sizeof(__half), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_C_ref.data(), d_C_ref, size * sizeof(__half), cudaMemcpyDeviceToHost));

    for (int i = 0; i < size; ++i) {
        float val = __half2float(h_C[i]);
        float ref = __half2float(h_C_ref[i]);
        float diff = std::fabs(val - ref);
        float max_val = std::fmax(std::fabs(val), std::fabs(ref));
        if (diff > rtol * max_val + 1e-3f) {
            return false;
        }
    }
    return true;
}

inline void WriteCSV(const std::vector<BenchmarkResult>& results, const std::string& filename)
{
    std::ofstream file(filename);
    file << "Label,N,TimeMs,TFLOPS\n";
    for (const auto& r : results) {
        file << "\"" << r.label << "\"," << r.N << "," << r.time_ms << "," << r.tflops << "\n";
    }
    file.close();
}

template<typename Kernel>
BenchmarkResult RunBenchmark(
    const char* label,
    int M, int N, int K,
    __half alpha, const __half* d_A, const __half* d_B,
    __half beta, __half* d_C, const __half* d_C_ref,
    int warmup_runs = 2,
    int timed_runs = 10)
{
    Kernel::Run(M, N, K, alpha, d_A, d_B, beta, d_C);
    CHECK_CUDA(cudaDeviceSynchronize());
    if (!VerifyGEMM(d_C, d_C_ref, M * N)) {
        std::cerr << "FAILED: " << label << std::endl;
        std::exit(EXIT_FAILURE);
    }

    for (int i = 0; i < warmup_runs; ++i) {
        Kernel::Run(M, N, K, alpha, d_A, d_B, beta, d_C);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < timed_runs; ++i) {
        Kernel::Run(M, N, K, alpha, d_A, d_B, beta, d_C);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    float avg_ms = ms / timed_runs;

    double flops = 2.0 * M * N * K;
    float tflops = static_cast<float>((flops / (avg_ms / 1000.0)) / 1e12);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    std::cout << label << ": " << avg_ms << " ms, " << tflops << " TFLOPS [PASS]" << std::endl;

    return BenchmarkResult{label, N, avg_ms, tflops};
}

template<typename CuBLASKernel>
BenchmarkResult RunCuBLASBenchmark(
    const char* label,
    cublasHandle_t handle,
    int M, int N, int K,
    __half alpha, const __half* d_A, const __half* d_B,
    __half beta, __half* d_C,
    int warmup_runs = 2,
    int timed_runs = 10)
{
    for (int i = 0; i < warmup_runs; ++i) {
        CuBLASKernel::Run(handle, M, N, K, alpha, d_A, d_B, beta, d_C);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < timed_runs; ++i) {
        CuBLASKernel::Run(handle, M, N, K, alpha, d_A, d_B, beta, d_C);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    float avg_ms = ms / timed_runs;

    double flops = 2.0 * M * N * K;
    float tflops = static_cast<float>((flops / (avg_ms / 1000.0)) / 1e12);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    std::cout << label << ": " << avg_ms << " ms, " << tflops << " TFLOPS [REF]" << std::endl;

    return BenchmarkResult{label, N, avg_ms, tflops};
}

// autotune.cuh
// Autotuning framework for HGEMM kernels (FP16)

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <functional>
#include <cfloat>
#include <cstdio>

#include "utils.cuh"
#include "01_wmma_block_tiling.cuh"
#include "02_wmma_vectorized.cuh"
#include "03_wmma_async_copy.cuh"
#include "04_wmma_padded.cuh"
#include "05_wmma_multistage.cuh"

struct TuneConfig {
    const char* name;
    std::function<void(int, int, int, __half, const __half*, const __half*, __half, __half*)> run;
};

struct WMMABlockTilingTag {};
struct WMMAVectorizedTag {};
struct WMMAAsyncTag {};
struct WMMAPaddedTag {};
struct WMMAMultistageTag {};

template<typename Tag>
struct Autotuned {
    static inline TuneConfig config;

    static void Run(int M, int N, int K, __half alpha,
                    const __half* A, const __half* B,
                    __half beta, __half* C) {
        config.run(M, N, K, alpha, A, B, beta, C);
    }
};

#define TUNE_CONFIG(Kernel, BM, BN, BK, WM, WN) \
    TuneConfig{#BM "x" #BN "x" #BK "_" #WM "x" #WN, Kernel<BM, BN, BK, WM, WN>::Run}

#define TUNE_CONFIG_MULTISTAGE(Kernel, BM, BN, BK, WM, WN, STAGES) \
    TuneConfig{#BM "x" #BN "x" #BK "_" #WM "x" #WN "_S" #STAGES, \
               Kernel<BM, BN, BK, WM, WN, STAGES>::Run}

template<template<int, int, int, int, int> class Kernel>
inline std::vector<TuneConfig> GetWMMAVariants() {
    return {
        // BK must be multiple of 16 (MMA_K)
        TUNE_CONFIG(Kernel, 64,  64,  16, 16, 16),
        TUNE_CONFIG(Kernel, 64,  64,  32, 16, 16),
        TUNE_CONFIG(Kernel, 64,  128, 16, 16, 32),
        TUNE_CONFIG(Kernel, 128, 64,  16, 32, 16),
        TUNE_CONFIG(Kernel, 128, 128, 16, 32, 32),
        TUNE_CONFIG(Kernel, 128, 128, 32, 32, 32),
        TUNE_CONFIG(Kernel, 128, 128, 64, 32, 32),
        TUNE_CONFIG(Kernel, 64,  128, 32, 32, 32),
    };
}

// For vectorized kernel: need (BM*BK)/8 >= NUM_THREADS and divisible
template<template<int, int, int, int, int> class Kernel>
inline std::vector<TuneConfig> GetWMMAVectorizedVariants() {
    return {
        // These all satisfy: (BM*BK)/8 % NUM_THREADS == 0
        TUNE_CONFIG(Kernel, 128, 128, 32, 32, 32),  // 512 threads, 512 vecs
        TUNE_CONFIG(Kernel, 128, 128, 64, 32, 32),  // 512 threads, 1024 vecs
        TUNE_CONFIG(Kernel, 128, 256, 32, 32, 64),  // 512 threads, 512 vecs
        TUNE_CONFIG(Kernel, 256, 128, 32, 64, 32),  // 512 threads, 1024 vecs
        TUNE_CONFIG(Kernel, 128, 128, 32, 64, 64),  // 128 threads, 512 vecs
        TUNE_CONFIG(Kernel, 64,  128, 64, 32, 32),  // 256 threads, 512 vecs
        TUNE_CONFIG(Kernel, 128, 64,  64, 32, 32),  // 256 threads, 1024 vecs
    };
}

// For multistage
template<template<int, int, int, int, int, int> class Kernel>
inline std::vector<TuneConfig> GetWMMAMultistageVariants() {
    return {
        // sm_80 (A100): 48KB shared mem, STAGES * BK <= 96
        // 2-stage
        TUNE_CONFIG_MULTISTAGE(Kernel, 128, 128, 32, 64, 64, 2), 
        TUNE_CONFIG_MULTISTAGE(Kernel, 128, 128, 32, 32, 32, 2), 
        // 3-stage
        TUNE_CONFIG_MULTISTAGE(Kernel, 128, 64, 32, 64, 32, 3),
        TUNE_CONFIG_MULTISTAGE(Kernel, 64, 128, 32, 32, 64, 3),
        // 4-stage
        TUNE_CONFIG_MULTISTAGE(Kernel, 128, 128, 16, 64, 64, 4),
        TUNE_CONFIG_MULTISTAGE(Kernel, 64, 64, 32, 32, 32, 4),
    };  
}

inline TuneConfig Autotune(
    const std::vector<TuneConfig>& variants,
    int M, int N, int K, __half alpha,
    const __half* A, const __half* B,
    __half beta, __half* C,
    int warmup = 2, int iters = 10)
{
    float best_time = FLT_MAX;
    TuneConfig best = variants[0];

    printf("\n[Autotune] Testing %zu configurations on %dx%dx%d...\n",
           variants.size(), M, N, K);

    for (const auto& config : variants) {
        for (int i = 0; i < warmup; i++) {
            config.run(M, N, K, alpha, A, B, beta, C);
        }
        cudaDeviceSynchronize();

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("  %-28s SKIP (%s)\n", config.name, cudaGetErrorString(err));
            continue;
        }

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        for (int i = 0; i < iters; i++) {
            config.run(M, N, K, alpha, A, B, beta, C);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        ms /= iters;

        double tflops = (2.0 * M * N * K) / (ms * 1e9);
        printf("  %-28s %7.3f ms  %6.2f TFLOPS\n", config.name, ms, tflops);

        if (ms < best_time) {
            best_time = ms;
            best = config;
        }

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    double best_tflops = (2.0 * M * N * K) / (best_time * 1e9);
    printf("[Autotune] Best: %s (%.3f ms, %.2f TFLOPS)\n\n",
           best.name, best_time, best_tflops);

    return best;
}

template<typename Tag>
inline void RunAutotune(
    const std::vector<TuneConfig>& variants,
    int tuneN = 4096,
    __half alpha = __float2half(1.0f),
    __half beta = __float2half(0.0f))
{
    __half *tune_A, *tune_B, *tune_C;
    CHECK_CUDA(cudaMalloc(&tune_A, (size_t)tuneN * tuneN * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&tune_B, (size_t)tuneN * tuneN * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&tune_C, (size_t)tuneN * tuneN * sizeof(__half)));

    FillRandomDevice(tune_A, (size_t)tuneN * tuneN);
    FillRandomDevice(tune_B, (size_t)tuneN * tuneN);

    Autotuned<Tag>::config = Autotune(
        variants, tuneN, tuneN, tuneN, alpha, tune_A, tune_B, beta, tune_C);

    CHECK_CUDA(cudaFree(tune_A));
    CHECK_CUDA(cudaFree(tune_B));
    CHECK_CUDA(cudaFree(tune_C));
}

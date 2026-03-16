# Tensor Core HGEMM Optimization

A step-by-step exploration of FP16 GEMM optimization using NVIDIA Tensor Cores and the WMMA API.

## Requirements

- CUDA Toolkit (tested with 12.x)
- GPU with SM 80+ (Ampere or later) for async copy and tensor core features
- Python 3 + matplotlib (for plotting)

## Build & Run

Make sure to set `ARCH` to match your GPU architecture (e.g., sm_90, sm_80, etc.).

```bash
make ARCH=sm_80
./hgemm_bench
python3 scripts/plot_results.py hgemm_results.csv
```

## Kernel Progression

| Kernel | Description |
|--------|-------------|
| cuBLAS | cuBLAS reference (FP16 tensor cores) |
| 01_WMMABlockTiling | Block tiling baseline with WMMA fragments |
| 02_WMMAVectorized | + Vectorized global memory loads (float4) |
| 03_WMMAAsync | + Asynchronous GMEM→SMEM copies (cp.async) |
| 04_WMMAPadded | + Shared memory padding for bank conflict reduction |
| 05_WMMAMultistage | + Multi-stage async pipeline (overlapped tile loads) |
| 06_WMMAPipelining | + Fragment double buffering, interleaved load/compute |
| 07_WMMAFinal | + Block swizzling, vectorized epilogue, autotuning |

## Results

**NVIDIA A100-SXM4 (40 GB)**

The final kernel achieves **80% of cuBLAS** performance (219 vs 274 TFLOPS at N=4096).

![HGEMM Performance](figures/wmma_hgemm_plot_a100.png)

| Kernel | N=4096 Time | TFLOPS | % cuBLAS |
|--------|-------------|--------|----------|
| cuBLAS (FP16) | 0.50 ms | 274 | 100% |
| 01_WMMABlockTiling | 3.27 ms | 42 | 15% |
| 02_WMMAVectorized | 1.85 ms | 74 | 27% |
| 03_WMMAAsync | 1.68 ms | 82 | 30% |
| 04_WMMAPadded | 0.88 ms | 156 | 57% |
| 05_WMMAMultistage | 0.71 ms | 193 | 71% |
| 06_WMMAPipelining | 0.68 ms | 203 | 74% |
| 07_WMMAFinal | 0.63 ms | 219 | **80%** |

## Blog Post

For a detailed walkthrough of the optimization techniques, see the accompanying blog post:  
[Tensor Core HGEMM: A Progressive Optimization Guide Using WMMA](https://yencal.github.io/gpu-hgemm-wmma/)
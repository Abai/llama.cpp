# CUDA: fuse KV cache dequantization into fattn-mma-f16

## Overview

  - Allow fattn-mma-f16 to consume quantized KV data, similar to fattn-vec
    * Enabled only for KV q8_0/q8_0, q4_0/q4_0 and q8_0/q4_0, head 128 and 256
    * Eliminates FA f16 convert buffer completely
    * Adds templated loaders in fattn-mma-load.cuh. These stage tiles with
      cp.async
    * Tiles dequantized in-kernel into the swizzled (#25635), f16, smem.
      These are verified bit-identical to master.
    * At large ubatch, cp.async nstages==2, K dequant causes regression,
      here, IMMA KQ is used quantizing Q instead, see below

  - Allow fattn-mma-f16 kernel to run KQ using IMMA on quantized KV
    * Enabled only for cp.async nstages==2 on hw that supports cp.async
    * Speeds up KQ targetting int8 tensor cores. Q is quantized to q8 blocks,
      in-kernel, K deinterleaved to int8.
    * IMMA KQ test-backend-ops passes and wikitext-2 perplexity is similar to
      master (table below).

## Additional information

  - Issue #29371, #29826
  - Related PRs:
    - #23907 (merged): pre-allocation of FA f16 convert buffer
    - #21054 (closed, not merged): checks free VRAM and falls back to fattn-vec
      when the f16 convert buffer does not fit
    - #27140 (open): adds q4_1/q5_0/q5_1 conversion kernels, no overlap with
      this PR
    - #25635: XOR smem swizzle
    - #7527: compile-time discussion
    - #29827: the other PR for same issue, caps the convert
      buffer at 64 MiB instead of in-kernel dequant. Works for fattn-mma-f16 and
      fattn-tile.

### f16 KV path unchanged

- f16 KV path verified unchanged. On sm_86 all 21 built
  `fattn-mma-f16-instance-*.cu.o` are SASS identical against master `36b101543`.
  Log: [bench-out/sassgate-2026-09-25/](https://github.com/Abai/llama.cpp/tree/benchmark/cuda-fattn-mma-fused-dequant/bench-out/sassgate-2026-09-25).
- f16 KV throughput verified unchanged on sm_86 and sm_120, just to be sure:

<details>
<summary>Click to expand - f16/f16 KV, `74c907e3d` vs master `36b101543`, 3 reps</summary>

| GPU | model | test | ub | master t/s | PR t/s | ratio |
|---|---|---|---:|---:|---:|---:|
| RTX 3090 | llama 8B Q4_0 | pp1024@d16384 | 4 | 442.57 | 442.33 | 1.00 |
| RTX 3090 | llama 8B Q4_0 | pp1024@d16384 | 64 | 2731.79 | 2733.42 | 1.00 |
| RTX 3090 | llama 8B Q4_0 | pp1024@d16384 | 512 | 3465.15 | 3478.39 | 1.00 |
| RTX 5070 Laptop | llama 8B Q4_0 | pp1024@d16384 | 4 | 224.24 | 223.70 | 1.00 |
| RTX 5070 Laptop | llama 8B Q4_0 | pp1024@d16384 | 64 | 1352.11 | 1351.10 | 1.00 |
| RTX 5070 Laptop | llama 8B Q4_0 | pp1024@d16384 | 512 | 1582.42 | 1581.90 | 1.00 |
| RTX 3090 | deepseek2 16B Q4_0 | pp1024@d32768 | 4 | 464.04 | 463.61 | 1.00 |
| RTX 3090 | deepseek2 16B Q4_0 | pp1024@d32768 | 64 | 1199.57 | 1202.68 | 1.00 |
| RTX 3090 | deepseek2 16B Q4_0 | pp1024@d32768 | 512 | 1396.74 | 1395.47 | 1.00 |
| RTX 3090 | gemma 2B Q4_0 | pp1024@d32768 | 4 | 1154.57 | 1154.00 | 1.00 |
| RTX 3090 | gemma 2B Q4_0 | pp1024@d32768 | 64 | 6167.14 | 6159.06 | 1.00 |
| RTX 3090 | gemma 2B Q4_0 | pp1024@d32768 | 512 | 8455.65 | 8451.36 | 1.00 |
| RTX 3090 | qwen35 27B Q4_0 | pp1024@d32768 | 4 | 145.23 | 145.28 | 1.00 |
| RTX 3090 | qwen35 27B Q4_0 | pp1024@d32768 | 64 | 887.32 | 887.22 | 1.00 |
| RTX 3090 | qwen35 27B Q4_0 | pp1024@d32768 | 512 | 1136.57 | 1137.75 | 1.00 |
| RTX 5070 Laptop | gemma 2B Q4_0 | pp1024@d32768 | 4 | 732.26 | 732.37 | 1.00 |
| RTX 5070 Laptop | gemma 2B Q4_0 | pp1024@d32768 | 64 | 4496.28 | 4496.18 | 1.00 |
| RTX 5070 Laptop | gemma 2B Q4_0 | pp1024@d32768 | 512 | 5420.10 | 5412.38 | 1.00 |
| RTX 3090 | deepseek2 16B Q4_0 | tg128@d32768 | - | 190.99 | 191.47 | 1.00 |
| RTX 3090 | gemma 2B Q4_0 | tg128@d32768 | - | 291.32 | 291.59 | 1.00 |
| RTX 3090 | llama 8B Q4_0 | tg128@d16384 | - | 114.92 | 114.92 | 1.00 |
| RTX 3090 | qwen35 27B Q4_0 | tg128@d32768 | - | 42.17 | 42.16 | 1.00 |
| RTX 5070 Laptop | gemma 2B Q4_0 | tg128@d32768 | - | 153.45 | 153.47 | 1.00 |
| RTX 5070 Laptop | llama 8B Q4_0 | tg128@d16384 | - | 53.01 | 53.29 | 1.01 |

</details>

### Performance

- Tested on RTX 3090 24GB and RTX 5070 8GB (laptop)
- Largest prefill gains on small ubatch and large ctx
- On sm_120 gemma q8_0/q8_0, head 256, causes 384 B of spill, worst case -23%
  regression vs master.  Retuned q8_0/q8_0 to occupancy 1 with nthreads 256
  (`cc >= GGML_CUDA_CC_BLACKWELL` on the host, `__CUDA_ARCH__ >=
  GGML_CUDA_CC_BLACKWELL` on the device), recovered to -5% worst case
  regression.

Speedup = PR t/s / master t/s. llama-bench, pp1024@d32768, 3 reps, PR
`74c907e3d` vs master `36b101543`:

| GPU | model | q8_0/q8_0 ub8 | ub64 | ub512 | q4_0/q4_0 ub8 | ub64 | ub512 |
|---|---|---:|---:|---:|---:|---:|---:|
| RTX 3090 | llama 8B Q4_0 (head 128) | 1.71 | 1.24 | 1.03 | 1.74 | 1.26 | 1.08 |
| RTX 3090 | qwen35 27B Q4_0 | 1.10 | 1.03 | 0.98 | 1.13 | 1.05 | 1.02 |
| RTX 3090 | gemma 2B Q4_0 | 1.23 | 1.07 | 1.00 | 1.25 | 1.10 | 1.07 |
| RTX 3090 | deepseek2 16B Q4_0 (MLA, not fused) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| RTX 5070 Laptop | llama 8B Q4_0 (head 128) | 2.01 | 1.70 | 1.52 | 2.12 | 1.91 | 1.73 |
| RTX 5070 Laptop | gemma 2B Q4_0 | 1.29 | 1.06 | 0.97 | 1.29 | 1.10 | 1.01 |

<details>
<summary>Click to expand - Full ub 1-512, pp512 and tg128 ladders</summary>


### KV q8_0/q8_0, pp1024@d32768

| GPU | Model | Microbatch size | Test | t/s master | t/s PR | Speedup |
|:----|:------|--------:|:-----|-----------:|-----------:|--------:|
| RTX 3090 | deepseek2 16B Q4_0 | 1 | pp1024@d32768 | 145.85 | 145.66 | 1.00 |
| RTX 3090 | deepseek2 16B Q4_0 | 2 | pp1024@d32768 | 249.97 | 249.93 | 1.00 |
| RTX 3090 | deepseek2 16B Q4_0 | 4 | pp1024@d32768 | 378.22 | 378.40 | 1.00 |
| RTX 3090 | deepseek2 16B Q4_0 | 8 | pp1024@d32768 | 517.74 | 517.68 | 1.00 |
| RTX 3090 | deepseek2 16B Q4_0 | 16 | pp1024@d32768 | 754.44 | 754.28 | 1.00 |
| RTX 3090 | deepseek2 16B Q4_0 | 32 | pp1024@d32768 | 983.76 | 984.00 | 1.00 |
| RTX 3090 | deepseek2 16B Q4_0 | 64 | pp1024@d32768 | 1155.36 | 1155.84 | 1.00 |
| RTX 3090 | deepseek2 16B Q4_0 | 128 | pp1024@d32768 | 1252.69 | 1252.45 | 1.00 |
| RTX 3090 | deepseek2 16B Q4_0 | 256 | pp1024@d32768 | 1293.65 | 1293.50 | 1.00 |
| RTX 3090 | deepseek2 16B Q4_0 | 512 | pp1024@d32768 | 1392.90 | 1392.85 | 1.00 |
| RTX 3090 | gemma 2B Q4_0 | 1 | pp1024@d32768 | 276.96 | 275.94 | 1.00 |
| RTX 3090 | gemma 2B Q4_0 | 2 | pp1024@d32768 | 443.83 | 624.98 | 1.41 |
| RTX 3090 | gemma 2B Q4_0 | 4 | pp1024@d32768 | 850.98 | 1162.49 | 1.37 |
| RTX 3090 | gemma 2B Q4_0 | 8 | pp1024@d32768 | 1319.04 | 1628.77 | 1.23 |
| RTX 3090 | gemma 2B Q4_0 | 16 | pp1024@d32768 | 2373.51 | 2775.29 | 1.17 |
| RTX 3090 | gemma 2B Q4_0 | 32 | pp1024@d32768 | 3676.71 | 4131.74 | 1.12 |
| RTX 3090 | gemma 2B Q4_0 | 64 | pp1024@d32768 | 5330.53 | 5694.79 | 1.07 |
| RTX 3090 | gemma 2B Q4_0 | 128 | pp1024@d32768 | 6484.14 | 6670.23 | 1.03 |
| RTX 3090 | gemma 2B Q4_0 | 256 | pp1024@d32768 | 7757.81 | 7728.21 | 1.00 |
| RTX 3090 | gemma 2B Q4_0 | 512 | pp1024@d32768 | 8263.66 | 8303.50 | 1.00 |
| RTX 3090 | llama 8B Q4_0 | 1 | pp1024@d32768 | 78.30 | 78.27 | 1.00 |
| RTX 3090 | llama 8B Q4_0 | 2 | pp1024@d32768 | 104.14 | 209.08 | 2.01 |
| RTX 3090 | llama 8B Q4_0 | 4 | pp1024@d32768 | 201.20 | 401.54 | 2.00 |
| RTX 3090 | llama 8B Q4_0 | 8 | pp1024@d32768 | 347.83 | 595.84 | 1.71 |
| RTX 3090 | llama 8B Q4_0 | 16 | pp1024@d32768 | 659.26 | 1091.59 | 1.66 |
| RTX 3090 | llama 8B Q4_0 | 32 | pp1024@d32768 | 1126.04 | 1556.46 | 1.38 |
| RTX 3090 | llama 8B Q4_0 | 64 | pp1024@d32768 | 1654.80 | 2047.76 | 1.24 |
| RTX 3090 | llama 8B Q4_0 | 128 | pp1024@d32768 | 1652.97 | 2255.92 | 1.36 |
| RTX 3090 | llama 8B Q4_0 | 256 | pp1024@d32768 | 2180.31 | 2349.58 | 1.08 |
| RTX 3090 | llama 8B Q4_0 | 512 | pp1024@d32768 | 2325.89 | 2406.93 | 1.03 |
| RTX 3090 | qwen35 27B Q4_0 | 1 | pp1024@d32768 | 41.02 | 41.02 | 1.00 |
| RTX 3090 | qwen35 27B Q4_0 | 2 | pp1024@d32768 | 71.30 | 86.18 | 1.21 |
| RTX 3090 | qwen35 27B Q4_0 | 4 | pp1024@d32768 | 126.20 | 148.48 | 1.18 |
| RTX 3090 | qwen35 27B Q4_0 | 8 | pp1024@d32768 | 188.07 | 207.22 | 1.10 |
| RTX 3090 | qwen35 27B Q4_0 | 16 | pp1024@d32768 | 377.97 | 412.46 | 1.09 |
| RTX 3090 | qwen35 27B Q4_0 | 32 | pp1024@d32768 | 577.56 | 612.85 | 1.06 |
| RTX 3090 | qwen35 27B Q4_0 | 64 | pp1024@d32768 | 838.06 | 865.21 | 1.03 |
| RTX 3090 | qwen35 27B Q4_0 | 128 | pp1024@d32768 | 910.43 | 1023.35 | 1.12 |
| RTX 3090 | qwen35 27B Q4_0 | 256 | pp1024@d32768 | 1080.40 | 1072.18 | 0.99 |
| RTX 3090 | qwen35 27B Q4_0 | 512 | pp1024@d32768 | 1122.82 | 1101.34 | 0.98 |
| RTX 5070 Laptop | gemma 2B Q4_0 | 1 | pp1024@d32768 | 155.09 | 155.07 | 1.00 |
| RTX 5070 Laptop | gemma 2B Q4_0 | 2 | pp1024@d32768 | 216.88 | 216.91 | 1.00 |
| RTX 5070 Laptop | gemma 2B Q4_0 | 4 | pp1024@d32768 | 576.32 | 758.19 | 1.32 |
| RTX 5070 Laptop | gemma 2B Q4_0 | 8 | pp1024@d32768 | 972.71 | 1251.56 | 1.29 |
| RTX 5070 Laptop | gemma 2B Q4_0 | 16 | pp1024@d32768 | 1797.54 | 2247.75 | 1.25 |
| RTX 5070 Laptop | gemma 2B Q4_0 | 32 | pp1024@d32768 | 2790.97 | 3215.58 | 1.15 |
| RTX 5070 Laptop | gemma 2B Q4_0 | 64 | pp1024@d32768 | 3845.72 | 4068.58 | 1.06 |
| RTX 5070 Laptop | gemma 2B Q4_0 | 128 | pp1024@d32768 | 4475.68 | 4622.51 | 1.03 |
| RTX 5070 Laptop | gemma 2B Q4_0 | 256 | pp1024@d32768 | 5038.20 | 4994.24 | 0.99 |
| RTX 5070 Laptop | gemma 2B Q4_0 | 512 | pp1024@d32768 | 5251.04 | 5114.65 | 0.97 |
| RTX 5070 Laptop | llama 8B Q4_0 | 1 | pp1024@d32768 | 41.02 | 41.01 | 1.00 |
| RTX 5070 Laptop | llama 8B Q4_0 | 2 | pp1024@d32768 | 66.41 | 66.39 | 1.00 |
| RTX 5070 Laptop | llama 8B Q4_0 | 4 | pp1024@d32768 | 94.81 | 212.31 | 2.24 |
| RTX 5070 Laptop | llama 8B Q4_0 | 8 | pp1024@d32768 | 179.07 | 360.22 | 2.01 |
| RTX 5070 Laptop | llama 8B Q4_0 | 16 | pp1024@d32768 | 345.73 | 693.04 | 2.00 |
| RTX 5070 Laptop | llama 8B Q4_0 | 32 | pp1024@d32768 | 531.24 | 958.33 | 1.80 |
| RTX 5070 Laptop | llama 8B Q4_0 | 64 | pp1024@d32768 | 709.31 | 1204.30 | 1.70 |
| RTX 5070 Laptop | llama 8B Q4_0 | 128 | pp1024@d32768 | 831.36 | 1353.03 | 1.63 |
| RTX 5070 Laptop | llama 8B Q4_0 | 256 | pp1024@d32768 | 915.14 | 1438.82 | 1.57 |
| RTX 5070 Laptop | llama 8B Q4_0 | 512 | pp1024@d32768 | 941.06 | 1430.50 | 1.52 |

### KV q8_0/q8_0, pp1024@d65536

| GPU | Model | Microbatch size | Test | t/s master | t/s PR | Speedup |
|:----|:------|--------:|:-----|-----------:|-----------:|--------:|
| RTX 3090 | deepseek2 16B Q4_0 | 1 | pp1024@d65536 | 101.04 | 101.03 | 1.00 |
| RTX 3090 | deepseek2 16B Q4_0 | 2 | pp1024@d65536 | 178.93 | 178.90 | 1.00 |
| RTX 3090 | deepseek2 16B Q4_0 | 4 | pp1024@d65536 | 279.33 | 279.27 | 1.00 |
| RTX 3090 | deepseek2 16B Q4_0 | 8 | pp1024@d65536 | 380.98 | 381.13 | 1.00 |
| RTX 3090 | deepseek2 16B Q4_0 | 16 | pp1024@d65536 | 526.73 | 526.34 | 1.00 |
| RTX 3090 | deepseek2 16B Q4_0 | 32 | pp1024@d65536 | 653.45 | 653.24 | 1.00 |
| RTX 3090 | deepseek2 16B Q4_0 | 64 | pp1024@d65536 | 740.55 | 739.80 | 1.00 |
| RTX 3090 | deepseek2 16B Q4_0 | 128 | pp1024@d65536 | 783.50 | 783.67 | 1.00 |
| RTX 3090 | deepseek2 16B Q4_0 | 256 | pp1024@d65536 | 739.45 | 740.41 | 1.00 |
| RTX 3090 | deepseek2 16B Q4_0 | 512 | pp1024@d65536 | 772.76 | 772.80 | 1.00 |

### KV q8_0/q8_0, pp1024@d131072

| GPU | Model | Microbatch size | Test | t/s master | t/s PR | Speedup |
|:----|:------|--------:|:-----|-----------:|-----------:|--------:|
| RTX 3090 | gemma 2B Q4_0 | 1 | pp1024@d131072 | 119.91 | 119.87 | 1.00 |
| RTX 3090 | gemma 2B Q4_0 | 2 | pp1024@d131072 | 201.18 | 461.34 | 2.29 |
| RTX 3090 | gemma 2B Q4_0 | 4 | pp1024@d131072 | 395.23 | 837.39 | 2.12 |
| RTX 3090 | gemma 2B Q4_0 | 8 | pp1024@d131072 | 691.62 | 1248.26 | 1.80 |
| RTX 3090 | gemma 2B Q4_0 | 16 | pp1024@d131072 | 1199.83 | 1869.38 | 1.56 |
| RTX 3090 | gemma 2B Q4_0 | 32 | pp1024@d131072 | 1805.22 | 2369.62 | 1.31 |
| RTX 3090 | gemma 2B Q4_0 | 64 | pp1024@d131072 | 2370.89 | 2700.89 | 1.14 |
| RTX 3090 | gemma 2B Q4_0 | 128 | pp1024@d131072 | 2758.94 | 2894.31 | 1.05 |
| RTX 3090 | gemma 2B Q4_0 | 256 | pp1024@d131072 | 3078.70 | 3056.27 | 0.99 |
| RTX 3090 | gemma 2B Q4_0 | 512 | pp1024@d131072 | 3125.53 | 3180.60 | 1.02 |
| RTX 5070 Laptop | gemma 2B Q4_0 | 1 | pp1024@d131072 | 65.62 | 65.65 | 1.00 |
| RTX 5070 Laptop | gemma 2B Q4_0 | 2 | pp1024@d131072 | 79.24 | 79.18 | 1.00 |
| RTX 5070 Laptop | gemma 2B Q4_0 | 4 | pp1024@d131072 | 193.03 | 469.29 | 2.43 |
| RTX 5070 Laptop | gemma 2B Q4_0 | 8 | pp1024@d131072 | 357.85 | 819.10 | 2.29 |
| RTX 5070 Laptop | gemma 2B Q4_0 | 16 | pp1024@d131072 | 692.49 | 1240.02 | 1.79 |
| RTX 5070 Laptop | gemma 2B Q4_0 | 32 | pp1024@d131072 | 1065.58 | 1521.87 | 1.43 |
| RTX 5070 Laptop | gemma 2B Q4_0 | 64 | pp1024@d131072 | 1445.57 | 1689.91 | 1.17 |
| RTX 5070 Laptop | gemma 2B Q4_0 | 128 | pp1024@d131072 | 1699.85 | 1780.72 | 1.05 |
| RTX 5070 Laptop | gemma 2B Q4_0 | 256 | pp1024@d131072 | 1878.26 | 1824.38 | 0.97 |
| RTX 5070 Laptop | gemma 2B Q4_0 | 512 | pp1024@d131072 | 1956.75 | 1854.49 | 0.95 |

### KV q8_0/q8_0, pp512 depth ladder

| GPU | Model | Depth | Test | t/s master | t/s PR | Speedup |
|:----|:------|--------:|:-----|-----------:|-----------:|--------:|
| RTX 3090 | deepseek2 16B Q4_0 | 0 | pp512@d0 | 6830.44 | 6811.99 | 1.00 |
| RTX 3090 | deepseek2 16B Q4_0 | 16384 | pp512@d16384 | 2304.28 | 2305.58 | 1.00 |
| RTX 3090 | deepseek2 16B Q4_0 | 32768 | pp512@d32768 | 1392.22 | 1391.70 | 1.00 |
| RTX 3090 | deepseek2 16B Q4_0 | 65536 | pp512@d65536 | 771.91 | 771.76 | 1.00 |
| RTX 3090 | gemma 2B Q4_0 | 0 | pp512@d0 | 16768.77 | 16668.40 | 0.99 |
| RTX 3090 | gemma 2B Q4_0 | 16384 | pp512@d16384 | 10826.86 | 10899.03 | 1.01 |
| RTX 3090 | gemma 2B Q4_0 | 32768 | pp512@d32768 | 8003.10 | 8078.01 | 1.01 |
| RTX 3090 | gemma 2B Q4_0 | 65536 | pp512@d65536 | 5210.75 | 5302.60 | 1.02 |
| RTX 3090 | llama 8B Q4_0 | 0 | pp512@d0 | 5783.23 | 5750.96 | 0.99 |
| RTX 3090 | llama 8B Q4_0 | 16384 | pp512@d16384 | 3311.71 | 3386.73 | 1.02 |
| RTX 3090 | llama 8B Q4_0 | 32768 | pp512@d32768 | 2318.36 | 2399.56 | 1.04 |
| RTX 3090 | llama 8B Q4_0 | 65536 | pp512@d65536 | 1445.70 | 1502.77 | 1.04 |
| RTX 3090 | qwen35 27B Q4_0 | 0 | pp512@d0 | 1534.59 | 1531.97 | 1.00 |
| RTX 3090 | qwen35 27B Q4_0 | 16384 | pp512@d16384 | 1296.42 | 1282.96 | 0.99 |
| RTX 3090 | qwen35 27B Q4_0 | 32768 | pp512@d32768 | 1120.08 | 1098.95 | 0.98 |
| RTX 3090 | qwen35 27B Q4_0 | 65536 | pp512@d65536 | 880.13 | 852.85 | 0.97 |
| RTX 5070 Laptop | gemma 2B Q4_0 | 0 | pp512@d0 | 11082.49 | 10987.23 | 0.99 |
| RTX 5070 Laptop | gemma 2B Q4_0 | 16384 | pp512@d16384 | 7007.77 | 6836.22 | 0.98 |
| RTX 5070 Laptop | gemma 2B Q4_0 | 32768 | pp512@d32768 | 5140.46 | 4992.87 | 0.97 |
| RTX 5070 Laptop | gemma 2B Q4_0 | 65536 | pp512@d65536 | 3322.10 | 3178.02 | 0.96 |
| RTX 5070 Laptop | llama 8B Q4_0 | 0 | pp512@d0 | 3696.08 | 3680.74 | 1.00 |
| RTX 5070 Laptop | llama 8B Q4_0 | 16384 | pp512@d16384 | 1557.60 | 2183.42 | 1.40 |
| RTX 5070 Laptop | llama 8B Q4_0 | 32768 | pp512@d32768 | 955.73 | 1436.62 | 1.50 |

### KV q8_0/q8_0, tg128 depth ladder

| GPU | Model | Depth | Test | t/s master | t/s PR | Speedup |
|:----|:------|--------:|:-----|-----------:|-----------:|--------:|
| RTX 3090 | deepseek2 16B Q4_0 | 0 | tg128@d0 | 266.24 | 266.24 | 1.00 |
| RTX 3090 | deepseek2 16B Q4_0 | 16384 | tg128@d16384 | 179.16 | 179.17 | 1.00 |
| RTX 3090 | deepseek2 16B Q4_0 | 32768 | tg128@d32768 | 140.61 | 140.49 | 1.00 |
| RTX 3090 | deepseek2 16B Q4_0 | 65536 | tg128@d65536 | 98.46 | 98.53 | 1.00 |
| RTX 3090 | gemma 2B Q4_0 | 0 | tg128@d0 | 351.28 | 352.05 | 1.00 |
| RTX 3090 | gemma 2B Q4_0 | 16384 | tg128@d16384 | 277.57 | 275.90 | 0.99 |
| RTX 3090 | gemma 2B Q4_0 | 32768 | tg128@d32768 | 228.37 | 230.96 | 1.01 |
| RTX 3090 | gemma 2B Q4_0 | 65536 | tg128@d65536 | 169.58 | 169.23 | 1.00 |
| RTX 3090 | llama 8B Q4_0 | 0 | tg128@d0 | 154.71 | 154.67 | 1.00 |
| RTX 3090 | llama 8B Q4_0 | 16384 | tg128@d16384 | 101.64 | 101.67 | 1.00 |
| RTX 3090 | llama 8B Q4_0 | 32768 | tg128@d32768 | 75.07 | 75.02 | 1.00 |
| RTX 3090 | llama 8B Q4_0 | 65536 | tg128@d65536 | 49.07 | 49.20 | 1.00 |
| RTX 3090 | qwen35 27B Q4_0 | 0 | tg128@d0 | 46.57 | 46.59 | 1.00 |
| RTX 3090 | qwen35 27B Q4_0 | 16384 | tg128@d16384 | 42.52 | 42.51 | 1.00 |
| RTX 3090 | qwen35 27B Q4_0 | 32768 | tg128@d32768 | 39.00 | 38.98 | 1.00 |
| RTX 3090 | qwen35 27B Q4_0 | 65536 | tg128@d65536 | 33.17 | 33.21 | 1.00 |
| RTX 5070 Laptop | gemma 2B Q4_0 | 0 | tg128@d0 | 199.25 | 199.21 | 1.00 |
| RTX 5070 Laptop | gemma 2B Q4_0 | 16384 | tg128@d16384 | 156.07 | 156.07 | 1.00 |
| RTX 5070 Laptop | gemma 2B Q4_0 | 32768 | tg128@d32768 | 126.99 | 126.99 | 1.00 |
| RTX 5070 Laptop | gemma 2B Q4_0 | 65536 | tg128@d65536 | 92.12 | 92.06 | 1.00 |
| RTX 5070 Laptop | llama 8B Q4_0 | 0 | tg128@d0 | 76.99 | 76.98 | 1.00 |
| RTX 5070 Laptop | llama 8B Q4_0 | 16384 | tg128@d16384 | 51.43 | 51.44 | 1.00 |
| RTX 5070 Laptop | llama 8B Q4_0 | 32768 | tg128@d32768 | 38.78 | 38.82 | 1.00 |

### KV q4_0/q4_0, pp1024@d32768

| GPU | Model | Microbatch size | Test | t/s master | t/s PR | Speedup |
|:----|:------|--------:|:-----|-----------:|-----------:|--------:|
| RTX 3090 | deepseek2 16B Q4_0 | 1 | pp1024@d32768 | 149.07 | 149.07 | 1.00 |
| RTX 3090 | deepseek2 16B Q4_0 | 2 | pp1024@d32768 | 256.04 | 256.00 | 1.00 |
| RTX 3090 | deepseek2 16B Q4_0 | 4 | pp1024@d32768 | 387.15 | 387.13 | 1.00 |
| RTX 3090 | deepseek2 16B Q4_0 | 8 | pp1024@d32768 | 518.32 | 518.19 | 1.00 |
| RTX 3090 | deepseek2 16B Q4_0 | 16 | pp1024@d32768 | 763.95 | 764.03 | 1.00 |
| RTX 3090 | deepseek2 16B Q4_0 | 32 | pp1024@d32768 | 992.93 | 993.08 | 1.00 |
| RTX 3090 | deepseek2 16B Q4_0 | 64 | pp1024@d32768 | 1165.29 | 1165.40 | 1.00 |
| RTX 3090 | deepseek2 16B Q4_0 | 128 | pp1024@d32768 | 1258.26 | 1257.88 | 1.00 |
| RTX 3090 | deepseek2 16B Q4_0 | 256 | pp1024@d32768 | 1297.00 | 1297.19 | 1.00 |
| RTX 3090 | deepseek2 16B Q4_0 | 512 | pp1024@d32768 | 1394.96 | 1394.55 | 1.00 |
| RTX 3090 | gemma 2B Q4_0 | 1 | pp1024@d32768 | 249.62 | 249.52 | 1.00 |
| RTX 3090 | gemma 2B Q4_0 | 2 | pp1024@d32768 | 456.22 | 675.09 | 1.48 |
| RTX 3090 | gemma 2B Q4_0 | 4 | pp1024@d32768 | 863.49 | 1214.95 | 1.41 |
| RTX 3090 | gemma 2B Q4_0 | 8 | pp1024@d32768 | 1359.40 | 1695.17 | 1.25 |
| RTX 3090 | gemma 2B Q4_0 | 16 | pp1024@d32768 | 2437.45 | 2872.54 | 1.18 |
| RTX 3090 | gemma 2B Q4_0 | 32 | pp1024@d32768 | 3757.00 | 4320.69 | 1.15 |
| RTX 3090 | gemma 2B Q4_0 | 64 | pp1024@d32768 | 5419.52 | 5980.38 | 1.10 |
| RTX 3090 | gemma 2B Q4_0 | 128 | pp1024@d32768 | 6573.89 | 6984.75 | 1.06 |
| RTX 3090 | gemma 2B Q4_0 | 256 | pp1024@d32768 | 7768.88 | 8068.19 | 1.04 |
| RTX 3090 | gemma 2B Q4_0 | 512 | pp1024@d32768 | 8206.18 | 8760.84 | 1.07 |
| RTX 3090 | llama 8B Q4_0 | 1 | pp1024@d32768 | 75.12 | 75.15 | 1.00 |
| RTX 3090 | llama 8B Q4_0 | 2 | pp1024@d32768 | 110.52 | 232.72 | 2.11 |
| RTX 3090 | llama 8B Q4_0 | 4 | pp1024@d32768 | 211.69 | 440.23 | 2.08 |
| RTX 3090 | llama 8B Q4_0 | 8 | pp1024@d32768 | 365.73 | 636.41 | 1.74 |
| RTX 3090 | llama 8B Q4_0 | 16 | pp1024@d32768 | 695.60 | 1144.24 | 1.64 |
| RTX 3090 | llama 8B Q4_0 | 32 | pp1024@d32768 | 1162.54 | 1627.11 | 1.40 |
| RTX 3090 | llama 8B Q4_0 | 64 | pp1024@d32768 | 1706.42 | 2148.43 | 1.26 |
| RTX 3090 | llama 8B Q4_0 | 128 | pp1024@d32768 | 1678.93 | 2421.57 | 1.44 |
| RTX 3090 | llama 8B Q4_0 | 256 | pp1024@d32768 | 2202.04 | 2446.54 | 1.11 |
| RTX 3090 | llama 8B Q4_0 | 512 | pp1024@d32768 | 2335.47 | 2511.24 | 1.08 |
| RTX 3090 | qwen35 27B Q4_0 | 1 | pp1024@d32768 | 39.44 | 39.43 | 1.00 |
| RTX 3090 | qwen35 27B Q4_0 | 2 | pp1024@d32768 | 71.69 | 87.84 | 1.23 |
| RTX 3090 | qwen35 27B Q4_0 | 4 | pp1024@d32768 | 125.19 | 148.75 | 1.19 |
| RTX 3090 | qwen35 27B Q4_0 | 8 | pp1024@d32768 | 186.17 | 210.10 | 1.13 |
| RTX 3090 | qwen35 27B Q4_0 | 16 | pp1024@d32768 | 381.26 | 418.25 | 1.10 |
| RTX 3090 | qwen35 27B Q4_0 | 32 | pp1024@d32768 | 581.31 | 622.09 | 1.07 |
| RTX 3090 | qwen35 27B Q4_0 | 64 | pp1024@d32768 | 838.79 | 881.95 | 1.05 |
| RTX 3090 | qwen35 27B Q4_0 | 128 | pp1024@d32768 | 911.27 | 1057.52 | 1.16 |
| RTX 3090 | qwen35 27B Q4_0 | 256 | pp1024@d32768 | 1080.81 | 1112.72 | 1.03 |
| RTX 3090 | qwen35 27B Q4_0 | 512 | pp1024@d32768 | 1123.60 | 1140.63 | 1.02 |
| RTX 5070 Laptop | gemma 2B Q4_0 | 1 | pp1024@d32768 | 56.42 | 56.41 | 1.00 |
| RTX 5070 Laptop | gemma 2B Q4_0 | 2 | pp1024@d32768 | 97.76 | 97.76 | 1.00 |
| RTX 5070 Laptop | gemma 2B Q4_0 | 4 | pp1024@d32768 | 596.89 | 844.95 | 1.42 |
| RTX 5070 Laptop | gemma 2B Q4_0 | 8 | pp1024@d32768 | 985.54 | 1270.92 | 1.29 |
| RTX 5070 Laptop | gemma 2B Q4_0 | 16 | pp1024@d32768 | 1812.11 | 2283.43 | 1.26 |
| RTX 5070 Laptop | gemma 2B Q4_0 | 32 | pp1024@d32768 | 2794.11 | 3339.07 | 1.20 |
| RTX 5070 Laptop | gemma 2B Q4_0 | 64 | pp1024@d32768 | 3925.47 | 4334.56 | 1.10 |
| RTX 5070 Laptop | gemma 2B Q4_0 | 128 | pp1024@d32768 | 4545.92 | 4840.59 | 1.06 |
| RTX 5070 Laptop | gemma 2B Q4_0 | 256 | pp1024@d32768 | 5098.15 | 5225.81 | 1.03 |
| RTX 5070 Laptop | gemma 2B Q4_0 | 512 | pp1024@d32768 | 5287.82 | 5339.18 | 1.01 |
| RTX 5070 Laptop | llama 8B Q4_0 | 1 | pp1024@d32768 | 18.60 | 18.60 | 1.00 |
| RTX 5070 Laptop | llama 8B Q4_0 | 2 | pp1024@d32768 | 33.95 | 33.93 | 1.00 |
| RTX 5070 Laptop | llama 8B Q4_0 | 4 | pp1024@d32768 | 100.87 | 249.30 | 2.47 |
| RTX 5070 Laptop | llama 8B Q4_0 | 8 | pp1024@d32768 | 189.13 | 400.61 | 2.12 |
| RTX 5070 Laptop | llama 8B Q4_0 | 16 | pp1024@d32768 | 367.92 | 763.69 | 2.08 |
| RTX 5070 Laptop | llama 8B Q4_0 | 32 | pp1024@d32768 | 551.89 | 1089.40 | 1.97 |
| RTX 5070 Laptop | llama 8B Q4_0 | 64 | pp1024@d32768 | 732.62 | 1397.02 | 1.91 |
| RTX 5070 Laptop | llama 8B Q4_0 | 128 | pp1024@d32768 | 855.49 | 1587.40 | 1.86 |
| RTX 5070 Laptop | llama 8B Q4_0 | 256 | pp1024@d32768 | 927.74 | 1657.57 | 1.79 |
| RTX 5070 Laptop | llama 8B Q4_0 | 512 | pp1024@d32768 | 955.86 | 1654.48 | 1.73 |

### KV q8_0/q4_0, pp1024@d32768

| GPU | Model | Microbatch size | Test | t/s master | t/s PR | Speedup |
|:----|:------|--------:|:-----|-----------:|-----------:|--------:|
| RTX 3090 | gemma 2B Q4_0 | 1 | pp1024@d32768 | 252.30 | 252.33 | 1.00 |
| RTX 3090 | gemma 2B Q4_0 | 2 | pp1024@d32768 | 448.96 | 652.93 | 1.45 |
| RTX 3090 | gemma 2B Q4_0 | 4 | pp1024@d32768 | 866.36 | 1163.06 | 1.34 |
| RTX 3090 | gemma 2B Q4_0 | 8 | pp1024@d32768 | 1330.07 | 1650.23 | 1.24 |
| RTX 3090 | gemma 2B Q4_0 | 16 | pp1024@d32768 | 2407.27 | 2819.60 | 1.17 |
| RTX 3090 | gemma 2B Q4_0 | 32 | pp1024@d32768 | 3710.23 | 4235.48 | 1.14 |
| RTX 3090 | gemma 2B Q4_0 | 64 | pp1024@d32768 | 5351.30 | 5857.87 | 1.09 |
| RTX 3090 | gemma 2B Q4_0 | 128 | pp1024@d32768 | 6534.76 | 6855.43 | 1.05 |
| RTX 3090 | gemma 2B Q4_0 | 256 | pp1024@d32768 | 7751.64 | 7902.31 | 1.02 |
| RTX 3090 | gemma 2B Q4_0 | 512 | pp1024@d32768 | 8202.07 | 8750.05 | 1.07 |
| RTX 3090 | llama 8B Q4_0 | 1 | pp1024@d32768 | 74.38 | 74.38 | 1.00 |
| RTX 3090 | llama 8B Q4_0 | 2 | pp1024@d32768 | 107.40 | 219.74 | 2.05 |
| RTX 3090 | llama 8B Q4_0 | 4 | pp1024@d32768 | 206.00 | 417.45 | 2.03 |
| RTX 3090 | llama 8B Q4_0 | 8 | pp1024@d32768 | 356.26 | 613.29 | 1.72 |
| RTX 3090 | llama 8B Q4_0 | 16 | pp1024@d32768 | 676.78 | 1112.06 | 1.64 |
| RTX 3090 | llama 8B Q4_0 | 32 | pp1024@d32768 | 1142.77 | 1594.06 | 1.39 |
| RTX 3090 | llama 8B Q4_0 | 64 | pp1024@d32768 | 1676.51 | 2113.71 | 1.26 |
| RTX 3090 | llama 8B Q4_0 | 128 | pp1024@d32768 | 1664.76 | 2372.33 | 1.43 |
| RTX 3090 | llama 8B Q4_0 | 256 | pp1024@d32768 | 2188.24 | 2523.87 | 1.15 |
| RTX 3090 | llama 8B Q4_0 | 512 | pp1024@d32768 | 2330.73 | 2594.47 | 1.11 |
| RTX 3090 | qwen35 27B Q4_0 | 1 | pp1024@d32768 | 39.68 | 39.66 | 1.00 |
| RTX 3090 | qwen35 27B Q4_0 | 2 | pp1024@d32768 | 71.59 | 87.04 | 1.22 |
| RTX 3090 | qwen35 27B Q4_0 | 4 | pp1024@d32768 | 125.98 | 149.46 | 1.19 |
| RTX 3090 | qwen35 27B Q4_0 | 8 | pp1024@d32768 | 187.18 | 208.18 | 1.11 |
| RTX 3090 | qwen35 27B Q4_0 | 16 | pp1024@d32768 | 379.88 | 415.74 | 1.09 |
| RTX 3090 | qwen35 27B Q4_0 | 32 | pp1024@d32768 | 579.52 | 617.77 | 1.07 |
| RTX 3090 | qwen35 27B Q4_0 | 64 | pp1024@d32768 | 837.18 | 875.83 | 1.05 |
| RTX 3090 | qwen35 27B Q4_0 | 128 | pp1024@d32768 | 910.54 | 1050.10 | 1.15 |
| RTX 3090 | qwen35 27B Q4_0 | 256 | pp1024@d32768 | 1079.49 | 1102.50 | 1.02 |
| RTX 3090 | qwen35 27B Q4_0 | 512 | pp1024@d32768 | 1122.48 | 1131.87 | 1.01 |
| RTX 5070 Laptop | gemma 2B Q4_0 | 1 | pp1024@d32768 | 58.87 | 58.87 | 1.00 |
| RTX 5070 Laptop | gemma 2B Q4_0 | 2 | pp1024@d32768 | 97.91 | 97.92 | 1.00 |
| RTX 5070 Laptop | gemma 2B Q4_0 | 4 | pp1024@d32768 | 598.47 | 798.47 | 1.33 |
| RTX 5070 Laptop | gemma 2B Q4_0 | 8 | pp1024@d32768 | 1007.41 | 1259.13 | 1.25 |
| RTX 5070 Laptop | gemma 2B Q4_0 | 16 | pp1024@d32768 | 1838.00 | 2246.19 | 1.22 |
| RTX 5070 Laptop | gemma 2B Q4_0 | 32 | pp1024@d32768 | 2864.68 | 3274.84 | 1.14 |
| RTX 5070 Laptop | gemma 2B Q4_0 | 64 | pp1024@d32768 | 3901.21 | 4216.34 | 1.08 |
| RTX 5070 Laptop | gemma 2B Q4_0 | 128 | pp1024@d32768 | 4521.47 | 4739.91 | 1.05 |
| RTX 5070 Laptop | gemma 2B Q4_0 | 256 | pp1024@d32768 | 5055.89 | 5120.83 | 1.01 |
| RTX 5070 Laptop | gemma 2B Q4_0 | 512 | pp1024@d32768 | 5273.92 | 5208.29 | 0.99 |
| RTX 5070 Laptop | llama 8B Q4_0 | 1 | pp1024@d32768 | 18.56 | 18.56 | 1.00 |
| RTX 5070 Laptop | llama 8B Q4_0 | 2 | pp1024@d32768 | 34.10 | 34.10 | 1.00 |
| RTX 5070 Laptop | llama 8B Q4_0 | 4 | pp1024@d32768 | 98.56 | 230.55 | 2.34 |
| RTX 5070 Laptop | llama 8B Q4_0 | 8 | pp1024@d32768 | 185.29 | 380.70 | 2.05 |
| RTX 5070 Laptop | llama 8B Q4_0 | 16 | pp1024@d32768 | 358.77 | 732.94 | 2.04 |
| RTX 5070 Laptop | llama 8B Q4_0 | 32 | pp1024@d32768 | 544.43 | 1030.87 | 1.89 |
| RTX 5070 Laptop | llama 8B Q4_0 | 64 | pp1024@d32768 | 722.17 | 1308.84 | 1.81 |
| RTX 5070 Laptop | llama 8B Q4_0 | 128 | pp1024@d32768 | 847.02 | 1480.15 | 1.75 |
| RTX 5070 Laptop | llama 8B Q4_0 | 256 | pp1024@d32768 | 921.87 | 1575.61 | 1.71 |
| RTX 5070 Laptop | llama 8B Q4_0 | 512 | pp1024@d32768 | 952.26 | 1589.85 | 1.67 |

### KV f16/f16, pp1024@d16384

| GPU | Model | Microbatch size | Test | t/s master | t/s PR | Speedup |
|:----|:------|--------:|:-----|-----------:|-----------:|--------:|
| RTX 3090 | llama 8B Q4_0 | 4 | pp1024@d16384 | 442.57 | 442.33 | 1.00 |
| RTX 3090 | llama 8B Q4_0 | 64 | pp1024@d16384 | 2731.79 | 2733.42 | 1.00 |
| RTX 3090 | llama 8B Q4_0 | 512 | pp1024@d16384 | 3465.15 | 3478.39 | 1.00 |
| RTX 5070 Laptop | llama 8B Q4_0 | 4 | pp1024@d16384 | 224.24 | 223.70 | 1.00 |
| RTX 5070 Laptop | llama 8B Q4_0 | 64 | pp1024@d16384 | 1352.11 | 1351.10 | 1.00 |
| RTX 5070 Laptop | llama 8B Q4_0 | 512 | pp1024@d16384 | 1582.42 | 1581.90 | 1.00 |

### KV f16/f16, pp1024@d32768

| GPU | Model | Microbatch size | Test | t/s master | t/s PR | Speedup |
|:----|:------|--------:|:-----|-----------:|-----------:|--------:|
| RTX 3090 | deepseek2 16B Q4_0 | 4 | pp1024@d32768 | 464.04 | 463.61 | 1.00 |
| RTX 3090 | deepseek2 16B Q4_0 | 64 | pp1024@d32768 | 1199.57 | 1202.68 | 1.00 |
| RTX 3090 | deepseek2 16B Q4_0 | 512 | pp1024@d32768 | 1396.74 | 1395.47 | 1.00 |
| RTX 3090 | gemma 2B Q4_0 | 4 | pp1024@d32768 | 1154.57 | 1154.00 | 1.00 |
| RTX 3090 | gemma 2B Q4_0 | 64 | pp1024@d32768 | 6167.14 | 6159.06 | 1.00 |
| RTX 3090 | gemma 2B Q4_0 | 512 | pp1024@d32768 | 8455.65 | 8451.36 | 1.00 |
| RTX 3090 | qwen35 27B Q4_0 | 4 | pp1024@d32768 | 145.23 | 145.28 | 1.00 |
| RTX 3090 | qwen35 27B Q4_0 | 64 | pp1024@d32768 | 887.32 | 887.22 | 1.00 |
| RTX 3090 | qwen35 27B Q4_0 | 512 | pp1024@d32768 | 1136.57 | 1137.75 | 1.00 |
| RTX 5070 Laptop | gemma 2B Q4_0 | 4 | pp1024@d32768 | 732.26 | 732.37 | 1.00 |
| RTX 5070 Laptop | gemma 2B Q4_0 | 64 | pp1024@d32768 | 4496.28 | 4496.18 | 1.00 |
| RTX 5070 Laptop | gemma 2B Q4_0 | 512 | pp1024@d32768 | 5420.10 | 5412.38 | 1.00 |

### KV f16/f16, tg128 depth ladder

| GPU | Model | Depth | Test | t/s master | t/s PR | Speedup |
|:----|:------|--------:|:-----|-----------:|-----------:|--------:|
| RTX 3090 | deepseek2 16B Q4_0 | 32768 | tg128@d32768 | 190.99 | 191.47 | 1.00 |
| RTX 3090 | gemma 2B Q4_0 | 32768 | tg128@d32768 | 291.32 | 291.59 | 1.00 |
| RTX 3090 | llama 8B Q4_0 | 16384 | tg128@d16384 | 114.92 | 114.92 | 1.00 |
| RTX 3090 | qwen35 27B Q4_0 | 32768 | tg128@d32768 | 42.17 | 42.16 | 1.00 |
| RTX 5070 Laptop | gemma 2B Q4_0 | 32768 | tg128@d32768 | 153.45 | 153.47 | 1.00 |
| RTX 5070 Laptop | llama 8B Q4_0 | 16384 | tg128@d16384 | 53.01 | 53.29 | 1.01 |

</details>

### VRAM

- Gains 470 MiB at 131k ctx for qwen35 27B

`CUDA0 compute buffer`, llama-server `-ngl 99 -fa on -ub 512`, KV q8_0/q8_0,
RTX 3090, qwen35 27B (weight-independent):

| n_ctx | master | PR | saved |
|---:|---:|---:|---:|
| 16384 | 160.28 MiB | 138.28 MiB | 22.00 MiB |
| 32768 | 240.28 MiB | 154.28 MiB | 86.00 MiB |
| 65536 | 400.28 MiB | 186.28 MiB | 214.00 MiB |
| 131072 | 720.28 MiB | 250.28 MiB | 470.00 MiB |

Largest context PR vs master, same settings, RTX 3090 24 GB:

| weights | side | starts at | compute buffer | fails at |
|---|---|---:|---:|---:|
| qwen35 27B Q4_0 (14.9 GiB) | master | 229376 | 1200.28 MiB | 237568 |
| qwen35 27B Q4_0 (14.9 GiB) | PR | 253952 | 370.28 MiB | 262144 |

<details>
<summary>Click to expand - CUDA0 compute buffer (MiB), four models, context ladder, all KV pairs (RTX 3090, llama-server -ub 512)</summary>


### VRAM: CUDA0 compute buffer (MiB), KV q8_0/q8_0, -ub 512

| GPU | Model | n_ctx | master | PR | saved MiB | KV buffer MiB |
|:----|:------|------:|-------:|--------------------:|----------:|--------------:|
| RTX 3090 | deepseek | 16384 | 104.27 | 104.27 | 0.00 | 258.20 |
| RTX 3090 | deepseek | 32768 | 138.27 | 138.27 | 0.00 | 516.39 |
| RTX 3090 | deepseek | 65536 | 206.27 | 206.27 | 0.00 | 1032.76 |
| RTX 3090 | deepseek | 131072 | 342.27 | 342.27 | 0.00 | 2065.51 |
| RTX 3090 | gemma | 16384 | 124.28 | 120.28 | 4.00 | 153.01 |
| RTX 3090 | gemma | 32768 | 140.28 | 136.28 | 4.00 | 306.01 |
| RTX 3090 | gemma | 65536 | 172.28 | 168.28 | 4.00 | 612.01 |
| RTX 3090 | gemma | 131072 | 272.28 | 232.28 | 40.00 | 1224.01 |
| RTX 3090 | llama | 16384 | 124.09 | 116.09 | 8.00 | 1088.00 |
| RTX 3090 | llama | 32768 | 192.09 | 132.09 | 60.00 | 2176.00 |
| RTX 3090 | llama | 65536 | 352.09 | 164.09 | 188.00 | 4352.00 |
| RTX 3090 | llama | 131072 | 672.09 | 228.09 | 444.00 | 8704.00 |
| RTX 3090 | qwen | 16384 | 160.28 | 138.28 | 22.00 | 544.00 |
| RTX 3090 | qwen | 32768 | 240.28 | 154.28 | 86.00 | 1088.00 |
| RTX 3090 | qwen | 65536 | 400.28 | 186.28 | 214.00 | 2176.00 |
| RTX 3090 | qwen | 131072 | 720.28 | 250.28 | 470.00 | 4352.00 |

### VRAM: CUDA0 compute buffer (MiB), KV q4_0/q4_0, -ub 512

| GPU | Model | n_ctx | master | PR | saved MiB | KV buffer MiB |
|:----|:------|------:|-------:|--------------------:|----------:|--------------:|
| RTX 3090 | deepseek | 32768 | 138.27 | 138.27 | 0.00 | 273.38 |
| RTX 3090 | deepseek | 131072 | 342.27 | 342.27 | 0.00 | 1093.51 |
| RTX 3090 | gemma | 32768 | 140.28 | 136.28 | 4.00 | 162.01 |
| RTX 3090 | gemma | 131072 | 272.28 | 232.28 | 40.00 | 648.01 |
| RTX 3090 | llama | 32768 | 192.09 | 132.09 | 60.00 | 1152.00 |
| RTX 3090 | llama | 131072 | 672.09 | 228.09 | 444.00 | 4608.00 |
| RTX 3090 | qwen | 32768 | 240.28 | 154.28 | 86.00 | 576.00 |
| RTX 3090 | qwen | 131072 | 720.28 | 250.28 | 470.00 | 2304.00 |

### VRAM: CUDA0 compute buffer (MiB), KV q8_0/q4_0, -ub 512

| GPU | Model | n_ctx | master | PR | saved MiB | KV buffer MiB |
|:----|:------|------:|-------:|--------------------:|----------:|--------------:|
| RTX 3090 | gemma | 32768 | 140.28 | 136.28 | 4.00 | 234.01 |
| RTX 3090 | gemma | 131072 | 272.28 | 232.28 | 40.00 | 936.01 |
| RTX 3090 | llama | 32768 | 192.09 | 132.09 | 60.00 | 1664.00 |
| RTX 3090 | llama | 131072 | 672.09 | 228.09 | 444.00 | 6656.00 |
| RTX 3090 | qwen | 32768 | 240.28 | 154.28 | 86.00 | 832.00 |
| RTX 3090 | qwen | 131072 | 720.28 | 250.28 | 470.00 | 3328.00 |

### VRAM: CUDA0 compute buffer (MiB), KV f16/f16, -ub 512

| GPU | Model | n_ctx | master | PR | saved MiB | KV buffer MiB |
|:----|:------|------:|-------:|--------------------:|----------:|--------------:|
| RTX 3090 | deepseek | 16384 | 92.13 | 92.13 | 0.00 | 486.00 |
| RTX 3090 | deepseek | 32768 | 108.13 | 108.13 | 0.00 | 972.00 |
| RTX 3090 | deepseek | 65536 | 140.13 | 140.13 | 0.00 | 1944.00 |
| RTX 3090 | deepseek | 131072 | 204.13 | 204.13 | 0.00 | 3888.00 |
| RTX 3090 | gemma | 16384 | 124.01 | 124.01 | 0.00 | 288.00 |
| RTX 3090 | gemma | 32768 | 140.01 | 140.01 | 0.00 | 576.00 |
| RTX 3090 | gemma | 65536 | 172.01 | 172.01 | 0.00 | 1152.00 |
| RTX 3090 | gemma | 131072 | 236.01 | 236.01 | 0.00 | 2304.00 |
| RTX 3090 | llama | 16384 | 124.01 | 124.01 | 0.00 | 2048.00 |
| RTX 3090 | llama | 32768 | 140.01 | 140.01 | 0.00 | 4096.00 |
| RTX 3090 | llama | 65536 | 172.01 | 172.01 | 0.00 | 8192.00 |
| RTX 3090 | llama | 131072 | 236.01 | 236.01 | 0.00 | 16384.00 |
| RTX 3090 | qwen | 16384 | 138.02 | 138.02 | 0.00 | 1024.00 |
| RTX 3090 | qwen | 32768 | 154.02 | 154.02 | 0.00 | 2048.00 |
| RTX 3090 | qwen | 65536 | 186.02 | 186.02 | 0.00 | 4096.00 |
| RTX 3090 | qwen | 131072 | 250.02 | 250.02 | 0.00 | 8192.00 |

</details>

### Correctness

- f16/f16 is bit-identical to master; deepseek2 (MLA) is never fused and
  is bit-identical also; the fused pairs differ slightly because
  of Q quantization.
- `test-backend-ops test -o FLASH_ATTN_EXT -b CUDA0` (GGML_CUDA_FA_ALL_QUANTS
  =ON) at `74c907e3d` vs master `36b101543`, RTX 3090:
  3957/3957 vs 3955/3955; RTX 5070 Laptop: 3957/3957 vs
  3955/3955. This PR adds 2 cases.
- Full `test-backend-ops test -b CUDA0` (same build): RTX 3090
  15632/15632 vs 15630/15630, RTX 5070 Laptop 15632/15632 vs
  15630/15630, 2/2 backends everywhere.
- compute-sanitizer memcheck over the entire FLASH_ATTN_EXT:
  RTX 3090 - 0 errors, RTX 5070 Laptop - 0 errors.
- Artifacts generated with .devops/cuda.Dockerfile derived image:
  [VALIDATE-RTX3090-sm86-24GB.txt](https://github.com/Abai/llama.cpp/blob/benchmark/cuda-fattn-mma-fused-dequant/bench-out/jg-multi/cuda-fattn-mma-fused-dequant/VALIDATE-RTX3090-sm86-24GB.txt),
  [VALIDATE-RTX5070LaptopGPU-sm120-8GB.txt](https://github.com/Abai/llama.cpp/blob/benchmark/cuda-fattn-mma-fused-dequant/bench-out/jg-multi/cuda-fattn-mma-fused-dequant/VALIDATE-RTX5070LaptopGPU-sm120-8GB.txt),
  [PPL-RTX3090-sm86-24GB.md](https://github.com/Abai/llama.cpp/blob/benchmark/cuda-fattn-mma-fused-dequant/bench-out/jg-multi/cuda-fattn-mma-fused-dequant/PPL-RTX3090-sm86-24GB.md)

<details>
<summary>Click to expand - wikitext-2 perplexity at n_ctx 8192/16384/32768 and KL divergence vs the master f16-KV run</summary>

### Perplexity: wikitext-2 test, n_ctx=8192, 9 chunks (73728 tokens)

| GPU | Model | KV | PPL master | PPL PR | PR - master |
|:----|:------|:---|-----------:|-------:|------------:|
| RTX 3090 | gemma 2B Q4_0 | f16/f16 | 437.5897 +/- 10.30099 | 437.5897 +/- 10.30099 | +0.0000 |
| RTX 3090 | gemma 2B Q4_0 | q4_0/q4_0 | 439.2269 +/- 10.33557 | 434.5401 +/- 10.17492 | -4.6868 |
| RTX 3090 | gemma 2B Q4_0 | q8_0/q8_0 | 437.2399 +/- 10.27944 | 436.6813 +/- 10.26770 | -0.5586 |
| RTX 5070 Laptop | gemma 2B Q4_0 | f16/f16 | 438.2826 +/- 10.32349 | 438.2826 +/- 10.32349 | +0.0000 |
| RTX 5070 Laptop | gemma 2B Q4_0 | q4_0/q4_0 | 441.9798 +/- 10.39866 | 433.6014 +/- 10.15060 | -8.3784 |
| RTX 5070 Laptop | gemma 2B Q4_0 | q8_0/q8_0 | 438.1430 +/- 10.30890 | 438.6653 +/- 10.33142 | +0.5223 |

### Perplexity: wikitext-2 test, n_ctx=16384, 9 chunks (147456 tokens)

| GPU | Model | KV | PPL master | PPL PR | PR - master |
|:----|:------|:---|-----------:|-------:|------------:|
| RTX 5070 Laptop | llama 8B Q4_0 | f16/f16 | 5.7306 +/- 0.04682 | 5.7306 +/- 0.04682 | +0.0000 |
| RTX 5070 Laptop | llama 8B Q4_0 | q4_0/q4_0 | 5.8087 +/- 0.04733 | 5.8021 +/- 0.04723 | -0.0066 |
| RTX 5070 Laptop | llama 8B Q4_0 | q8_0/q8_0 | 5.7303 +/- 0.04684 | 5.7307 +/- 0.04684 | +0.0004 |

### Perplexity: wikitext-2 test, n_ctx=32768, 9 chunks (294912 tokens)

| GPU | Model | KV | PPL master | PPL PR | PR - master |
|:----|:------|:---|-----------:|-------:|------------:|
| RTX 3090 | deepseek2 16B Q4_0 | f16/f16 | 6.1287 +/- 0.03662 | 6.1287 +/- 0.03662 | +0.0000 |
| RTX 3090 | deepseek2 16B Q4_0 | q4_0/q4_0 | 6.8641 +/- 0.04108 | 6.8641 +/- 0.04108 | +0.0000 |
| RTX 3090 | deepseek2 16B Q4_0 | q8_0/q8_0 | 6.1297 +/- 0.03663 | 6.1297 +/- 0.03663 | +0.0000 |
| RTX 3090 | llama 8B Q4_0 | f16/f16 | 6.9836 +/- 0.04504 | 6.9836 +/- 0.04504 | +0.0000 |
| RTX 3090 | llama 8B Q4_0 | q4_0/q4_0 | 7.0886 +/- 0.04559 | 7.0874 +/- 0.04558 | -0.0012 |
| RTX 3090 | llama 8B Q4_0 | q8_0/q8_0 | 6.9818 +/- 0.04510 | 6.9824 +/- 0.04510 | +0.0006 |
| RTX 3090 | qwen35 27B Q4_0 | f16/f16 | 6.3757 +/- 0.04159 | 6.3757 +/- 0.04159 | +0.0000 |
| RTX 3090 | qwen35 27B Q4_0 | q4_0/q4_0 | 6.3941 +/- 0.04180 | 6.3963 +/- 0.04182 | +0.0022 |
| RTX 3090 | qwen35 27B Q4_0 | q8_0/q8_0 | 6.3813 +/- 0.04169 | 6.3811 +/- 0.04167 | -0.0002 |

### KL divergence vs the master f16-KV run: wikitext-2 test, n_ctx=4096, 4 chunks (16384 tokens)

| GPU | Model | KV | side | PPL | mean KLD | median KLD | 99% KLD | max KLD | same top-1 % | mean ln(PPL/PPL_base) |
|:----|:------|:---|:-----|----:|---------:|-----------:|--------:|--------:|-------------:|----------------------:|
| RTX 3090 | llama 8B Q4_0 | q4_0/q4_0 | master | - | 0.012990 +/- 0.000308 | 0.006985 | 0.114697 | 0.919942 | 95.115 +/- 0.238 | 0.011960 +/- 0.001937 |
| RTX 3090 | llama 8B Q4_0 | q4_0/q4_0 | PR | - | 0.012844 +/- 0.000321 | 0.006797 | 0.107760 | 1.069711 | 95.408 +/- 0.231 | 0.014801 +/- 0.001924 |
| RTX 3090 | llama 8B Q4_0 | q8_0/q8_0 | master | - | 0.000574 +/- 0.000012 | 0.000329 | 0.004628 | 0.038513 | 98.852 +/- 0.118 | 0.000394 +/- 0.000443 |
| RTX 3090 | llama 8B Q4_0 | q8_0/q8_0 | PR | - | 0.000624 +/- 0.000013 | 0.000376 | 0.005278 | 0.058872 | 98.742 +/- 0.123 | 0.000432 +/- 0.000456 |
| RTX 5070 Laptop | llama 8B Q4_0 | q4_0/q4_0 | master | - | 0.012769 +/- 0.000313 | 0.006725 | 0.103546 | 0.790731 | 94.968 +/- 0.242 | 0.011886 +/- 0.001905 |
| RTX 5070 Laptop | llama 8B Q4_0 | q4_0/q4_0 | PR | - | 0.012487 +/- 0.000281 | 0.006803 | 0.107028 | 0.803367 | 94.944 +/- 0.242 | 0.008877 +/- 0.001843 |
| RTX 5070 Laptop | llama 8B Q4_0 | q8_0/q8_0 | master | - | 0.000573 +/- 0.000012 | 0.000334 | 0.004850 | 0.027627 | 98.876 +/- 0.116 | -0.000170 +/- 0.000446 |
| RTX 5070 Laptop | llama 8B Q4_0 | q8_0/q8_0 | PR | - | 0.000602 +/- 0.000011 | 0.000356 | 0.004708 | 0.019079 | 98.742 +/- 0.123 | -0.000338 +/- 0.000452 |

</details>

### Build time

- Time Up +49% for default ggml-cuda target, +53% with FA_ALL_QUANTS
- Size Up +14 MB for default ggml-cuda target, +21 MB with FA_ALL_QUANTS

<details>
<summary>Click to expand - build wall time and libggml-cuda.so size, sm_86</summary>

ggml-cuda target, sm_86 only, cold, ccache disabled, isolated, vs master `36b101543`. Added cost mainly due to 48 added template-instances:

| config | master | PR | delta |
|---|---:|---:|---|
| default, wall | 117 s | 174 s | +49% |
| default, libggml-cuda.so | 58 MB | 72 MB | +14 MB |
| FA_ALL_QUANTS, wall | 152 s | 232 s | +53% |
| FA_ALL_QUANTS, .so | 76 MB | 97 MB | +21 MB |

</details>

## Requirements

- I have read and agree with the [contributing guidelines](https://github.com/ggml-org/llama.cpp/blob/master/CONTRIBUTING.md)
- AI usage disclosure: YES - code written with Fable 5, manually reviewed by me

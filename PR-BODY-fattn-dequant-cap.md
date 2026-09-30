# CUDA: cap FA convert buffer for quantized KV

## Overview

The fattn-mma-f16 and fattn-tile kernels ingest f16 KV data layer by layer. For
non-f16 (quantized) KV types this requires a conversion buffer, pre-allocated
per #23907. This size of this buffer is equal to one layer of f16 KV and scales
with ctx length. For quantized KV types, this negates large part of VRAM savings
obtained by quantization in the first place.

Implement serial split-KV - Cap the buffer to a fixed size (64 MiB, env override
GGML_CUDA_FATTN_CONVERT_BYTES for testing only), process KV in chunks re-using
the buffer. For each processed KV chunk fattn kernel emits an unnormalized
partial and (max_row_val, row_sum), these get folded into running accumulators
by flash_attn_combine_results kernel found in fattn-common.

KV chunking is enabled based on two decisions: Firstly, in addition to picking
the best fattn kernel, ggml_cuda_flash_attn_ext_get_alloc_size() also sets the
kernel_supports_chunking flag to true, currently enabled for fattn-mma-f16 and
fattn-tile kernels. Secondly, ggml_cuda_fattn_type_chunk_enabled()
disables/enables chunking depending on other criteria such as KV type.
Currently, it returns true for quantized KV types only.

For launches with KV types f16 and f32, launches with fattn-vec kernel and
launches with FA convert buffer size below the GGML_CUDA_FATTN_CONVERT_BYTES
cap, the outputs are bit-identical to master.

For master, the scratch buffer costs one layer of f16 KV. For Qwen3.6-27B at
ctx 131072, for example, this negates 13.3% of the VRAM savings at q8_0 and 8.7%
at q4_0 KV quantization. For this PR the cost is bounded at 64 MiB max,
regardless of context length.

The f16 conversion of quantized KV chunk, similar to master, picks between
to_fp16_cuda and to_fp16_nc_cuda conversion kernels depending on the KV chunk's
allocation contiguity.

## Additional information

  - Issue #29826
  - Related PRs:
    - #23907 (merged): pre-allocation of FA f16 convert buffer
    - #21054 (closed, not merged): checks free VRAM and falls back to fattn-vec
      when the f16 convert buffer does not fit
    - #27140 (open): adds q4_1/q5_0/q5_1 conversion kernels, no overlap with
      this PR
    - #29846: the other PR for #29826, eliminates convert buffer entirely by
      fusing the conversion into fattn-mma-f16 kernel. Covers q8_0 and q4_0 KV
      types on fattn-mma-f16 only. Both PRs are complementary: fused KV pairs
      need no buffer, and every other quantized type uses the buffer cap.
      Measured data on the fork: [throughput, two GPUs](https://github.com/Abai/llama.cpp/blob/benchmark/cuda-fattn-mma-fused-dequant/bench-out/jg-multi/cuda-fattn-mma-fused-dequant/TABLES-RTX3090-sm86-24GB.md) | [VRAM](https://github.com/Abai/llama.cpp/blob/benchmark/cuda-fattn-mma-fused-dequant/bench-out/jg-multi/cuda-fattn-mma-fused-dequant/TABLES-VRAM-RTX3090-sm86-24GB.md) | [write-up](https://github.com/Abai/llama.cpp/blob/benchmark/cuda-fattn-mma-fused-dequant/PR-BODY-fattn-fused-dequant.md)

### VRAM
- Shrinks the FA convert buffer, that is ctx dependent in size (~3.7 MiB/1k
  tokens on qwen35 27B, 470 MiB at 131k ctx weight-independent and cannot be
  reduced by lowering ub) to 64MiB, regardless of context size.

`CUDA0 compute buffer`, llama-server `-ngl 99 -fa on -ub 512 --no-warmup`,
KV q8_0/q8_0, RTX 3090, qwen35 27B (weight-independent; other models and
KV pairs in the four-model ladder below), master `36b101543` vs PR
`54b94d245`, measured 2026-09-22/23:

| n_ctx | master | PR (64 MiB cap) | saved | % of q8_0 KV saving reclaimed |
|---:|---:|---:|---:|---:|
| 16384 | 160.28 MiB | 160.28 MiB | 0.00 MiB | 0% (below cap) |
| 32768 | 240.28 MiB | 176.28 MiB | 64.00 MiB | 6.7% |
| 65536 | 400.28 MiB | 208.28 MiB | 192.00 MiB | 10.0% |
| 131072 | 720.28 MiB | 272.28 MiB | 448.00 MiB | 11.7% |

<details>
<summary>Click to expand - CUDA0 compute buffer (MiB), four models, context ladder, all KV pairs, default and 32 MiB caps (RTX 3090, llama-server -ub 512)</summary>

### VRAM: CUDA0 compute buffer (MiB), KV q8_0/q8_0, -ub 512

| GPU | Model | n_ctx | master | PR (64 MiB cap) | PR (32 MiB cap) | saved MiB | KV buffer MiB |
|:----|:------|------:|-------:|--------------------:|---------:|----------:|--------------:|
| RTX 3090 | deepseek | 16384 | 104.27 | 104.27 | 104.27 | 0.00 | 258.20 |
| RTX 3090 | deepseek | 32768 | 138.27 | 138.27 | 134.05 | 0.00 | 516.39 |
| RTX 3090 | deepseek | 65536 | 206.27 | 198.11 | 166.05 | 8.16 | 1032.76 |
| RTX 3090 | deepseek | 131072 | 342.27 | 262.11 | 230.05 | 80.16 | 2065.51 |
| RTX 3090 | gemma | 16384 | 124.28 | 124.28 | 124.28 | 0.00 | 153.01 |
| RTX 3090 | gemma | 32768 | 140.28 | 140.28 | 140.28 | 0.00 | 306.01 |
| RTX 3090 | gemma | 65536 | 172.28 | 172.28 | 172.28 | 0.00 | 612.01 |
| RTX 3090 | gemma | 131072 | 272.28 | 236.28 | 236.28 | 36.00 | 1224.01 |
| RTX 3090 | llama | 16384 | 124.09 | 124.09 | 124.09 | 0.00 | 1088.00 |
| RTX 3090 | llama | 32768 | 192.09 | 140.09 | 140.09 | 52.00 | 2176.00 |
| RTX 3090 | llama | 65536 | 352.09 | 172.09 | 172.09 | 180.00 | 4352.00 |
| RTX 3090 | llama | 131072 | 672.09 | 236.09 | 236.09 | 436.00 | 8704.00 |
| RTX 3090 | qwen | 16384 | 160.28 | 160.28 | 138.28 | 0.00 | 544.00 |
| RTX 3090 | qwen | 32768 | 240.28 | 176.28 | 154.28 | 64.00 | 1088.00 |
| RTX 3090 | qwen | 65536 | 400.28 | 208.28 | 186.28 | 192.00 | 2176.00 |
| RTX 3090 | qwen | 131072 | 720.28 | 272.28 | 250.28 | 448.00 | 4352.00 |

### VRAM: CUDA0 compute buffer (MiB), KV q4_0/q4_0, -ub 512

| GPU | Model | n_ctx | master | PR (64 MiB cap) | saved MiB | KV buffer MiB |
|:----|:------|------:|-------:|--------------------:|----------:|--------------:|
| RTX 3090 | deepseek | 32768 | 138.27 | 138.27 | 0.00 | 273.38 |
| RTX 3090 | deepseek | 131072 | 342.27 | 262.11 | 80.16 | 1093.51 |
| RTX 3090 | gemma | 32768 | 140.28 | 140.28 | 0.00 | 162.01 |
| RTX 3090 | gemma | 131072 | 272.28 | 236.28 | 36.00 | 648.01 |
| RTX 3090 | llama | 32768 | 192.09 | 140.09 | 52.00 | 1152.00 |
| RTX 3090 | llama | 131072 | 672.09 | 236.09 | 436.00 | 4608.00 |
| RTX 3090 | qwen | 32768 | 240.28 | 176.28 | 64.00 | 576.00 |
| RTX 3090 | qwen | 131072 | 720.28 | 272.28 | 448.00 | 2304.00 |

### VRAM: CUDA0 compute buffer (MiB), KV q8_0/q4_0, -ub 512

| GPU | Model | n_ctx | master | PR (64 MiB cap) | saved MiB | KV buffer MiB |
|:----|:------|------:|-------:|--------------------:|----------:|--------------:|
| RTX 3090 | gemma | 32768 | 140.28 | 140.28 | 0.00 | 234.01 |
| RTX 3090 | gemma | 131072 | 272.28 | 236.28 | 36.00 | 936.01 |
| RTX 3090 | llama | 32768 | 192.09 | 140.09 | 52.00 | 1664.00 |
| RTX 3090 | llama | 131072 | 672.09 | 236.09 | 436.00 | 6656.00 |
| RTX 3090 | qwen | 32768 | 240.28 | 176.28 | 64.00 | 832.00 |
| RTX 3090 | qwen | 131072 | 720.28 | 272.28 | 448.00 | 3328.00 |

### VRAM: CUDA0 compute buffer (MiB), KV f16/f16, -ub 512

| GPU | Model | n_ctx | master | PR (64 MiB cap) | saved MiB | KV buffer MiB |
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

### Performance

- Througput and VRAM were tested on RTX 3090 and a laptop RTX 5070 8GB
- Bit identical output for f32, f16, KV pairs that fit below cap and fattn-vec
  paths
- Yields a large prefill gain with large ctx and large ubatch, when memory
  bound, by reducing VRAM pressure if chunk fits into L2 cache. Tested on RTX
  5070 Laptop, 32 MiB L2, cap of 32 MiB, yields 1.59x agains master for llama 8B

Speedup = PR t/s / master t/s at the default 64 MiB cap. llama-bench,
pp1024 at depth 32768, uniform Q4_0 weights, `-fa 1`, PR `54b94d245` vs
master `36b101543`, builds from .devops/cuda.Dockerfile, 3 reps, measured
2026-09-22/23 (RTX 3090) and 2026-09-22..25 (RTX 5070 Laptop, 115 W Dynamic Boost):

| GPU | model | q8_0/q8_0 ub8 | ub64 | ub512 | q4_0/q4_0 ub8 | ub64 | ub512 |
|---|---|---:|---:|---:|---:|---:|---:|
| RTX 3090 | llama 8B Q4_0 (head 128) | 0.96 | 0.95 | 0.97 | 0.95 | 0.95 | 0.97 |
| RTX 3090 | qwen35 27B Q4_0 | 0.98 | 0.98 | 1.00 | 0.98 | 0.98 | 1.00 |
| RTX 3090 | gemma 2B Q4_0 (under cap at d32768) | 1.00 | 1.00 | 1.01 | 1.00 | 1.00 | 1.00 |
| RTX 3090 | deepseek2 16B Q4_0 (MLA, under cap at d32768) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| RTX 5070 Laptop | llama 8B Q4_0 (head 128) | 1.03 | 1.04 | 1.02 | 1.03 | 1.03 | 1.02 |
| RTX 5070 Laptop | gemma 2B Q4_0 (under cap at d32768) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

<details>
<summary>Click to expand - Full llama-bench matrix, both GPUs: ub ladders 1-512 for q8_0/q8_0, q4_0/q4_0, q8_0/q4_0; q8_0 at d65536 and d131072; pp512 and tg128 depth ladders; f16/f16 controls; columns PR 64MiB (default cap) and PR 32MiB (env override)</summary>


### KV q8_0/q8_0, pp1024@d32768

| GPU | Model | Microbatch size | Test | t/s master | t/s PR 32MiB | t/s PR 64MiB | Speedup 32MiB | Speedup 64MiB |
|:----|:------|--------:|:-----|-----------:|-----------:|-----------:|-----------:|-----------:|
| RTX 3090 | deepseek2 16B Q4_0 | 1 | pp1024@d32768 | 145.77 | 130.98 | 145.92 | 0.90 | 1.00 |
| RTX 3090 | deepseek2 16B Q4_0 | 2 | pp1024@d32768 | 250.51 | 226.76 | 250.30 | 0.91 | 1.00 |
| RTX 3090 | deepseek2 16B Q4_0 | 4 | pp1024@d32768 | 378.46 | 343.47 | 378.31 | 0.91 | 1.00 |
| RTX 3090 | deepseek2 16B Q4_0 | 8 | pp1024@d32768 | 518.61 | 483.62 | 518.24 | 0.93 | 1.00 |
| RTX 3090 | deepseek2 16B Q4_0 | 16 | pp1024@d32768 | 756.04 | 714.58 | 755.54 | 0.95 | 1.00 |
| RTX 3090 | deepseek2 16B Q4_0 | 32 | pp1024@d32768 | 985.38 | 946.25 | 985.39 | 0.96 | 1.00 |
| RTX 3090 | deepseek2 16B Q4_0 | 64 | pp1024@d32768 | 1157.26 | 1121.88 | 1156.60 | 0.97 | 1.00 |
| RTX 3090 | deepseek2 16B Q4_0 | 128 | pp1024@d32768 | 1253.91 | 1216.10 | 1254.09 | 0.97 | 1.00 |
| RTX 3090 | deepseek2 16B Q4_0 | 256 | pp1024@d32768 | 1293.96 | 1267.66 | 1293.90 | 0.98 | 1.00 |
| RTX 3090 | deepseek2 16B Q4_0 | 512 | pp1024@d32768 | 1393.46 | 1363.99 | 1393.34 | 0.98 | 1.00 |
| RTX 3090 | gemma 2B Q4_0 | 1 | pp1024@d32768 | 277.83 | 276.74 | 277.13 | 1.00 | 1.00 |
| RTX 3090 | gemma 2B Q4_0 | 2 | pp1024@d32768 | 444.65 | 421.29 | 441.43 | 0.95 | 0.99 |
| RTX 3090 | gemma 2B Q4_0 | 4 | pp1024@d32768 | 850.77 | 796.51 | 848.23 | 0.94 | 1.00 |
| RTX 3090 | gemma 2B Q4_0 | 8 | pp1024@d32768 | 1318.19 | 1234.31 | 1318.24 | 0.94 | 1.00 |
| RTX 3090 | gemma 2B Q4_0 | 16 | pp1024@d32768 | 2369.81 | 2234.08 | 2371.84 | 0.94 | 1.00 |
| RTX 3090 | gemma 2B Q4_0 | 32 | pp1024@d32768 | 3659.76 | 3471.08 | 3665.09 | 0.95 | 1.00 |
| RTX 3090 | gemma 2B Q4_0 | 64 | pp1024@d32768 | 5286.04 | 4999.23 | 5283.22 | 0.95 | 1.00 |
| RTX 3090 | gemma 2B Q4_0 | 128 | pp1024@d32768 | 6475.53 | 6159.18 | 6456.27 | 0.95 | 1.00 |
| RTX 3090 | gemma 2B Q4_0 | 256 | pp1024@d32768 | 7720.88 | 7464.03 | 7730.06 | 0.97 | 1.00 |
| RTX 3090 | gemma 2B Q4_0 | 512 | pp1024@d32768 | 8188.88 | 7961.34 | 8231.99 | 0.97 | 1.01 |
| RTX 3090 | llama 8B Q4_0 | 1 | pp1024@d32768 | 78.51 | 78.42 | 78.35 | 1.00 | 1.00 |
| RTX 3090 | llama 8B Q4_0 | 2 | pp1024@d32768 | 104.01 | 97.16 | 100.40 | 0.93 | 0.97 |
| RTX 3090 | llama 8B Q4_0 | 4 | pp1024@d32768 | 201.03 | 184.42 | 192.06 | 0.92 | 0.96 |
| RTX 3090 | llama 8B Q4_0 | 8 | pp1024@d32768 | 347.35 | 317.11 | 331.86 | 0.91 | 0.96 |
| RTX 3090 | llama 8B Q4_0 | 16 | pp1024@d32768 | 658.35 | 582.96 | 621.06 | 0.89 | 0.94 |
| RTX 3090 | llama 8B Q4_0 | 32 | pp1024@d32768 | 1122.98 | 1008.37 | 1058.30 | 0.90 | 0.94 |
| RTX 3090 | llama 8B Q4_0 | 64 | pp1024@d32768 | 1649.66 | 1502.67 | 1572.38 | 0.91 | 0.95 |
| RTX 3090 | llama 8B Q4_0 | 128 | pp1024@d32768 | 1650.90 | 1478.58 | 1564.24 | 0.90 | 0.95 |
| RTX 3090 | llama 8B Q4_0 | 256 | pp1024@d32768 | 2173.79 | 2066.16 | 2109.89 | 0.95 | 0.97 |
| RTX 3090 | llama 8B Q4_0 | 512 | pp1024@d32768 | 2322.83 | 2214.56 | 2257.66 | 0.95 | 0.97 |
| RTX 3090 | qwen35 27B Q4_0 | 1 | pp1024@d32768 | 41.08 | 41.07 | 41.08 | 1.00 | 1.00 |
| RTX 3090 | qwen35 27B Q4_0 | 2 | pp1024@d32768 | 71.21 | 68.21 | 69.84 | 0.96 | 0.98 |
| RTX 3090 | qwen35 27B Q4_0 | 4 | pp1024@d32768 | 126.05 | 120.30 | 122.80 | 0.95 | 0.97 |
| RTX 3090 | qwen35 27B Q4_0 | 8 | pp1024@d32768 | 187.67 | 179.06 | 183.54 | 0.95 | 0.98 |
| RTX 3090 | qwen35 27B Q4_0 | 16 | pp1024@d32768 | 377.52 | 360.36 | 368.41 | 0.95 | 0.98 |
| RTX 3090 | qwen35 27B Q4_0 | 32 | pp1024@d32768 | 576.19 | 554.35 | 564.09 | 0.96 | 0.98 |
| RTX 3090 | qwen35 27B Q4_0 | 64 | pp1024@d32768 | 836.47 | 808.00 | 819.89 | 0.97 | 0.98 |
| RTX 3090 | qwen35 27B Q4_0 | 128 | pp1024@d32768 | 909.26 | 872.42 | 890.55 | 0.96 | 0.98 |
| RTX 3090 | qwen35 27B Q4_0 | 256 | pp1024@d32768 | 1078.35 | 1063.94 | 1072.20 | 0.99 | 0.99 |
| RTX 3090 | qwen35 27B Q4_0 | 512 | pp1024@d32768 | 1122.16 | 1108.56 | 1116.59 | 0.99 | 1.00 |
| RTX 5070 Laptop | gemma 2B Q4_0 | 1 | pp1024@d32768 | 155.03 | 154.88 | 154.80 | 1.00 | 1.00 |
| RTX 5070 Laptop | gemma 2B Q4_0 | 2 | pp1024@d32768 | 216.86 | 216.84 | 216.88 | 1.00 | 1.00 |
| RTX 5070 Laptop | gemma 2B Q4_0 | 4 | pp1024@d32768 | 576.50 | 558.98 | 576.15 | 0.97 | 1.00 |
| RTX 5070 Laptop | gemma 2B Q4_0 | 8 | pp1024@d32768 | 974.16 | 935.47 | 974.12 | 0.96 | 1.00 |
| RTX 5070 Laptop | gemma 2B Q4_0 | 16 | pp1024@d32768 | 1803.00 | 1741.95 | 1803.37 | 0.97 | 1.00 |
| RTX 5070 Laptop | gemma 2B Q4_0 | 32 | pp1024@d32768 | 2818.66 | 2731.62 | 2819.68 | 0.97 | 1.00 |
| RTX 5070 Laptop | gemma 2B Q4_0 | 64 | pp1024@d32768 | 3846.81 | 3757.10 | 3849.52 | 0.98 | 1.00 |
| RTX 5070 Laptop | gemma 2B Q4_0 | 128 | pp1024@d32768 | 4478.42 | 4326.32 | 4475.90 | 0.97 | 1.00 |
| RTX 5070 Laptop | gemma 2B Q4_0 | 256 | pp1024@d32768 | 5050.58 | 4950.06 | 5052.66 | 0.98 | 1.00 |
| RTX 5070 Laptop | gemma 2B Q4_0 | 512 | pp1024@d32768 | 5261.35 | 5165.53 | 5254.16 | 0.98 | 1.00 |
| RTX 5070 Laptop | llama 8B Q4_0 | 1 | pp1024@d32768 | 40.91 | 40.80 | 40.89 | 1.00 | 1.00 |
| RTX 5070 Laptop | llama 8B Q4_0 | 2 | pp1024@d32768 | 66.02 | 66.21 | 66.21 | 1.00 | 1.00 |
| RTX 5070 Laptop | llama 8B Q4_0 | 4 | pp1024@d32768 | 94.82 | 111.75 | 98.94 | 1.18 | 1.04 |
| RTX 5070 Laptop | llama 8B Q4_0 | 8 | pp1024@d32768 | 178.98 | 203.36 | 184.57 | 1.14 | 1.03 |
| RTX 5070 Laptop | llama 8B Q4_0 | 16 | pp1024@d32768 | 345.75 | 374.02 | 350.48 | 1.08 | 1.01 |
| RTX 5070 Laptop | llama 8B Q4_0 | 32 | pp1024@d32768 | 531.51 | 607.18 | 541.21 | 1.14 | 1.02 |
| RTX 5070 Laptop | llama 8B Q4_0 | 64 | pp1024@d32768 | 709.32 | 911.95 | 737.38 | 1.29 | 1.04 |
| RTX 5070 Laptop | llama 8B Q4_0 | 128 | pp1024@d32768 | 832.13 | 1199.16 | 861.23 | 1.44 | 1.03 |
| RTX 5070 Laptop | llama 8B Q4_0 | 256 | pp1024@d32768 | 914.63 | 1390.45 | 930.16 | 1.52 | 1.02 |
| RTX 5070 Laptop | llama 8B Q4_0 | 512 | pp1024@d32768 | 941.39 | 1494.30 | 957.05 | 1.59 | 1.02 |

### KV q8_0/q8_0, pp1024@d65536

| GPU | Model | Microbatch size | Test | t/s master | t/s PR 32MiB | t/s PR 64MiB | Speedup 32MiB | Speedup 64MiB |
|:----|:------|--------:|:-----|-----------:|-----------:|-----------:|-----------:|-----------:|
| RTX 3090 | deepseek2 16B Q4_0 | 1 | pp1024@d65536 | 101.23 | 86.11 | 93.32 | 0.85 | 0.92 |
| RTX 3090 | deepseek2 16B Q4_0 | 2 | pp1024@d65536 | 179.28 | 154.04 | 166.40 | 0.86 | 0.93 |
| RTX 3090 | deepseek2 16B Q4_0 | 4 | pp1024@d65536 | 279.82 | 237.82 | 256.96 | 0.85 | 0.92 |
| RTX 3090 | deepseek2 16B Q4_0 | 8 | pp1024@d65536 | 381.57 | 343.73 | 360.78 | 0.90 | 0.95 |
| RTX 3090 | deepseek2 16B Q4_0 | 16 | pp1024@d65536 | 527.68 | 487.78 | 506.61 | 0.92 | 0.96 |
| RTX 3090 | deepseek2 16B Q4_0 | 32 | pp1024@d65536 | 654.03 | 620.07 | 635.29 | 0.95 | 0.97 |
| RTX 3090 | deepseek2 16B Q4_0 | 64 | pp1024@d65536 | 740.73 | 714.16 | 725.14 | 0.96 | 0.98 |
| RTX 3090 | deepseek2 16B Q4_0 | 128 | pp1024@d65536 | 784.64 | 755.20 | 767.53 | 0.96 | 0.98 |
| RTX 3090 | deepseek2 16B Q4_0 | 256 | pp1024@d65536 | 739.92 | 726.25 | 731.47 | 0.98 | 0.99 |
| RTX 3090 | deepseek2 16B Q4_0 | 512 | pp1024@d65536 | 773.38 | 758.51 | 764.10 | 0.98 | 0.99 |

### KV q8_0/q8_0, pp1024@d131072

| GPU | Model | Microbatch size | Test | t/s master | t/s PR 32MiB | t/s PR 64MiB | Speedup 32MiB | Speedup 64MiB |
|:----|:------|--------:|:-----|-----------:|-----------:|-----------:|-----------:|-----------:|
| RTX 3090 | gemma 2B Q4_0 | 1 | pp1024@d131072 | 120.10 | 120.10 | 120.12 | 1.00 | 1.00 |
| RTX 3090 | gemma 2B Q4_0 | 2 | pp1024@d131072 | 201.29 | 166.97 | 184.69 | 0.83 | 0.92 |
| RTX 3090 | gemma 2B Q4_0 | 4 | pp1024@d131072 | 395.53 | 324.73 | 360.11 | 0.82 | 0.91 |
| RTX 3090 | gemma 2B Q4_0 | 8 | pp1024@d131072 | 691.45 | 537.25 | 615.96 | 0.78 | 0.89 |
| RTX 3090 | gemma 2B Q4_0 | 16 | pp1024@d131072 | 1199.04 | 1014.30 | 1110.45 | 0.85 | 0.93 |
| RTX 3090 | gemma 2B Q4_0 | 32 | pp1024@d131072 | 1803.78 | 1588.90 | 1694.82 | 0.88 | 0.94 |
| RTX 3090 | gemma 2B Q4_0 | 64 | pp1024@d131072 | 2370.79 | 2155.48 | 2250.78 | 0.91 | 0.95 |
| RTX 3090 | gemma 2B Q4_0 | 128 | pp1024@d131072 | 2764.40 | 2578.70 | 2658.10 | 0.93 | 0.96 |
| RTX 3090 | gemma 2B Q4_0 | 256 | pp1024@d131072 | 3073.48 | 2938.84 | 2995.03 | 0.96 | 0.97 |
| RTX 3090 | gemma 2B Q4_0 | 512 | pp1024@d131072 | 3123.56 | 3014.75 | 3064.83 | 0.97 | 0.98 |
| RTX 5070 Laptop | gemma 2B Q4_0 | 1 | pp1024@d131072 | 65.55 | 65.61 | 65.59 | 1.00 | 1.00 |
| RTX 5070 Laptop | gemma 2B Q4_0 | 2 | pp1024@d131072 | 79.18 | 79.16 | 79.17 | 1.00 | 1.00 |
| RTX 5070 Laptop | gemma 2B Q4_0 | 4 | pp1024@d131072 | 193.02 | 223.56 | 199.89 | 1.16 | 1.04 |
| RTX 5070 Laptop | gemma 2B Q4_0 | 8 | pp1024@d131072 | 357.93 | 391.14 | 361.76 | 1.09 | 1.01 |
| RTX 5070 Laptop | gemma 2B Q4_0 | 16 | pp1024@d131072 | 692.70 | 717.77 | 687.28 | 1.04 | 0.99 |
| RTX 5070 Laptop | gemma 2B Q4_0 | 32 | pp1024@d131072 | 1065.72 | 1083.65 | 1088.48 | 1.02 | 1.02 |
| RTX 5070 Laptop | gemma 2B Q4_0 | 64 | pp1024@d131072 | 1444.93 | 1447.62 | 1441.49 | 1.00 | 1.00 |
| RTX 5070 Laptop | gemma 2B Q4_0 | 128 | pp1024@d131072 | 1700.91 | 1667.80 | 1668.58 | 0.98 | 0.98 |
| RTX 5070 Laptop | gemma 2B Q4_0 | 256 | pp1024@d131072 | 1879.27 | 1855.76 | 1861.23 | 0.99 | 0.99 |
| RTX 5070 Laptop | gemma 2B Q4_0 | 512 | pp1024@d131072 | 1960.93 | 1941.64 | 1945.01 | 0.99 | 0.99 |

### KV q8_0/q8_0, pp512 depth ladder

| GPU | Model | Depth | Test | t/s master | t/s PR 32MiB | t/s PR 64MiB | Speedup 32MiB | Speedup 64MiB |
|:----|:------|--------:|:-----|-----------:|-----------:|-----------:|-----------:|-----------:|
| RTX 3090 | deepseek2 16B Q4_0 | 0 | pp512@d0 | 6811.54 | 6824.68 | 6836.19 | 1.00 | 1.00 |
| RTX 3090 | deepseek2 16B Q4_0 | 16384 | pp512@d16384 | 2314.91 | 2313.75 | 2306.10 | 1.00 | 1.00 |
| RTX 3090 | deepseek2 16B Q4_0 | 32768 | pp512@d32768 | 1390.47 | 1360.29 | 1390.57 | 0.98 | 1.00 |
| RTX 3090 | deepseek2 16B Q4_0 | 65536 | pp512@d65536 | 771.90 | 757.59 | 762.90 | 0.98 | 0.99 |
| RTX 3090 | gemma 2B Q4_0 | 0 | pp512@d0 | 16792.49 | 16692.00 | 16599.81 | 0.99 | 0.99 |
| RTX 3090 | gemma 2B Q4_0 | 16384 | pp512@d16384 | 10939.58 | 10891.36 | 10926.62 | 1.00 | 1.00 |
| RTX 3090 | gemma 2B Q4_0 | 32768 | pp512@d32768 | 8071.74 | 7677.70 | 7992.26 | 0.95 | 0.99 |
| RTX 3090 | gemma 2B Q4_0 | 65536 | pp512@d65536 | 5217.74 | 5022.54 | 5091.01 | 0.96 | 0.98 |
| RTX 3090 | llama 8B Q4_0 | 0 | pp512@d0 | 5802.90 | 5811.34 | 5797.77 | 1.00 | 1.00 |
| RTX 3090 | llama 8B Q4_0 | 16384 | pp512@d16384 | 3300.56 | 3192.08 | 3232.73 | 0.97 | 0.98 |
| RTX 3090 | llama 8B Q4_0 | 32768 | pp512@d32768 | 2319.45 | 2210.39 | 2252.42 | 0.95 | 0.97 |
| RTX 3090 | llama 8B Q4_0 | 65536 | pp512@d65536 | 1447.41 | 1362.95 | 1397.29 | 0.94 | 0.97 |
| RTX 3090 | qwen35 27B Q4_0 | 0 | pp512@d0 | 1536.20 | 1537.02 | 1537.95 | 1.00 | 1.00 |
| RTX 3090 | qwen35 27B Q4_0 | 16384 | pp512@d16384 | 1296.52 | 1286.12 | 1291.97 | 0.99 | 1.00 |
| RTX 3090 | qwen35 27B Q4_0 | 32768 | pp512@d32768 | 1120.42 | 1107.05 | 1115.01 | 0.99 | 1.00 |
| RTX 3090 | qwen35 27B Q4_0 | 65536 | pp512@d65536 | 880.03 | 864.50 | 875.23 | 0.98 | 0.99 |
| RTX 5070 Laptop | gemma 2B Q4_0 | 0 | pp512@d0 | 11113.87 | 11104.13 | 11153.58 | 1.00 | 1.00 |
| RTX 5070 Laptop | gemma 2B Q4_0 | 16384 | pp512@d16384 | 7055.99 | 7063.15 | 7077.65 | 1.00 | 1.00 |
| RTX 5070 Laptop | gemma 2B Q4_0 | 32768 | pp512@d32768 | 5152.73 | 5062.50 | 5144.74 | 0.98 | 1.00 |
| RTX 5070 Laptop | gemma 2B Q4_0 | 65536 | pp512@d65536 | 3320.05 | 3269.63 | 3287.85 | 0.98 | 0.99 |
| RTX 5070 Laptop | llama 8B Q4_0 | 0 | pp512@d0 | 3708.56 | 3719.80 | 3716.05 | 1.00 | 1.00 |
| RTX 5070 Laptop | llama 8B Q4_0 | 16384 | pp512@d16384 | 1558.30 | 2118.05 | 1557.33 | 1.36 | 1.00 |
| RTX 5070 Laptop | llama 8B Q4_0 | 32768 | pp512@d32768 | 958.52 | 1479.03 | 979.73 | 1.54 | 1.02 |

### KV q8_0/q8_0, tg128 depth ladder

| GPU | Model | Depth | Test | t/s master | t/s PR 32MiB | t/s PR 64MiB | Speedup 32MiB | Speedup 64MiB |
|:----|:------|--------:|:-----|-----------:|-----------:|-----------:|-----------:|-----------:|
| RTX 3090 | deepseek2 16B Q4_0 | 0 | tg128@d0 | 266.31 | 266.93 | 266.62 | 1.00 | 1.00 |
| RTX 3090 | deepseek2 16B Q4_0 | 16384 | tg128@d16384 | 179.63 | 179.22 | 179.23 | 1.00 | 1.00 |
| RTX 3090 | deepseek2 16B Q4_0 | 32768 | tg128@d32768 | 139.98 | 124.50 | 139.93 | 0.89 | 1.00 |
| RTX 3090 | deepseek2 16B Q4_0 | 65536 | tg128@d65536 | 98.54 | 83.53 | 90.79 | 0.85 | 0.92 |
| RTX 3090 | gemma 2B Q4_0 | 0 | tg128@d0 | 352.11 | 352.37 | 352.75 | 1.00 | 1.00 |
| RTX 3090 | gemma 2B Q4_0 | 16384 | tg128@d16384 | 278.64 | 279.02 | 278.77 | 1.00 | 1.00 |
| RTX 3090 | gemma 2B Q4_0 | 32768 | tg128@d32768 | 231.73 | 231.21 | 231.64 | 1.00 | 1.00 |
| RTX 3090 | gemma 2B Q4_0 | 65536 | tg128@d65536 | 169.36 | 169.11 | 169.20 | 1.00 | 1.00 |
| RTX 3090 | llama 8B Q4_0 | 0 | tg128@d0 | 154.62 | 154.59 | 154.58 | 1.00 | 1.00 |
| RTX 3090 | llama 8B Q4_0 | 16384 | tg128@d16384 | 101.74 | 101.65 | 101.70 | 1.00 | 1.00 |
| RTX 3090 | llama 8B Q4_0 | 32768 | tg128@d32768 | 75.22 | 75.12 | 75.06 | 1.00 | 1.00 |
| RTX 3090 | llama 8B Q4_0 | 65536 | tg128@d65536 | 49.21 | 49.20 | 49.18 | 1.00 | 1.00 |
| RTX 3090 | qwen35 27B Q4_0 | 0 | tg128@d0 | 46.59 | 46.59 | 46.61 | 1.00 | 1.00 |
| RTX 3090 | qwen35 27B Q4_0 | 16384 | tg128@d16384 | 42.53 | 42.53 | 42.53 | 1.00 | 1.00 |
| RTX 3090 | qwen35 27B Q4_0 | 32768 | tg128@d32768 | 38.97 | 38.98 | 38.98 | 1.00 | 1.00 |
| RTX 3090 | qwen35 27B Q4_0 | 65536 | tg128@d65536 | 33.23 | 33.23 | 33.24 | 1.00 | 1.00 |
| RTX 5070 Laptop | gemma 2B Q4_0 | 0 | tg128@d0 | 199.19 | 199.30 | 199.30 | 1.00 | 1.00 |
| RTX 5070 Laptop | gemma 2B Q4_0 | 16384 | tg128@d16384 | 156.19 | 156.19 | 156.12 | 1.00 | 1.00 |
| RTX 5070 Laptop | gemma 2B Q4_0 | 32768 | tg128@d32768 | 126.98 | 127.09 | 127.02 | 1.00 | 1.00 |
| RTX 5070 Laptop | gemma 2B Q4_0 | 65536 | tg128@d65536 | 92.12 | 92.06 | 92.06 | 1.00 | 1.00 |
| RTX 5070 Laptop | llama 8B Q4_0 | 0 | tg128@d0 | 76.98 | 77.00 | 76.99 | 1.00 | 1.00 |
| RTX 5070 Laptop | llama 8B Q4_0 | 16384 | tg128@d16384 | 51.42 | 51.41 | 51.40 | 1.00 | 1.00 |
| RTX 5070 Laptop | llama 8B Q4_0 | 32768 | tg128@d32768 | 38.80 | 38.77 | 38.71 | 1.00 | 1.00 |

### KV q4_0/q4_0, pp1024@d32768

| GPU | Model | Microbatch size | Test | t/s master | t/s PR 32MiB | t/s PR 64MiB | Speedup 32MiB | Speedup 64MiB |
|:----|:------|--------:|:-----|-----------:|-----------:|-----------:|-----------:|-----------:|
| RTX 3090 | deepseek2 16B Q4_0 | 1 | pp1024@d32768 | 148.90 | 132.68 | 148.69 | 0.89 | 1.00 |
| RTX 3090 | deepseek2 16B Q4_0 | 2 | pp1024@d32768 | 255.96 | 228.59 | 255.31 | 0.89 | 1.00 |
| RTX 3090 | deepseek2 16B Q4_0 | 4 | pp1024@d32768 | 387.07 | 346.64 | 386.63 | 0.90 | 1.00 |
| RTX 3090 | deepseek2 16B Q4_0 | 8 | pp1024@d32768 | 518.27 | 481.63 | 518.11 | 0.93 | 1.00 |
| RTX 3090 | deepseek2 16B Q4_0 | 16 | pp1024@d32768 | 764.32 | 720.44 | 764.09 | 0.94 | 1.00 |
| RTX 3090 | deepseek2 16B Q4_0 | 32 | pp1024@d32768 | 992.69 | 952.41 | 992.33 | 0.96 | 1.00 |
| RTX 3090 | deepseek2 16B Q4_0 | 64 | pp1024@d32768 | 1164.77 | 1128.36 | 1164.84 | 0.97 | 1.00 |
| RTX 3090 | deepseek2 16B Q4_0 | 128 | pp1024@d32768 | 1260.95 | 1221.50 | 1258.38 | 0.97 | 1.00 |
| RTX 3090 | deepseek2 16B Q4_0 | 256 | pp1024@d32768 | 1297.21 | 1270.93 | 1296.94 | 0.98 | 1.00 |
| RTX 3090 | deepseek2 16B Q4_0 | 512 | pp1024@d32768 | 1395.98 | 1364.70 | 1393.52 | 0.98 | 1.00 |
| RTX 3090 | gemma 2B Q4_0 | 1 | pp1024@d32768 | 249.90 | 249.98 | 250.04 | 1.00 | 1.00 |
| RTX 3090 | gemma 2B Q4_0 | 2 | pp1024@d32768 | 456.08 | 423.85 | 454.42 | 0.93 | 1.00 |
| RTX 3090 | gemma 2B Q4_0 | 4 | pp1024@d32768 | 861.83 | 812.36 | 860.37 | 0.94 | 1.00 |
| RTX 3090 | gemma 2B Q4_0 | 8 | pp1024@d32768 | 1356.49 | 1273.03 | 1353.41 | 0.94 | 1.00 |
| RTX 3090 | gemma 2B Q4_0 | 16 | pp1024@d32768 | 2435.47 | 2282.91 | 2432.69 | 0.94 | 1.00 |
| RTX 3090 | gemma 2B Q4_0 | 32 | pp1024@d32768 | 3751.13 | 3559.65 | 3748.47 | 0.95 | 1.00 |
| RTX 3090 | gemma 2B Q4_0 | 64 | pp1024@d32768 | 5427.51 | 5111.59 | 5403.84 | 0.94 | 1.00 |
| RTX 3090 | gemma 2B Q4_0 | 128 | pp1024@d32768 | 6580.49 | 6251.49 | 6578.40 | 0.95 | 1.00 |
| RTX 3090 | gemma 2B Q4_0 | 256 | pp1024@d32768 | 7771.85 | 7496.62 | 7766.96 | 0.96 | 1.00 |
| RTX 3090 | gemma 2B Q4_0 | 512 | pp1024@d32768 | 8224.90 | 7994.52 | 8243.54 | 0.97 | 1.00 |
| RTX 3090 | llama 8B Q4_0 | 1 | pp1024@d32768 | 75.26 | 75.14 | 75.21 | 1.00 | 1.00 |
| RTX 3090 | llama 8B Q4_0 | 2 | pp1024@d32768 | 110.44 | 100.43 | 105.31 | 0.91 | 0.95 |
| RTX 3090 | llama 8B Q4_0 | 4 | pp1024@d32768 | 211.46 | 192.04 | 202.32 | 0.91 | 0.96 |
| RTX 3090 | llama 8B Q4_0 | 8 | pp1024@d32768 | 365.26 | 331.00 | 348.74 | 0.91 | 0.95 |
| RTX 3090 | llama 8B Q4_0 | 16 | pp1024@d32768 | 694.79 | 606.51 | 651.16 | 0.87 | 0.94 |
| RTX 3090 | llama 8B Q4_0 | 32 | pp1024@d32768 | 1161.85 | 1032.06 | 1089.75 | 0.89 | 0.94 |
| RTX 3090 | llama 8B Q4_0 | 64 | pp1024@d32768 | 1702.81 | 1542.28 | 1616.51 | 0.91 | 0.95 |
| RTX 3090 | llama 8B Q4_0 | 128 | pp1024@d32768 | 1679.72 | 1504.16 | 1589.92 | 0.90 | 0.95 |
| RTX 3090 | llama 8B Q4_0 | 256 | pp1024@d32768 | 2199.08 | 2085.79 | 2127.69 | 0.95 | 0.97 |
| RTX 3090 | llama 8B Q4_0 | 512 | pp1024@d32768 | 2336.18 | 2225.54 | 2270.45 | 0.95 | 0.97 |
| RTX 3090 | qwen35 27B Q4_0 | 1 | pp1024@d32768 | 39.48 | 39.46 | 39.48 | 1.00 | 1.00 |
| RTX 3090 | qwen35 27B Q4_0 | 2 | pp1024@d32768 | 71.73 | 68.56 | 70.31 | 0.96 | 0.98 |
| RTX 3090 | qwen35 27B Q4_0 | 4 | pp1024@d32768 | 125.16 | 120.36 | 122.89 | 0.96 | 0.98 |
| RTX 3090 | qwen35 27B Q4_0 | 8 | pp1024@d32768 | 186.22 | 177.82 | 182.34 | 0.95 | 0.98 |
| RTX 3090 | qwen35 27B Q4_0 | 16 | pp1024@d32768 | 381.12 | 362.37 | 371.46 | 0.95 | 0.97 |
| RTX 3090 | qwen35 27B Q4_0 | 32 | pp1024@d32768 | 581.37 | 557.38 | 568.09 | 0.96 | 0.98 |
| RTX 3090 | qwen35 27B Q4_0 | 64 | pp1024@d32768 | 838.50 | 808.59 | 821.53 | 0.96 | 0.98 |
| RTX 3090 | qwen35 27B Q4_0 | 128 | pp1024@d32768 | 910.98 | 872.57 | 892.02 | 0.96 | 0.98 |
| RTX 3090 | qwen35 27B Q4_0 | 256 | pp1024@d32768 | 1080.48 | 1064.10 | 1072.93 | 0.98 | 0.99 |
| RTX 3090 | qwen35 27B Q4_0 | 512 | pp1024@d32768 | 1122.87 | 1109.25 | 1117.40 | 0.99 | 1.00 |
| RTX 5070 Laptop | gemma 2B Q4_0 | 1 | pp1024@d32768 | 56.31 | 56.31 | 56.30 | 1.00 | 1.00 |
| RTX 5070 Laptop | gemma 2B Q4_0 | 2 | pp1024@d32768 | 97.56 | 97.56 | 97.55 | 1.00 | 1.00 |
| RTX 5070 Laptop | gemma 2B Q4_0 | 4 | pp1024@d32768 | 596.01 | 579.49 | 596.30 | 0.97 | 1.00 |
| RTX 5070 Laptop | gemma 2B Q4_0 | 8 | pp1024@d32768 | 986.58 | 951.73 | 987.30 | 0.96 | 1.00 |
| RTX 5070 Laptop | gemma 2B Q4_0 | 16 | pp1024@d32768 | 1811.57 | 1743.30 | 1812.92 | 0.96 | 1.00 |
| RTX 5070 Laptop | gemma 2B Q4_0 | 32 | pp1024@d32768 | 2798.69 | 2702.51 | 2797.17 | 0.97 | 1.00 |
| RTX 5070 Laptop | gemma 2B Q4_0 | 64 | pp1024@d32768 | 3921.89 | 3829.50 | 3932.94 | 0.98 | 1.00 |
| RTX 5070 Laptop | gemma 2B Q4_0 | 128 | pp1024@d32768 | 4557.38 | 4413.25 | 4562.63 | 0.97 | 1.00 |
| RTX 5070 Laptop | gemma 2B Q4_0 | 256 | pp1024@d32768 | 5108.08 | 4968.19 | 5108.81 | 0.97 | 1.00 |
| RTX 5070 Laptop | gemma 2B Q4_0 | 512 | pp1024@d32768 | 5289.36 | 5206.18 | 5288.51 | 0.98 | 1.00 |
| RTX 5070 Laptop | llama 8B Q4_0 | 1 | pp1024@d32768 | 18.61 | 18.61 | 18.61 | 1.00 | 1.00 |
| RTX 5070 Laptop | llama 8B Q4_0 | 2 | pp1024@d32768 | 33.92 | 33.91 | 33.91 | 1.00 | 1.00 |
| RTX 5070 Laptop | llama 8B Q4_0 | 4 | pp1024@d32768 | 100.92 | 120.38 | 105.11 | 1.19 | 1.04 |
| RTX 5070 Laptop | llama 8B Q4_0 | 8 | pp1024@d32768 | 189.17 | 217.23 | 194.94 | 1.15 | 1.03 |
| RTX 5070 Laptop | llama 8B Q4_0 | 16 | pp1024@d32768 | 367.89 | 400.89 | 373.32 | 1.09 | 1.01 |
| RTX 5070 Laptop | llama 8B Q4_0 | 32 | pp1024@d32768 | 551.46 | 621.41 | 560.73 | 1.13 | 1.02 |
| RTX 5070 Laptop | llama 8B Q4_0 | 64 | pp1024@d32768 | 733.81 | 947.55 | 759.00 | 1.29 | 1.03 |
| RTX 5070 Laptop | llama 8B Q4_0 | 128 | pp1024@d32768 | 855.63 | 1230.05 | 874.95 | 1.44 | 1.02 |
| RTX 5070 Laptop | llama 8B Q4_0 | 256 | pp1024@d32768 | 927.81 | 1415.97 | 937.37 | 1.53 | 1.01 |
| RTX 5070 Laptop | llama 8B Q4_0 | 512 | pp1024@d32768 | 955.97 | 1508.48 | 979.84 | 1.58 | 1.02 |

### KV q8_0/q4_0, pp1024@d32768

| GPU | Model | Microbatch size | Test | t/s master | t/s PR 32MiB | t/s PR 64MiB | Speedup 32MiB | Speedup 64MiB |
|:----|:------|--------:|:-----|-----------:|-----------:|-----------:|-----------:|-----------:|
| RTX 3090 | gemma 2B Q4_0 | 1 | pp1024@d32768 | 252.74 | 252.75 | 252.67 | 1.00 | 1.00 |
| RTX 3090 | gemma 2B Q4_0 | 2 | pp1024@d32768 | 448.49 | 422.17 | 447.01 | 0.94 | 1.00 |
| RTX 3090 | gemma 2B Q4_0 | 4 | pp1024@d32768 | 866.43 | 815.53 | 864.68 | 0.94 | 1.00 |
| RTX 3090 | gemma 2B Q4_0 | 8 | pp1024@d32768 | 1327.67 | 1259.80 | 1328.07 | 0.95 | 1.00 |
| RTX 3090 | gemma 2B Q4_0 | 16 | pp1024@d32768 | 2403.88 | 2254.81 | 2404.01 | 0.94 | 1.00 |
| RTX 3090 | gemma 2B Q4_0 | 32 | pp1024@d32768 | 3702.92 | 3500.71 | 3704.10 | 0.95 | 1.00 |
| RTX 3090 | gemma 2B Q4_0 | 64 | pp1024@d32768 | 5351.55 | 5050.07 | 5351.80 | 0.94 | 1.00 |
| RTX 3090 | gemma 2B Q4_0 | 128 | pp1024@d32768 | 6507.80 | 6176.98 | 6503.15 | 0.95 | 1.00 |
| RTX 3090 | gemma 2B Q4_0 | 256 | pp1024@d32768 | 7752.63 | 7490.89 | 7746.78 | 0.97 | 1.00 |
| RTX 3090 | gemma 2B Q4_0 | 512 | pp1024@d32768 | 8214.32 | 7926.90 | 8211.31 | 0.97 | 1.00 |
| RTX 3090 | llama 8B Q4_0 | 1 | pp1024@d32768 | 74.45 | 74.42 | 74.43 | 1.00 | 1.00 |
| RTX 3090 | llama 8B Q4_0 | 2 | pp1024@d32768 | 107.46 | 99.49 | 103.31 | 0.93 | 0.96 |
| RTX 3090 | llama 8B Q4_0 | 4 | pp1024@d32768 | 206.02 | 189.51 | 197.77 | 0.92 | 0.96 |
| RTX 3090 | llama 8B Q4_0 | 8 | pp1024@d32768 | 356.32 | 324.18 | 341.24 | 0.91 | 0.96 |
| RTX 3090 | llama 8B Q4_0 | 16 | pp1024@d32768 | 677.22 | 597.64 | 637.56 | 0.88 | 0.94 |
| RTX 3090 | llama 8B Q4_0 | 32 | pp1024@d32768 | 1144.09 | 1019.06 | 1075.87 | 0.89 | 0.94 |
| RTX 3090 | llama 8B Q4_0 | 64 | pp1024@d32768 | 1676.86 | 1525.82 | 1596.35 | 0.91 | 0.95 |
| RTX 3090 | llama 8B Q4_0 | 128 | pp1024@d32768 | 1667.77 | 1496.94 | 1580.48 | 0.90 | 0.95 |
| RTX 3090 | llama 8B Q4_0 | 256 | pp1024@d32768 | 2187.18 | 2080.73 | 2119.05 | 0.95 | 0.97 |
| RTX 3090 | llama 8B Q4_0 | 512 | pp1024@d32768 | 2329.90 | 2221.19 | 2262.36 | 0.95 | 0.97 |
| RTX 3090 | qwen35 27B Q4_0 | 1 | pp1024@d32768 | 39.70 | 39.70 | 39.71 | 1.00 | 1.00 |
| RTX 3090 | qwen35 27B Q4_0 | 2 | pp1024@d32768 | 71.67 | 68.52 | 70.24 | 0.96 | 0.98 |
| RTX 3090 | qwen35 27B Q4_0 | 4 | pp1024@d32768 | 126.14 | 120.90 | 123.54 | 0.96 | 0.98 |
| RTX 3090 | qwen35 27B Q4_0 | 8 | pp1024@d32768 | 187.33 | 178.62 | 182.98 | 0.95 | 0.98 |
| RTX 3090 | qwen35 27B Q4_0 | 16 | pp1024@d32768 | 380.08 | 362.29 | 370.68 | 0.95 | 0.98 |
| RTX 3090 | qwen35 27B Q4_0 | 32 | pp1024@d32768 | 580.18 | 557.13 | 567.72 | 0.96 | 0.98 |
| RTX 3090 | qwen35 27B Q4_0 | 64 | pp1024@d32768 | 838.75 | 809.85 | 821.99 | 0.97 | 0.98 |
| RTX 3090 | qwen35 27B Q4_0 | 128 | pp1024@d32768 | 911.52 | 873.50 | 891.92 | 0.96 | 0.98 |
| RTX 3090 | qwen35 27B Q4_0 | 256 | pp1024@d32768 | 1081.22 | 1065.52 | 1073.65 | 0.99 | 0.99 |
| RTX 3090 | qwen35 27B Q4_0 | 512 | pp1024@d32768 | 1124.31 | 1110.68 | 1118.18 | 0.99 | 0.99 |
| RTX 5070 Laptop | gemma 2B Q4_0 | 1 | pp1024@d32768 | 58.59 | 58.59 | 58.59 | 1.00 | 1.00 |
| RTX 5070 Laptop | gemma 2B Q4_0 | 2 | pp1024@d32768 | 97.71 | 97.70 | 97.70 | 1.00 | 1.00 |
| RTX 5070 Laptop | gemma 2B Q4_0 | 4 | pp1024@d32768 | 599.07 | 580.67 | 598.78 | 0.97 | 1.00 |
| RTX 5070 Laptop | gemma 2B Q4_0 | 8 | pp1024@d32768 | 1009.84 | 970.49 | 1010.46 | 0.96 | 1.00 |
| RTX 5070 Laptop | gemma 2B Q4_0 | 16 | pp1024@d32768 | 1842.51 | 1776.67 | 1841.27 | 0.96 | 1.00 |
| RTX 5070 Laptop | gemma 2B Q4_0 | 32 | pp1024@d32768 | 2865.38 | 2774.60 | 2866.91 | 0.97 | 1.00 |
| RTX 5070 Laptop | gemma 2B Q4_0 | 64 | pp1024@d32768 | 3911.53 | 3809.32 | 3906.94 | 0.97 | 1.00 |
| RTX 5070 Laptop | gemma 2B Q4_0 | 128 | pp1024@d32768 | 4522.53 | 4378.16 | 4514.99 | 0.97 | 1.00 |
| RTX 5070 Laptop | gemma 2B Q4_0 | 256 | pp1024@d32768 | 5075.71 | 4980.67 | 5082.10 | 0.98 | 1.00 |
| RTX 5070 Laptop | gemma 2B Q4_0 | 512 | pp1024@d32768 | 5276.02 | 5206.03 | 5281.84 | 0.99 | 1.00 |
| RTX 5070 Laptop | llama 8B Q4_0 | 1 | pp1024@d32768 | 18.56 | 18.56 | 18.56 | 1.00 | 1.00 |
| RTX 5070 Laptop | llama 8B Q4_0 | 2 | pp1024@d32768 | 34.11 | 34.11 | 34.11 | 1.00 | 1.00 |
| RTX 5070 Laptop | llama 8B Q4_0 | 4 | pp1024@d32768 | 98.57 | 117.25 | 102.59 | 1.19 | 1.04 |
| RTX 5070 Laptop | llama 8B Q4_0 | 8 | pp1024@d32768 | 185.43 | 212.65 | 191.50 | 1.15 | 1.03 |
| RTX 5070 Laptop | llama 8B Q4_0 | 16 | pp1024@d32768 | 358.59 | 389.35 | 363.21 | 1.09 | 1.01 |
| RTX 5070 Laptop | llama 8B Q4_0 | 32 | pp1024@d32768 | 544.52 | 624.03 | 555.63 | 1.15 | 1.02 |
| RTX 5070 Laptop | llama 8B Q4_0 | 64 | pp1024@d32768 | 722.03 | 937.35 | 748.98 | 1.30 | 1.04 |
| RTX 5070 Laptop | llama 8B Q4_0 | 128 | pp1024@d32768 | 847.18 | 1222.62 | 869.77 | 1.44 | 1.03 |
| RTX 5070 Laptop | llama 8B Q4_0 | 256 | pp1024@d32768 | 921.87 | 1408.89 | 931.95 | 1.53 | 1.01 |
| RTX 5070 Laptop | llama 8B Q4_0 | 512 | pp1024@d32768 | 952.30 | 1509.11 | 967.22 | 1.58 | 1.02 |

### KV f16/f16, pp1024@d16384

| GPU | Model | Microbatch size | Test | t/s master | t/s PR | Speedup |
|:----|:------|--------:|:-----|-----------:|-----------:|--------:|
| RTX 3090 | llama 8B Q4_0 | 4 | pp1024@d16384 | 442.93 | 442.66 | 1.00 |
| RTX 3090 | llama 8B Q4_0 | 64 | pp1024@d16384 | 2735.06 | 2738.92 | 1.00 |
| RTX 3090 | llama 8B Q4_0 | 512 | pp1024@d16384 | 3467.34 | 3470.25 | 1.00 |
| RTX 5070 Laptop | llama 8B Q4_0 | 4 | pp1024@d16384 | 224.19 | 224.16 | 1.00 |
| RTX 5070 Laptop | llama 8B Q4_0 | 64 | pp1024@d16384 | 1343.68 | 1340.98 | 1.00 |
| RTX 5070 Laptop | llama 8B Q4_0 | 512 | pp1024@d16384 | 1578.46 | 1579.35 | 1.00 |

### KV f16/f16, pp1024@d32768

| GPU | Model | Microbatch size | Test | t/s master | t/s PR | Speedup |
|:----|:------|--------:|:-----|-----------:|-----------:|--------:|
| RTX 3090 | deepseek2 16B Q4_0 | 4 | pp1024@d32768 | 464.92 | 464.49 | 1.00 |
| RTX 3090 | deepseek2 16B Q4_0 | 64 | pp1024@d32768 | 1205.49 | 1204.97 | 1.00 |
| RTX 3090 | deepseek2 16B Q4_0 | 512 | pp1024@d32768 | 1401.77 | 1401.71 | 1.00 |
| RTX 3090 | gemma 2B Q4_0 | 4 | pp1024@d32768 | 1155.05 | 1155.65 | 1.00 |
| RTX 3090 | gemma 2B Q4_0 | 64 | pp1024@d32768 | 6181.81 | 6174.76 | 1.00 |
| RTX 3090 | gemma 2B Q4_0 | 512 | pp1024@d32768 | 8460.54 | 8442.19 | 1.00 |
| RTX 3090 | qwen35 27B Q4_0 | 4 | pp1024@d32768 | 145.40 | 145.44 | 1.00 |
| RTX 3090 | qwen35 27B Q4_0 | 64 | pp1024@d32768 | 888.24 | 888.53 | 1.00 |
| RTX 3090 | qwen35 27B Q4_0 | 512 | pp1024@d32768 | 1137.27 | 1137.26 | 1.00 |
| RTX 5070 Laptop | gemma 2B Q4_0 | 4 | pp1024@d32768 | 732.16 | 731.91 | 1.00 |
| RTX 5070 Laptop | gemma 2B Q4_0 | 64 | pp1024@d32768 | 4504.09 | 4514.49 | 1.00 |
| RTX 5070 Laptop | gemma 2B Q4_0 | 512 | pp1024@d32768 | 5425.47 | 5420.52 | 1.00 |

### KV f16/f16, tg128 depth ladder

| GPU | Model | Depth | Test | t/s master | t/s PR | Speedup |
|:----|:------|--------:|:-----|-----------:|-----------:|--------:|
| RTX 3090 | deepseek2 16B Q4_0 | 32768 | tg128@d32768 | 192.26 | 192.07 | 1.00 |
| RTX 3090 | gemma 2B Q4_0 | 32768 | tg128@d32768 | 293.26 | 293.25 | 1.00 |
| RTX 3090 | llama 8B Q4_0 | 16384 | tg128@d16384 | 115.17 | 115.15 | 1.00 |
| RTX 3090 | qwen35 27B Q4_0 | 32768 | tg128@d32768 | 42.26 | 42.26 | 1.00 |
| RTX 5070 Laptop | gemma 2B Q4_0 | 32768 | tg128@d32768 | 153.45 | 153.30 | 1.00 |
| RTX 5070 Laptop | llama 8B Q4_0 | 16384 | tg128@d16384 | 53.32 | 53.30 | 1.00 |

</details>

### Correctness

- `test-backend-ops test -o FLASH_ATTN_EXT -b CUDA0` (GGML_CUDA_FA_ALL_QUANTS
  build) at tip `54b94d245` vs master `36b101543`, RTX 3090: 3969/3969 at the
  default budget, 3969/3969 at GGML_CUDA_FATTN_CONVERT_BYTES=1048576 and
  3969/3969 at =262144 (forces multi-chunk stream-k execution including
  both fixup variants); master 3955/3955. RTX 5070 Laptop: 3969/3969 /
  3969/3969 / 3969/3969 vs master 3955/3955. The delta of 14 is exactly
  the cases this PR adds.
- Full `test-backend-ops test -b CUDA0` (same build): RTX 3090 15644/15644 vs
  15630/15630; RTX 5070 Laptop 15644/15644 vs 15630/15630.
- compute-sanitizer memcheck over the entire FLASH_ATTN_EXT suite at the tip:
  RTX 3090 ERROR SUMMARY: 0 errors (exit 0); RTX 5070 Laptop ERROR SUMMARY: 0 errors (exit 0).
- Artifacts with environment + exact commands, produced by `benchmark.sh
  --tests` in the .devops-derived toolchain image (both GPUs 2026-09-25):
  [VALIDATE-RTX3090-sm86-24GB.txt](https://github.com/Abai/llama.cpp/blob/benchmark/cuda-fattn-convert-buffer-cap/bench-out/jg-multi/cuda-fattn-convert-buffer-cap/VALIDATE-RTX3090-sm86-24GB.txt),
  [VALIDATE-RTX5070LaptopGPU-sm120-8GB.txt](https://github.com/Abai/llama.cpp/blob/benchmark/cuda-fattn-convert-buffer-cap/bench-out/jg-multi/cuda-fattn-convert-buffer-cap/VALIDATE-RTX5070LaptopGPU-sm120-8GB.txt).
- The 14 cases, added to the existing test_flash_attn_ext: 6 default-budget
  cases + 8 chunk-shaped small-kv cases. The plain suite exercises chunking
  on CUDA with default 64 MiB cap.
- wikitext-2 perplexity and KL divergence, PR vs master `36b101543`, uniform
  Q4_0 weights, `benchmark.sh --ppl --kld` (renders:
  [PPL-RTX3090-sm86-24GB.md](https://github.com/Abai/llama.cpp/blob/benchmark/cuda-fattn-convert-buffer-cap/bench-out/jg-multi/cuda-fattn-convert-buffer-cap/PPL-RTX3090-sm86-24GB.md)).
  Every statistic is bit-identical to master when not capped. When capped,
  llama and qwen at n_ctx 32768 on the RTX 3090, |delta| <= 0.011: chunk maxima
  differ, causing a rescale during accumulation.

### Perplexity: wikitext-2 test, n_ctx=8192, 9 chunks (73728 tokens)

| GPU | Model | KV | PPL master | PPL PR | PR - master |
|:----|:------|:---|-----------:|-------:|------------:|
| RTX 3090 | gemma 2B Q4_0 | f16/f16 | 437.5897 +/- 10.30099 | 437.5897 +/- 10.30099 | +0.0000 |
| RTX 3090 | gemma 2B Q4_0 | q4_0/q4_0 | 439.2269 +/- 10.33557 | 439.2269 +/- 10.33557 | +0.0000 |
| RTX 3090 | gemma 2B Q4_0 | q8_0/q8_0 | 437.2399 +/- 10.27944 | 437.2399 +/- 10.27944 | +0.0000 |
| RTX 5070 Laptop | gemma 2B Q4_0 | f16/f16 | 438.2826 +/- 10.32349 | 438.2826 +/- 10.32349 | +0.0000 |
| RTX 5070 Laptop | gemma 2B Q4_0 | q4_0/q4_0 | 441.9798 +/- 10.39866 | 441.9798 +/- 10.39866 | +0.0000 |
| RTX 5070 Laptop | gemma 2B Q4_0 | q8_0/q8_0 | 438.1430 +/- 10.30890 | 438.1430 +/- 10.30890 | +0.0000 |

### Perplexity: wikitext-2 test, n_ctx=16384, 9 chunks (147456 tokens)

| GPU | Model | KV | PPL master | PPL PR | PR - master |
|:----|:------|:---|-----------:|-------:|------------:|
| RTX 5070 Laptop | llama 8B Q4_0 | f16/f16 | 5.7306 +/- 0.04682 | 5.7306 +/- 0.04682 | +0.0000 |
| RTX 5070 Laptop | llama 8B Q4_0 | q4_0/q4_0 | 5.8087 +/- 0.04733 | 5.8087 +/- 0.04733 | +0.0000 |
| RTX 5070 Laptop | llama 8B Q4_0 | q8_0/q8_0 | 5.7303 +/- 0.04684 | 5.7303 +/- 0.04684 | +0.0000 |

### Perplexity: wikitext-2 test, n_ctx=32768, 9 chunks (294912 tokens)

| GPU | Model | KV | PPL master | PPL PR | PR - master |
|:----|:------|:---|-----------:|-------:|------------:|
| RTX 3090 | deepseek2 16B Q4_0 | f16/f16 | 6.1287 +/- 0.03662 | 6.1287 +/- 0.03662 | +0.0000 |
| RTX 3090 | deepseek2 16B Q4_0 | q4_0/q4_0 | 6.8641 +/- 0.04108 | 6.8641 +/- 0.04108 | +0.0000 |
| RTX 3090 | deepseek2 16B Q4_0 | q8_0/q8_0 | 6.1297 +/- 0.03663 | 6.1297 +/- 0.03663 | +0.0000 |
| RTX 3090 | llama 8B Q4_0 | f16/f16 | 6.9836 +/- 0.04504 | 6.9836 +/- 0.04504 | +0.0000 |
| RTX 3090 | llama 8B Q4_0 | q4_0/q4_0 | 7.0886 +/- 0.04559 | 7.0857 +/- 0.04556 | -0.0029 |
| RTX 3090 | llama 8B Q4_0 | q8_0/q8_0 | 6.9818 +/- 0.04510 | 6.9834 +/- 0.04512 | +0.0016 |
| RTX 3090 | qwen35 27B Q4_0 | f16/f16 | 6.3757 +/- 0.04159 | 6.3757 +/- 0.04159 | +0.0000 |
| RTX 3090 | qwen35 27B Q4_0 | q4_0/q4_0 | 6.3941 +/- 0.04180 | 6.3903 +/- 0.04179 | -0.0038 |
| RTX 3090 | qwen35 27B Q4_0 | q8_0/q8_0 | 6.3813 +/- 0.04169 | 6.3703 +/- 0.04155 | -0.0110 |

### KL divergence vs the master f16-KV run: wikitext-2 test, n_ctx=4096, 4 chunks (16384 tokens)

| GPU | Model | KV | side | PPL | mean KLD | median KLD | 99% KLD | max KLD | same top-1 % | mean ln(PPL/PPL_base) |
|:----|:------|:---|:-----|----:|---------:|-----------:|--------:|--------:|-------------:|----------------------:|
| RTX 3090 | llama 8B Q4_0 | q4_0/q4_0 | master | - | 0.012990 +/- 0.000308 | 0.006985 | 0.114697 | 0.919942 | 95.115 +/- 0.238 | 0.011960 +/- 0.001937 |
| RTX 3090 | llama 8B Q4_0 | q4_0/q4_0 | PR | - | 0.012990 +/- 0.000308 | 0.006985 | 0.114697 | 0.919942 | 95.115 +/- 0.238 | 0.011960 +/- 0.001937 |
| RTX 3090 | llama 8B Q4_0 | q8_0/q8_0 | master | - | 0.000574 +/- 0.000012 | 0.000329 | 0.004628 | 0.038513 | 98.852 +/- 0.118 | 0.000394 +/- 0.000443 |
| RTX 3090 | llama 8B Q4_0 | q8_0/q8_0 | PR | - | 0.000574 +/- 0.000012 | 0.000329 | 0.004628 | 0.038513 | 98.852 +/- 0.118 | 0.000394 +/- 0.000443 |
| RTX 5070 Laptop | llama 8B Q4_0 | q4_0/q4_0 | master | - | 0.012769 +/- 0.000313 | 0.006725 | 0.103546 | 0.790731 | 94.968 +/- 0.242 | 0.011886 +/- 0.001905 |
| RTX 5070 Laptop | llama 8B Q4_0 | q4_0/q4_0 | PR | - | 0.012769 +/- 0.000313 | 0.006725 | 0.103546 | 0.790731 | 94.968 +/- 0.242 | 0.011886 +/- 0.001905 |
| RTX 5070 Laptop | llama 8B Q4_0 | q8_0/q8_0 | master | - | 0.000573 +/- 0.000012 | 0.000334 | 0.004850 | 0.027627 | 98.876 +/- 0.116 | -0.000170 +/- 0.000446 |
| RTX 5070 Laptop | llama 8B Q4_0 | q8_0/q8_0 | PR | - | 0.000573 +/- 0.000012 | 0.000334 | 0.004850 | 0.027627 | 98.876 +/- 0.116 | -0.000170 +/- 0.000446 |

## Requirements

- I have read and agree with the [contributing guidelines](https://github.com/ggml-org/llama.cpp/blob/master/CONTRIBUTING.md)
- AI usage disclosure: YES - code written with Fable 5, manually reviewed by me

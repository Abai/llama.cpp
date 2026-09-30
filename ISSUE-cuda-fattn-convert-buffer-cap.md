## Title

Feature Request: CUDA: reduce VRAM usage of FA convert buffer for
quantized KV caches

## Feature Description

The fattn-mma-f16 and fattn-tile kernels ingest f16 KV data layer by layer. For
non-f16 (quantized) KV types this requires a conversion buffer, pre-allocated
per #23907. The size of this buffer is equal to one layer of f16 KV and scales
with ctx length. For quantized KV types, this negates large part of VRAM savings
obtained by quantization in the first place. For example, for qwen35 type
models, like Qwen3.6-27B, at ctx 131072, this negates 13.3% of the VRAM savings
at q8_0 and 8.7% at q4_0 KV. This buffer needs to be be made ctx length
independent or eliminated entirely consuming the quantized KV directly in the
fattn kernels similar to fattn-vec kernel.

## Motivation

- The buffer negates a large part of VRAM savings that KV quantization provides
  at long ctx. Reducing it, arguably even at a small throughput cost, would be
  favorable when enabling KV quantization.
- Lowering ubatch size does not influence the size of this buffer. For least
  VRAM used at the cost of prefill throughput you would combine low ub + this
  feature
- qwen35_q4_0 KV q8_0 and ctx length from ~229k to ~254k on 24 GB (no mmproj)
- Reported previously in #24166, #24135. Related to #23635, #19979, #23907.

## Possible Implementation

**Solution A**: Implement serial split-KV - Cap the buffer to a fixed size
(64 MiB, env override available for testing only), process KV in chunks re-using
the buffer. For each processed KV chunk fattn kernel emits an unnormalized
partial and (max_row_val, row_sum), these get folded into running accumulators
using the existing combine kernel, flash_attn_combine_results in fattn-common.

Pros:
* Cap FA convert buffer to fixed size (64 MiB), ctx length independent
* Simple. Easy to implement/remove.
* Supports both fattn-mma-f16 and fattn-tile kernel
* Small testing surface. Minimal changes to fattn kernels.
* Bit identical output for f32, f16, KV pairs that fit below cap and fattn-vec
  paths
* Can yield a large prefill gain with large ctx and large ubatch, when memory
  bound, by reducing VRAM pressure if chunk fits into L2 cache. Tested on RTX
  5070 Laptop, 32 MiB L2, cap of 32 MiB, yields 1.59x agains master. Needs
  further testing to see if cap size should be a cli parameter.

Cons:
* convert, FA and combine launches per KV chunk instead of per KV layer. Prefill
  cost when chunking long ctx at mid to small ubatch.

| Solution A, measured | worst .. best |
|---|---|
| Throughput | -10% (RTX 3090, small-mid ub, deep ctx) .. +59% (RTX 5070 Laptop, 32 MiB cap via env, chunk fits L2) |
| VRAM gained | 36 MiB (gemma) .. 448 MiB (qwen35, q8_0 KV) at ctx 131072 |

Details in #29827. Measured data on the fork: [throughput, both GPUs](https://github.com/Abai/llama.cpp/blob/benchmark/cuda-fattn-convert-buffer-cap/bench-out/jg-multi/cuda-fattn-convert-buffer-cap/TABLES-RTX3090-sm86-24GB.md) | [VRAM](https://github.com/Abai/llama.cpp/blob/benchmark/cuda-fattn-convert-buffer-cap/bench-out/jg-multi/cuda-fattn-convert-buffer-cap/TABLES-VRAM-RTX3090-sm86-24GB.md)

**Solution B**: Dequantize KV in smem in the MMA kernel, stage through cp.async
for quantized KV only (initially for q8_0xq8_0, q4_0xq4_0, q8_0xq4_0).

Pros:
* Eliminates the KV convert buffer entirely
* Prefill gain at small ubatch
* Bit identical output for f32, f16 and fattn-vec paths

Cons:
* Complex. Large changes to the fattn-mma-f16 kernel.
* Supports fattn-mma-f16 only. No cp.async for tile.
* Large testing surface.
* Affects ggml-cuda build time and binary size due to templating the
  fattn-mma-f16 kernel on K and V cache types

| Solution B, measured | worst .. best |
|---|---|
| Throughput | -2% (qwen35, large ub) .. +129% (gemma, small ub, deep ctx) (RTX 3090) |
| VRAM gained | 40 MiB (gemma) .. 470 MiB (qwen35) at ctx 131072; MLA (not fused) 0 |

Details in #29846. Measured data on the fork:
[throughput, both GPUs](https://github.com/Abai/llama.cpp/blob/benchmark/cuda-fattn-mma-fused-dequant/bench-out/jg-multi/cuda-fattn-mma-fused-dequant/TABLES-RTX3090-sm86-24GB.md) | [VRAM](https://github.com/Abai/llama.cpp/blob/benchmark/cuda-fattn-mma-fused-dequant/bench-out/jg-multi/cuda-fattn-mma-fused-dequant/TABLES-VRAM-RTX3090-sm86-24GB.md) | [write-up](https://github.com/Abai/llama.cpp/blob/benchmark/cuda-fattn-mma-fused-dequant/PR-BODY-fattn-fused-dequant.md)

**Solution A and B can be implemented sequntially** since A also works for
fattn-tile, while B eliminates the buffer entirely for mma path quantized KV

(All measurements vs master 36b101543: RTX 3090 and RTX 5070 Laptop
(sm_120), llama-server for VRAM, llama-bench for throughput, containerized
builds/runs using .devops/cuda.Dockerfile. See PRs for the full benchmark
tables.)

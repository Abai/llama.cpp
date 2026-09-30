CUDA: fuse KV cache dequantization into fattn-mma-f16

Allow fattn-mma-f16 kernel to consume quantized KV data directly, similar to
fattn-vec kernel. This eliminates the FA convert buffer for quantized KV caches,
entirely. Currently enabled for the KV type pairs q8_0 x q8_0, q4_0 x q4_0, and
with GGML_CUDA_FA_ALL_QUANTS q8_0 x q4_0, at head sizes 128 and 256.

At n_ctx 131072 with q8_0 x q8_0 KV and ub 512, the CUDA0 compute buffer drops
470 MiB for qwen35 27B and 444 MiB for llama 8B on an RTX 3090.

The kernel is templated over K and V types and loads its tiles through
fattn-mma-load.cuh. Loaders for supported quantized KV types stage tiles as
raw bytes with cp.async and dequantize them into the swizzled (#25635) smem
layout. The conversion is bit-identical to the master's full KV layer to f16
conversion path.

At large ubatch the fused in-kernel dequantization of q8_0 and q4_0 KV-tiles
had a measured throughput regression (q8_0 ub64/ub512 = 1.011/0.963). To counter
it, on shapes that use multi-stage cp.async pipeline, instead of dequantizing K
to f16 for KQ, Q is quantized to q8 blocks in-kernel and K is deinterleaved to
int8, allowing KQ to run on int8 tensor cores. KQ then differs numerically,
with wikitext-2 perplexity similar to master.

On Blackwell sm_120 gemma q8_0/q8_0 with head 256, hits the 255-register ceiling
resulting in 384 B of spill and worst case -23% regression vs master. Retuned
sm_120 to occupancy 1 with nthreads 256 for this case, resulting in -5% worst
regression. Other architectures keep the shared config, as this shape does not
regress there.

f32 KV, f16 KV and fattn-vec paths are unaffected. The f16 instances of
fattn-mma-f16 are verified to produce SASS identical .cu.o objects.

Assisted-by: Claude Fable 5


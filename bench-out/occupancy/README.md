# Does chunking cost occupancy, or only extra launches? (2026-09-29)

Re-measurement of the wave-quantization claim. The 2026-08-24 original was never
committed, so this replaces it with evidence that is kept.

## Method

One binary (cap-src/build-cap at 54b94d245), one workload, one variable: the
GGML_CUDA_FATTN_CONVERT_BYTES knob decides whether the f16 conversion is split.

  single  = 1 GiB cap  -> the 129 MiB conversion fits, one FA launch per op
  chunked = 64 MiB cap -> the same conversion is split, several launches per op

llama 3.1 8B Q4_0, KV q8_0/q8_0, -fa 1 -ngl 99 -p 256 -d 32768 -ub 8, RTX 3090,
ncu 2025.1.1, GGML_CUDA_DISABLE_GRAPHS=1 so each launch is captured separately.
probe-occupancy.sh is the runner; ncu-single.csv / ncu-chunked.csv are the raw
exports.

## The knob really is the only difference

Same binary, pp1024@d32768, -r 2, no profiler:

| cap | t/s | dataset reference |
|---|---|---|
| 1 GiB (no chunking) | 347.93 | master 347.35, 0.17% apart |
| 64 MiB (chunking) | 332.23 | PR at 64 MiB 331.86, 0.11% apart |

Both ends reproduce the committed campaign values, so chunking genuinely engages
at 64 MiB and genuinely does not at 1 GiB. The 1 GiB end also lands on master's
throughput, which is an independent check on the SASS gate's claim that the
non-chunked path executes master's instructions.

## Result: launch geometry is invariant

| kernel | mode | grid | block | waves/SM | SM % |
|---|---|---|---|---|---|
| flash_attn_ext_f16<128,128,8,4> | single | 32 | 128 | 0.20 | 3.5 |
| flash_attn_ext_f16<128,128,8,4> | chunked | 32 | 128 | 0.20 | 3.6 |
| flash_attn_stream_k_fixup_uniform<128,8> | single | 256 | 128 | 0.26 | 9.1 |
| flash_attn_stream_k_fixup_uniform<128,8> | chunked | 256 | 128 | 0.26 | 9.3 |

Identical grids, identical block sizes, identical waves per multiprocessor. The
stream-k decomposition sizes its grid from the output shape and the SM count, not
from the KV length, so splitting the KV does not change how the work is spread
across the GPU - it repeats the same launch. The ~4.5% cost measured above is
therefore per-launch fixed work multiplied by the chunk count (conversion, fixup
and combine), not a loss of occupancy.

Waves per multiprocessor below 1 is expected at this shape: ub 8 gives few query
rows, so a single wave does not fill the GPU either way. That is a property of
the microbatch, present identically on master, not something chunking introduces.

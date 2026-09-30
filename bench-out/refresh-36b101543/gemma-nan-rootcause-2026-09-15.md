# gemma-2b NaN with fused-dequant MMA FlashAttention: root cause and fix (2026-09-15)

Feature commit: c18fac264 (feature/cuda-fattn-mma-fused-dequant, one commit on master 36b101543).
Rig: RTX 3090 (sm_86), image llamacpp-bench-dev:12.8.1-gcc14, CUDA 12.8.1, gcc-14, Release,
GGML_CUDA_FA_ALL_QUANTS=ON. Builds used: build-cap (feature, read-only), build-master (read-only),
nanprobe-build (worktree nanprobe-feat at c18fac264 + probe test cases + fix; removed at the end).

## 0. Correction of the premise

The task description says gemma-2b is "8 heads, 8 KV heads = MHA, GQA ratio 1". The model file
(converted/gemma-2b-Q4_0.gguf) is actually MQA:

    print_info: n_head = 8, n_head_kv = 1, n_gqa = 8, n_embd_head_k = n_embd_head_v = 256, n_ctx_train = 8192

So gemma-2b takes the gqa_ratio > 4 -> ncols2 = 8 dispatch path, not the ncols2 = 1 path. This
matters: ncols2 == 1 forces nstages = 0 (synchronous loader, no int8 KQ path), whereas ncols2 = 8
uses the 2-stage cp.async pipeline plus the int8-tensor-core KQ path (imma_KQ) for ncols >= 16.
Both the ncols2 == 1 synchronous path and the ncols2 == 8 path were probed; only the latter fails.

## 1. Reproduction matrix

llama-perplexity, gemma-2b-Q4_0.gguf, wikitext-2 test, -fa 1 -ngl 99, build-cap (feature, unfixed).
All at n_ctx 512, batch 512, 1 chunk (the failure does not need a long KV: it is present at kv <= 512).

| KV type | ubatch | Q->ne[1] | instance (DKQ, ncols1, ncols2) | kernel path | result [1] |
| ------- | ------ | -------- | ------------------------------ | ----------- | ---------- |
| q8_0 | 1   | 1   | vec kernel (Ampere: quantized KV, ne1 == 1)          | fattn-vec               | 6.2167 |
| q8_0 | 2   | 2   | (256, 2, 8),  ncols 16, nwarps 2, np 2               | 2-stage, imma_KQ        | 6.1978 |
| q8_0 | 4   | 4   | (256, 4, 8),  ncols 32, nwarps 4, np 2               | 2-stage, imma_KQ        | 6.1978 |
| q8_0 | 8   | 8   | (256, 8, 8),  ncols 64, nwarps 4, np 1               | 2-stage, imma_KQ        | nan    |
| q8_0 | 16..512 | 16..512 | (256, 8, 8) (ncols1 = 64/ncols2 = 8 for ne1 > 4) | 2-stage, imma_KQ     | nan (all of 16, 32, 64, 128, 256, 512) |
| q4_0 | 1   | 1   | vec kernel                                            | fattn-vec               | 6.3462 |
| q4_0 | 2   | 2   | (256, 2, 8)                                           | 2-stage, imma_KQ        | 6.4614 |
| q4_0 | 4   | 4   | (256, 4, 8)                                           | 2-stage, imma_KQ        | 6.4614 |
| q4_0 | 8..512 | 8..512 | (256, 8, 8)                                     | 2-stage, imma_KQ        | nan (all of 8, 16, 32, 64, 128, 256, 512) |
| f16  | 512 | 512 | (256, 8, 8) f16 instance                              | 2-stage, f16 KQ         | 6.3010 (c=512, chunk 1) |

Only the instance DKQ = 256, ncols1 = 8, ncols2 = 8 (ncols = 64) fails. The instance selection for
ncols2 = 8 on Ampere (ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1): ne1 <= 1 -> ncols1 1,
ne1 <= 2 -> 2, ne1 <= 4 -> 4, else 8. Ampere config for (256, 256, ncols 64): nthreads 128,
nbatch_fa 32, nbatch_K2 = nbatch_V2 = 128, nstages_target 2, Q_in_reg true.

test-backend-ops probes (nanprobe-build, unfixed feature code, GGML_CUDA_FA_ALL_QUANTS build), added
to the case list of the worktree only:

| probe group | cases | result before fix |
| ----------- | ----- | ----------------- |
| MQA shape: hs in {128, 256}, nh 1, nr23 {8,1}, kv 512, nb in {1,2,4,8,16,64}, q8_0/q4_0, permute {0,1,2,3} and {0,2,1,3} (+ q8_0/q4_0 mixed and f16 at hs 256 nb 8) | 50 | 37/50: FAIL exactly for hs 256, nb in {8,16,64}, both types, both permutes, and the q8_0/q4_0 mixed case (13 cases, NMSE 0.82..0.85) |
| GQA-1 shape (ncols2 == 1, synchronous loader): hs in {128, 256}, nh 8, nr23 {1,1}, kv 512, nb in {1,3,8,16,32,64,75}, q8_0/q4_0, both permutes | 56 | 56/56 OK |
| GQA-4 with no mask / ALiBi (gqa opt off -> ncols2 == 1): hs 256/128, nb 64, q8_0 | 3 | 3/3 OK |

The existing test list never builds this shape: the generic loop only tests quantized KV at hsk 64/72
(`if (type_KV != GGML_TYPE_F16 && hsk != 64 && hsk != 72) continue;`), nr2 == 8 is only used with
hsk 192, and the extra head-256 q8_0 cases use nr2 = 16 with nb = 1 (ncols 8, no imma_KQ) or a
kv of 1025 (gqa opt off -> ncols2 = 1). The prefill loop at test-backend-ops.cpp:10077 uses nr2 = 4
(ncols1 = 16 at nb 64), which does not overlap (see below).

## 2. Localization

The synchronous loader path (nstages 0/1, dequant_chunk/store_tile) is correct (56/56, 3/3).
The 2-stage cp.async loaders and the int8 KQ path are correct for ncols1 = 2, 4 (ncols 16, 32) and for
the ncols2 = 4 instances (ncols1 = 8, 16) covered by the existing tests. The failure is specific to
the instance with ncols = 64 and ncols1 = 8 at DKQ = 256, i.e. it depends on ncols1 and ncols2
separately, not only on their product. The only things in the kernel that depend on ncols1 alone are
the mask tile size and, through it, the position of the staging buffer.

## 3. Root cause: the Q-quantization scratch aliases the tail of tile_Q

Shared memory layout in flash_attn_ext_f16_process_tile (c18fac264, fattn-mma-f16.cuh):

    1267    half2 * tile_K    = Q_in_reg ? tile_Q : tile_Q + ncols*stride_tile_Q;
    1268    half2 * tile_V    = nstages > 1 ? tile_K + nbatch_fa*stride_tile_K : tile_K;
    1269    half  * tile_mask = (half *) (nstages > 1 ? tile_V + nbatch_fa*stride_tile_V : ...);
    1274    char * staging_K = (char *) (tile_mask + ncols1*(nbatch_fa + 8));

With Q_in_reg the K/V/mask/staging region starts at tile_Q: the host allocates only
max(nbytes_shared_Q, nbytes_shared_KV + nbytes_shared_mask) (line 2123-2124), because in the f16
kernel tile_Q is consumed into registers before the first K tile is loaded. The new int8 KQ path
quantizes Q in-kernel and uses the staging buffer as scratch while it is still reading tile_Q:

    1361        char  * const Q_s8   = staging_K;
    1362        float * const dQ_all = (float *) (staging_K + ncols*QK8_0);
    ...
    1381                ggml_cuda_memcpy_1<4*sizeof(half2)>(x + 0, tile_Q + jc*stride_tile_Q + b*(QK8_0/2) + 8*h + 0);   // reads tile_Q
    ...
    1404                ggml_cuda_memcpy_1<16>(Q_s8 + jc*QK8_0 + 16*h, q);                                              // writes scratch

The static_assert at line 1359-1360 only checks that the staging buffer is large enough for the
scratch, not that the scratch does not overlap tile_Q. Byte offsets for the failing instance
(DKQ 256, ncols1 8, ncols2 8, nbatch_fa 32, stride_tile_K = stride_tile_V = 128 (swizzled),
stride_tile_Q = 132):

    tile_Q      : [0, 64*132*4)                                   = [0, 33792)
    tile_K/V    : [0, 16384) and [16384, 32768)
    tile_mask   : [32768, 32768 + 8*(32+8)*2)                     = [32768, 33408)
    staging_K   : starts at 33408 = Q_s8; Q_s8 spans [33408, 35456); dQ_all [35456, 37504)

Q_s8 overlaps tile_Q row 63 from byte 33408 - 63*528 = 144, i.e. from element 72 of the last Q
column. In the block loop (b = 0..7) the lanes with jc <= 11 write their 32 q8 bytes into that range
at every b, so when column 63 is read for its blocks b >= 2 (elements 64..255) it contains int8 data
reinterpreted as f16. Byte patterns 0x7Cxx / 0x7Fxx read as inf / NaN, so amax = inf, dQ = inf, and
the whole KQ column becomes NaN (test-backend-ops sees NMSE ~0.83 because its inputs are small;
gemma with real activations gets inf -> NaN, then NaN logits through the residual stream and PPL nan
from chunk 1 on).

Column 63 = Q position 7 x head 7 in every 8-position tile, for every layer; that is enough for
NaN logits on most tokens.

General condition for the overlap (Q_in_reg, imma_KQ):

    ncols*(DKQ/2 + 4)*4  >  nbatch_fa*(stride_tile_K + stride_tile_V)*4 + ncols1*(nbatch_fa + 8)*2

For the Ampere/Ada/Hopper/Blackwell config table this holds only for DKQ 256, ncols 64
(33792 > 32768 + 80*ncols1  <=>  ncols1 <= 12), i.e. ncols1 = 8 / ncols2 = 8: head size 256 with
gqa_ratio > 4 and Q->ne[1] > 4. DKQ 128 instances (tile_Q 17408 < 32768), the ncols 16/32 instances
at DKQ 256, ncols2 = 4 (ncols1 = 16: 34048 > 33792) and the Turing configs (nbatch_fa 64) do not
overlap, which is why llama-8B, qwen3.5-27B, deepseek2 and test-backend-ops were clean. gemma-2b
(MQA, head 256) is the only model in the benchmark set that hits it; any head-256 model with
gqa_ratio >= 5 at ubatch > 4 with q8_0/q4_0 KV would.

The dQ_all part of the scratch and the K/V staging use after the quantization are fine (the
`__syncthreads()` at line 1420 orders the scratch reads before the first cp.async K load).

## 4. Fix (diff: gemma-nan-fix-2026-09-15.diff)

Place the Q-quantization scratch directly after tile_Q instead of in the staging buffer, and make the
host reserve that space when imma_KQ is active (max(nbytes_shared_Q + scratch, KV + mask) instead of
max(nbytes_shared_Q, KV + mask)). The 2-stage shared-memory estimate used for the nstages decision
(fattn_mma_staged_nbytes_shared_2stage, host and device) gets the same term so it keeps being an
upper bound of the layout. The scratch is ncols*(QK8_0 + 4*(DKQ/QK8_0)) bytes (4096 for the failing
instance); for every currently instantiated imma_KQ config nbytes_shared_Q + scratch is below
KV + mask + staging (e.g. 37888 vs 42112 for (256, 8, 8) q8_0 and 37888 vs 38016 for (256, 8, 8)
q4_0, the tightest one), so the allocation and the nstages decisions of all instances are unchanged;
only the scratch address moves out of tile_Q.

Kernel: 3 lines changed in process_tile (scratch base = tile_Q + ncols*stride_tile_Q, static_assert
dropped, comment). Host: 3 lines (imma_KQ predicate mirrored from the device, scratch size, max()).
Estimate: 2 lines.

## 5. Verification (nanprobe-build = c18fac264 + fix, same flags as build-cap)

test-backend-ops (GGML_CUDA_FA_ALL_QUANTS build, -o FLASH_ATTN_EXT -b CUDA0):

| build | case set | result |
| ----- | -------- | ------ |
| unfixed feature + probe cases | MQA probe group (50) | 37/50, the 13 hs-256 nr2-8 nb>=8 cases FAIL (NMSE 0.82..0.85) |
| fixed feature + probe cases | full FLASH_ATTN_EXT list (3955 upstream + 109 probes) | 4064/4064 |
| fixed feature + minimal 2-line test addition | `-p 'hsk=256,hsv=256,nh=1,nr23=\[8,1\],kv=512,nb=8,'` | 2/2 OK |

The two added cases have exactly the parameters of two of the 13 failing probe cases
(test_flash_attn_ext(256, 256, 1, {8, 1}, 512, 8, mask, q8_0/q8_0) at NMSE 0.847 and q4_0/q4_0 at
NMSE 0.844 before the fix), so they fail before and pass after.

llama-perplexity, gemma-2b-Q4_0.gguf, wikitext-2 test, -c 8192 -b 2048 -ub 512 -fa 1 -ngl 99,
35 chunks, identical command line on both builds (the 437.x reference values quoted in the task and
the PR notes come from the benchmark script with other settings; the same-command master numbers
below are the valid reference):

| KV type | master 36b101543 (build-master) | feature c18fac264 (build-cap) | feature + fix (nanprobe-build) |
| ------- | ------------------------------- | ----------------------------- | ------------------------------ |
| f16  | 427.3139 +/- 5.15 | 427.3139 (identical, f16 kernels unchanged) | 427.3139 +/- 5.15 (identical) |
| q8_0 | 426.2260 +/- 5.14 | nan from chunk 1 | 425.5855 +/- 5.13 |
| q4_0 | 422.0538 +/- 5.06 | nan from chunk 1 | 419.1250 +/- 5.02 |

At -c 512 (20 chunks, the normal PPL regime of this model):

| KV type | master (build-master) | feature + fix (nanprobe-build) |
| ------- | --------------------- | ------------------------------ |
| q8_0 | 10.5951 +/- 0.398 | 10.6046 +/- 0.398 |
| q4_0 | 10.6719 +/- 0.400 | 10.6330 +/- 0.398 |

Control for the residual drift: the same fixed build with the int8 KQ path disabled
(fattn_mma_imma_kq_available = false, so the fused instances use the f16-dequant KQ path that is
documented as bit-identical to master's convert-to-f16 path) reproduces master exactly at 8192:
q8_0 426.2260 +/- 5.13545 and q4_0 422.0538 +/- 5.05829, identical to the last digit. The
remaining difference with the int8 KQ path on (-0.15 % q8_0, -0.7 % q4_0 at 8192; +0.09 % / -0.36 %
at 512, all well inside one standard error) is therefore only the designed in-kernel q8 rounding of
Q, the same effect the PR notes measured on llama/qwen (KLD 0.000574 -> 0.000624 for q8_0), and not
a second defect. It is larger in relative terms for gemma-2b at 8192 because the model is in a
degenerate regime there (PPL 420+, vs 10.6 at 512).

Not re-run (out of scope, unchanged code paths): the other three models, sm_70/sm_75 builds,
compute-sanitizer. The fix does not change any allocation size or nstages decision of an existing
instance (section 4), so their behaviour is unchanged by construction; the f16 instances are not
touched (imma_KQ is compile-time false for them, nbytes_shared_Q_scratch = 0).



## 6. Alternative (not implemented)

The scratch (and the 2*(DKQ/32) __syncthreads per tile) could be removed entirely by quantizing Q
straight into the tile<16, 8, int> A fragments: each lane holds rows lane/4 and lane/4 + 8 of the
16-column slab at k = 4*(lane%4) (+16), which are exactly the C-tile rows whose scales dQ it needs;
the block amax is a 2-step __shfl_xor over the quad. That is a larger rewrite of the quantization
block and was out of scope for a minimal fix.

## 7. Files

- bench-out/refresh-36b101543/gemma-nan-fix-2026-09-15.diff: the fix against c18fac264
  (fattn-mma-f16.cuh 19 lines, fattn-mma-load.cuh 4 lines, test-backend-ops.cpp +3 lines).
- bench-out/refresh-36b101543/gemma-nan-2026-09-15-logs/: perplexity logs (matrix at 512, 8192
  master/fixed/IMMA-off, 512 master/fixed), test-backend-ops before-fix probe log and after-fix
  summary.
- Worktree nanprobe-feat, build dir nanprobe-build and the nanprobe-*.sh/.log helpers at the repo
  root were removed. The main working tree and all branches are untouched.

## 8. Summary (10 lines)

1. gemma-2b is MQA (8 Q heads, 1 KV head, head 256), not MHA: it uses ncols2 = 8, the 2-stage cp.async pipeline and the int8 KQ path.
2. NaN reproduces at n_ctx 512 already; it needs ubatch >= 8 (instance DKQ 256, ncols1 8, ncols2 8); ubatch 1/2/4 and the ncols2 == 1 synchronous path are correct.
3. Root cause: with Q_in_reg the host allocates max(Q tile, K/V + mask + staging) and the K/V/mask/staging region aliases tile_Q.
4. The in-kernel Q quantization (imma_KQ) uses the staging buffer as scratch while tile_Q is still being read; for this instance staging starts at byte 33408 inside tile_Q (33792 bytes).
5. The q8 bytes of columns 0..11 overwrite elements 72..255 of Q column 63; read back as f16 they give inf/NaN scales -> NaN KQ column -> NaN logits.
6. Only DKQ 256 with ncols1 8 / ncols2 8 satisfies the overlap condition on the Ampere+ config table, so llama/qwen/deepseek and the existing tests (no head-256 nr2-8 quantized case) never hit it.
7. Fix: put the scratch directly after tile_Q and reserve it on the host (max(Q + scratch, KV + mask)); the 2-stage smem estimate gets the same term. No existing allocation or nstages decision changes.
8. Verified: FLASH_ATTN_EXT 4064/4064 (3955 upstream + 109 probes) after the fix vs 13 failures before; the 2-line test addition fails before (NMSE 0.85) and passes after.
9. gemma 8192 PPL after the fix: q8_0 425.59, q4_0 419.13, f16 427.31 vs master 426.23 / 422.05 / 427.31 (same command); with the int8 path disabled the fixed build matches master bit-for-bit.
10. The residual q8_0/q4_0 drift is the designed int8 Q rounding, as on the other models; a fragment-direct Q quantization (no scratch) is a possible follow-up.


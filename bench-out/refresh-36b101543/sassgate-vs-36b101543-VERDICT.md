# SASS gate: branch vs master 36b101543 (post-rebase, 2026-09-01)

> **SUPERSEDED 2026-09-30 - the conclusion below is WRONG. Do not cite it.**
>
> This document concludes that literal instruction-identity cannot be restored
> without forking the f16 path out of the trait design, and proposes settling for
> opcode-histogram equivalence. Both were overtaken by events. The drift was NOT a
> cicc inlining-boundary effect: the root cause was the K/V pointer parameter
> types (`const void *` where the f16 path wants `const half2 *`), fixed with the
> `fattn_mma_kv_ptr_t` alias now at `ggml/src/ggml-cuda/fattn-mma-load.cuh:34`.
> At the shipped tree `74c907e3d` the gate is **21/21 literally identical**, with
> the f16 path still inside the trait design - see `bench-out/sassgate-2026-09-25/`.
>
> What remains valid here is the measurement, not the verdict: the per-instance
> split (13 of 21 diverging, all nstages>1 shapes), the identical opcode
> histograms across all 285 compared kernel pairs, the drift bounds (instruction
> totals +/-8, REG -19..+3, STACK/SHARED equal), and the three source-restructuring
> experiments that failed. Kept as the record of how the pointer-type cause was
> narrowed down; the PR body cites the 2026-09-25 gate instead.

Measured at the rebase-resolution commit 0fa7291c3 (since superseded by the
swizzled-dequant implementation commit; the f16 objects are unchanged by
the later quant-side edits, so the analysis carries over).

## Result
Strict instruction-identity NO LONGER holds for the f16 MMA instances against the
post-swizzle master (upstream e4b9af007 "CUDA: XOR swizzle flash attn K,V smem fp16
tiles", #25635). 13 of 21 f16 instances show SASS differences; 8 pass identically.

Pattern: every diverging kernel is an nstages>1 (2-stage cp.async pipeline) shape —
DKQ 64/80/96/112/128/256 with ncols2 in {2,4,8,16}. All nstages==1 kernels
(all ncols2==1 instances, DKQ 512/576, ncols2_32 instances) remain instruction-identical.

## Nature of the divergence (bounded)
- SASS opcode histograms are IDENTICAL for all 285 compared kernel pairs
  (exact HMMA/LDSM/LDGSTS/LDS/STS/cp.async/bar.sync counts) — verified programmatically.
- PTX op mixes identical (67 xor.b32, 108 ldmatrix, 66 cp.async, 36 bar.sync in the
  DKQ-64 probe kernel on both sides); differences are basic-block placement (loop
  rotation of the main kb0 loop) and virtual-register numbering.
- Effects: instruction totals equal or ±8; REG counts drift by -19..+3
  (e.g. DKQ-64 ncols 32 kernel: 211 (branch) vs 192 (master)); STACK/SHARED equal.

## Root-cause investigation (all negative)
Three source experiments each rebuilt+re-diffed (ncols1_16-ncols2_2 probe):
1. f16 call sites made textually identical to upstream (direct
   flash_attn_ext_f16_load_tile under if constexpr) -> still 9 diverging kernels.
2. All no-op loader finish() calls compiled out for f16 -> unchanged.
3. staging_K/staging_V nulled for f16 pairs -> unchanged.
Conclusion: the drift is a cicc inlining/value-numbering boundary effect of the
loader-trait indirection itself interacting with the new (larger) swizzle kernel;
it was coincidentally transparent against the pre-swizzle kernel. Restoring literal
identity would require forking the f16 path out of the trait design.
All experiments REVERTED; branch is the clean committed state.

## Revised f16-neutrality evidence for the PR
1. This document: identical opcode histograms, drift bounded to allocation/scheduling.
2. Runtime backstop: f16 bench rows vs master 36b101543 (bench-out/refresh-36b101543/).
3. test-backend-ops FLASH_ATTN_EXT: 3955/3955 pass post-rebase.

Full per-instance log: sassgate-vs-36b101543.txt (this directory)
Probe kernel full diff: sassgate-kdiff-16-2.txt (this directory)

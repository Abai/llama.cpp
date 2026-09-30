# SASS gate root cause: f16 MMA instances, ff8cc2867 vs master 36b101543 (2026-09-14)

Scope: feature commit ff8cc2867 (branch feature/cuda-fattn-mma-fused-dequant) vs master
36b101543 (contains upstream #25635, XOR smem swizzle of the f16 K/V tiles). Toolchain:
docker image llamacpp-bench-dev:12.8.1-gcc14 (nvcc 12.8.93, gcc-14), sm_86, gate flags
(Release, GGML_CUDA_FA_ALL_QUANTS=ON, tests/examples/tools off). Compile only, no GPU.

Verdict up front: FIXABLE. The divergence is caused by ONE construct: the K/V data pointer
parameters of the inlined device functions (flash_attn_ext_f16_process_tile and
flash_attn_ext_f16_iter) are `const void *` on the feature side and `const half2 *` on master.
Typing them per KV type (`const half2 *` for f16, `const void *` otherwise) restores literal
SASS identity for all 21 f16 instance objects and leaves the quantized objects byte-identical.
Diff: bench-out/refresh-36b101543/sassgate-fix-2026-09-14.diff (against ff8cc2867).

## 1. Reproduction

Probe object: fattn-mma-f16-instance-ncols1_16-ncols2_2.cu (kernels for DKQ 64..512).
Built once per side from the nvcc command in compile_commands.json, compared with
scripts/sass-diff-fattn-helper.py diff (same normalization as the gate).

| pair                        | SASS-DIFF kernels | RES-DIFF | note |
|-----------------------------|-------------------|----------|------|
| feat vs master              | 8                 | 1        | DKQ 256 (x2 softcap), 128 (x2), 112, 96, 80, 64; all nstages == 2 shapes |
| feat -lineinfo vs master -lineinfo | 8          | 1        | same kernels, same instruction counts |
| feat vs feat -lineinfo      | 0                 | 0        | -lineinfo does not perturb SASS |
| master vs master -lineinfo  | 0                 | 0        | idem |

Full f16 glob set (gate script, build-dir mode, objects compiled with -P24 in 214 s for
3 sides x 21 objects + 6 quantized): feat vs master FAILS 13/21 instances, exactly the 2-stage
(ncols2 >= 2, DKQ <= 256) shapes; 38 kernels differ in REG. Same picture as the 2026-09-01
verdict. Per-instance log: sassgate-rootcause-2026-09-14-fullgate.txt (this directory).

## 2. Where the PTX first diverges

Method: both sides rebuilt with -lineinfo, `cuobjdump -ptx`, kernel
flash_attn_ext_f16<64,64,16,2,(f16,f16,)false,false> (PTX entry 0 == SASS kernel 35) extracted
from both, virtual registers and labels renamed by order of first appearance, `.loc` directives
stripped into a side table, mangled name replaced. Instruction streams: 5707 (feat) vs 5698
(master).

The first 577 instructions are IDENTICAL (kernel prologue, index math, Q tile load, the
2-stage K/mask preload with cp.async). The first difference is instruction 578, the guard of
the main KV loop in flash_attn_ext_f16_process_tile:

    feature  fattn-mma-f16.cuh:1464   for (; kb0 < kb0_stop-1; ++kb0) {   // ncols2 != 1 branch
    master   fattn-mma-f16.cuh:1310   for (; kb0 < kb0_stop-1; ++kb0) {   // same line of code

PTX excerpt (A = feature, B = master; regs renamed; file 1 = fattn-mma-f16.cuh):

    A   577 1:1464 | add.s32 %r_452, %r_328, -1;          // kb0_stop - 1   (identical)
    A   578 1:1464 | mov.u32 %r_453, 0;
    A   579 1:1464 | setp.lt.s32 %p_454, %r_68, %r_452;    // kb0 < kb0_stop-1
    A   580 1:1464 | @%p_454 bra $LBL_36;                   // -> loop preheader
    A   581 1:1464 | bra.uni $LBL_37;                       // -> loop exit
    A   582 1:1464 | $LBL_36:
    A   583 1:0    | mov.f32 %f_455, 0fFEFFFFFF;            // KQ_max init etc.
    A   ...        | 11 x mov.u32 %r_459..%r_469, %r_453;
    A   598 1:0    | $LBL_38:                               // loop header
    A   599 1:0    | mov.u32 %r_470, 0;
    A   601 6:53   | cp.async.wait_all;                     // first instruction of iter()

    B   577 1:1310 | add.s32 %r_452, %r_328, -1;
    B   578 1:1310 | setp.ge.s32 %p_453, %r_68, %r_452;    // inverted guard
    B   579 1:1310 | mov.u32 %r_454, 0;
    B   583 1:1310 | mov.f32 ...;  12 x mov.u32 %r_459..%r_470, %r_454;   // hoisted above the branch
    B   596 1:1310 | @%p_453 bra $LBL_35;                   // -> loop exit
    B   597 1:0    | mov.u32 %r_459, 0;  ...  11 x mov.u32 (again)        // fallthrough = preheader
    B   612 1:0    | $LBL_36:                               // loop header
    B   613 1:0    | mov.u32 %r_473, 0;
    B   615 6:53   | cp.async.wait_all;

Master has the loop-entry copies hoisted above the guard branch (the SimplifyCFG "hoist common
code from successors" shape); the feature side keeps the plain diamond. With registers fully
masked the whole-kernel structural diff is 24 hunks: this guard, and then large blocks that
are simply placed elsewhere (an 800-instruction copy of the process_tile prologue + loop guard
at A[1208:2008] vs a 630-instruction copy of the iter body at B[2169:2799], a 107-instruction
tail block at B[3566:3673], plus one-instruction `mov 0` / branch-polarity leftovers). No hunk
changes the work done (opcode histograms identical, as previously established); it is one
early CFG-shaping decision whose fallout is block layout, virtual register numbering and
finally ptxas allocation/scheduling drift.

## 3. Hypotheses tested (each: one edit in a scratch worktree of ff8cc2867, single probe
object rebuilt, gate helper diff vs master)

| id  | edit on the feature side                                                            | result vs master |
|-----|-------------------------------------------------------------------------------------|------------------|
| E1  | remove the `Q_A8[1]` / `dQ[1]` local arrays (pass nullptr to iter)                  | 8 SASS-DIFF (unchanged) |
| E2  | remove all four extra iter params (staging_K/V, Q_A8, dQ), recompute inside iter     | 8 SASS-DIFF (unchanged) |
| E3  | K_h2/V_h2 as `const half2 *` in kernel locals + process_tile + iter params           | 0 SASS-DIFF, 0 RES-DIFF: IDENTICAL |
| E4  | drop `type_K, type_V` from the __global__ template (mangled name == master)          | 8 SASS-DIFF (unchanged) |
| E5  | `if constexpr (Q_in_reg && !imma_KQ)` -> plain `if`                                 | 8 SASS-DIFF (unchanged) |
| E3a | only the kernel-local K_h2/V_h2 typed `const half2 *` (params stay void*)           | 8 SASS-DIFF (unchanged) |
| E3b | process_tile + iter params `const half2 *` (kernel locals void*, cast at call)       | IDENTICAL |
| E3c | iter params only `const half2 *` (cast at the 4 iter call sites)                    | IDENTICAL |
| E3d | process_tile params only `const half2 *` (iter params stay void*)                   | IDENTICAL |
| E3e | keep `const void *` params, remove `__restrict__` from them                          | 8 SASS-DIFF; byte-identical to the unmodified feature object |
| E3f | `const char *` instead of `const void *` (params + locals)                          | 8 SASS-DIFF (unchanged) |
| E3g | `const half2 *` params WITHOUT `__restrict__`                                        | IDENTICAL |
| prior (2026-09-01): upstream-identical f16 call sites; finish() compiled out; staging nullptr | unchanged (consistent with E1/E2) |

## 4. Why (root cause)

The trigger is the POINTEE TYPE of the K/V pointer parameters of the always-inlined device
functions on the path kernel -> process_tile -> iter -> loader::load -> load_tile:

- `const half2 *` parameter (master, E3b/c/d/g): identical SASS.
- `const void *` or `const char *` parameter (feature, E3f): divergent SASS.
- `__restrict__` is irrelevant in both directions (E3e == feature byte for byte, E3g == master).
- The type of the kernel-local variable is irrelevant (E3a); only the parameter type of an
  inlined function matters, and either of the two levels is enough.
- Extra parameters, extra dead local arrays, the extra template parameters, the trait
  indirection itself and the `if constexpr` shapes are all irrelevant (E1, E2, E4, E5, prior
  three experiments).

What cicc does with that type cannot be read directly (closed source, no IR dump), but the
observable behaviour pins it to type-derived information the front end attaches to a typed
pointer parameter and not to an untyped one (in LLVM terms: the `align 4` / element-type
information of a `half2 *` argument, which survives inlining, versus an `i8*` argument that
carries none; the same address arithmetic is then canonicalized differently). That difference
is enough to flip the SimplifyCFG hoisting decision at the kb0 loop guard (section 2), and
everything downstream is fallout. Against the pre-swizzle master the same construct happened
to fall on the identical side of that decision; the larger swizzled kernel moved it across.
The earlier "cicc inlining-boundary effect of the trait indirection" conclusion was wrong in
its attribution: the trait is not the cause, the `void *` typing that came with it is.

## 5. The fix

bench-out/refresh-36b101543/sassgate-fix-2026-09-14.diff (17 insertions, 8 deletions, two
files; diff against ff8cc2867):

- fattn-mma-load.cuh: `template <ggml_type type> using fattn_mma_kv_ptr_t =
  std::conditional_t<type == GGML_TYPE_F16, const half2 *, const void *>;` (<type_traits> is
  already used by common.cuh).
- fattn-mma-f16.cuh: the K_h2/V_h2 parameters of flash_attn_ext_f16_iter and
  flash_attn_ext_f16_process_tile become `fattn_mma_kv_ptr_t<type_K/V> const __restrict__`;
  the two kernel-local K_h2/V_h2 pairs use the same alias (with explicit casts on both arms of
  the V_is_K_view conditional). Loader interface (`const void *`) unchanged.

For quantized type_K/type_V the alias is exactly the previous `const void *`, so those
instantiations see the same types as before.

Verification (gate script, build-dir mode, sm_86):
- fix vs master, glob `ncols1_*-ncols2_?`  : 16/16 PASS, "all instance objects instruction-identical"
- fix vs master, glob `ncols1_*-ncols2_??` :  5/5  PASS, idem (21/21 f16 instances total)
- fix vs feature, quantized probe instances ncols1_16-ncols2_2-{q8_0-q8_0, q4_0-q4_0, q8_0-q4_0}:
  3/3 PASS, byte-identical SASS and resources (the quantized paths are untouched).
- Probe object fix vs master: SASS line count 130855 == master (feature: 130903).

Not done here (out of scope for the compile-only rig): rerunning test-backend-ops / the
benchmark rows after applying the diff, and a sm_70/sm_75 compile check of the fixed tree.

## 6. Residual on the unfixed feature commit: occupancy of the 38 REG-drifting kernels

All 38 RES-DIFF kernels (feature vs master, full f16 glob) have `.maxntid 128` and
`.minnctapersm 2` except the three DKQ 256 ones (`.maxntid 64`, `.minnctapersm 4`); dynamic
shared memory per block is identical on both sides by construction (nbytes_staging == 0 for
f16, same host formula otherwise). Register-limited blocks/SM on sm_86 (65536 regs, 256-reg
warp allocation unit, 4-warp allocation granularity, 48 warps max):

| DKQ | ncols1 | ncols2 | softcap | REG feat | REG master | blk/SM feat | blk/SM master |
|-----|--------|--------|---------|----------|------------|-------------|---------------|
| 64  | 1  | 8  | 0 | 130 | 134 | 3 | 3 |
| 64  | 2  | 4  | 0 | 133 | 135 | 3 | 3 |
| 64  | 4  | 2  | 0 | 134 | 136 | 3 | 3 |
| 64  | 16 | 4  | 0 | 211 | 192 | 2 | 2 |
| 80  | 8  | 8  | 0 | 164 | 166 | 3 | 3 |
| 96  | 1  | 8  | 0 | 146 | 148 | 3 | 3 |
| 96  | 2  | 4  | 0 | 146 | 148 | 3 | 3 |
| 96  | 4  | 2  | 0 | 142 | 144 | 3 | 3 |
| 96  | 16 | 2  | 0 | 193 | 196 | 2 | 2 |
| 96  | 16 | 4  | 0 | 188 | 187 | 2 | 2 |
| 112 | 1  | 8  | 0 | 172 | 174 | 2 | 2 |
| 112 | 2  | 4  | 0 | 172 | 174 | 2 | 2 |
| 112 | 4  | 2  | 0 | 168 | 170 | 3 | 2 |  <- only register-limit change, see below
| 112 | 4  | 8  | 0 | 185 | 186 | 2 | 2 |
| 112 | 8  | 4  | 0 | 185 | 186 | 2 | 2 |
| 112 | 16 | 4  | 0 | 205 | 207 | 2 | 2 |
| 112 | 32 | 2  | 0 | 209 | 208 | 2 | 2 |
| 128 | 1  | 8  | 0 | 178 | 180 | 2 | 2 |
| 128 | 2  | 4  | 1 | 178 | 179 | 2 | 2 |
| 128 | 2  | 8  | 0 | 163 | 164 | 3 | 3 |
| 128 | 2  | 8  | 1 | 163 | 164 | 3 | 3 |
| 128 | 4  | 2  | 0 | 180 | 182 | 2 | 2 |
| 128 | 4  | 2  | 1 | 177 | 178 | 2 | 2 |
| 128 | 4  | 4  | 0 | 163 | 164 | 3 | 3 |
| 128 | 4  | 4  | 1 | 163 | 164 | 3 | 3 |
| 128 | 4  | 8  | 0 | 195 | 194 | 2 | 2 |
| 128 | 4  | 8  | 1 | 194 | 192 | 2 | 2 |
| 128 | 8  | 2  | 0 | 163 | 164 | 3 | 3 |
| 128 | 8  | 2  | 1 | 163 | 164 | 3 | 3 |
| 128 | 8  | 4  | 0 | 195 | 194 | 2 | 2 |
| 128 | 8  | 4  | 1 | 194 | 192 | 2 | 2 |
| 128 | 8  | 8  | 1 | 214 | 215 | 2 | 2 |
| 128 | 32 | 2  | 1 | 229 | 227 | 2 | 2 |
| 192 | 4  | 16 | 0 | 211 | 213 | 2 | 2 |
| 192 | 8  | 8  | 0 | 211 | 213 | 2 | 2 |
| 256 | 2  | 8  | 0 | 243 | 245 | 4 | 4 |
| 256 | 4  | 4  | 0 | 243 | 245 | 4 | 4 |
| 256 | 8  | 2  | 0 | 243 | 245 | 4 | 4 |

Drift range -19..+3 registers (sum over the 38 kernels: -23, i.e. the feature side uses
slightly fewer registers on balance). 37 of 38 kernels: identical register-limited blocks/SM.
The single flip (DKQ 112, ncols1 4, ncols2 2: 168 -> 12 warps -> 3 blocks vs 170 -> 8 warps
-> 2 blocks) is masked by shared memory: that shape uses nbatch_fa 128, nbatch_K2/V2 56
(not bank-aligned, so unswizzled stride 60), 4 warps x 16 cols, Q in registers, giving
128*(60+60)*4 + 4*(64+4)*4 = 62528 B dynamic smem per block. On sm_86 (100 KB/SM) that is
1 block/SM, on sm_80 (164 KB/SM) 2 blocks/SM, on both sides. Achieved occupancy is therefore
unchanged for every divergent kernel; the residual of the unfixed commit is a scheduling /
allocation-only difference with no occupancy effect, consistent with the 0.999-1.001 runtime
rows. With the fix applied the residual is zero.

## 7. Method notes / reproducibility

- Per-object compile of the probe instance (nvcc 12.8, sm_86 only): about 2 min single-thread;
  full 21-object f16 set for one side in about 70 s with -P24.
- The gate script works unchanged inside docker in build-dir mode with SASS_DIFF_REBUILD=0
  after compiling the objects from compile_commands.json with the helper's `commands` output.
- Running the container as -u $(id -u):$(id -g) avoids the root-owned build dirs noted on
  2026-09-01.
- PTX comparison recipe: split by `.entry`, rename %r/%rd/%f/%p/%rs/%h registers and $L labels
  by first appearance, drop `.loc` lines into a side table, replace the mangled name, then
  difflib on the instruction stream; mask registers entirely for the structural view.
- All scratch worktrees (sassprobe-*) and build dirs (sassprobe-build-*) were removed; the
  diff file and the full-gate log in this directory are the retained artifacts.

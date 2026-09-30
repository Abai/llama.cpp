# SASS gate root cause, sm_70: kernel flash_attn_ext_f16<80,80,16,2,f16,f16,false,false> (2026-09-15)

Scope: feature commit c18fac264 (fused-dequant MMA with the 2026-09-14 pointer-typing fix) vs
master 36b101543, object fattn-mma-f16-instance-ncols1_16-ncols2_2.cu, CMAKE_CUDA_ARCHITECTURES=70.
Toolchain: docker image llamacpp-bench-dev:12.8.1-gcc14 (nvcc/cicc/ptxas 12.8.93, gcc-14), gate
flags (Release, GGML_CUDA_FA_ALL_QUANTS=ON, tests/examples/tools off). Compile only, no GPU.
Kernel 29 of the object = flash_attn_ext_f16<80,80,16,2,(f16,f16,)false,false>, 6448 SASS
instructions, REG 225, STACK 48 on both sides (gate log sassgate-f16a-sm70-c18fac264-vs-36b101543.txt).

Verdict up front: NOT A SOURCE DIVERGENCE, NOT FIXABLE IN SOURCE. The kernel is compiled
nondeterministically by cicc (the NVVM front/middle end) for compute_70: the same source with
the same command line yields one of several PTX variants that differ only in whether a
partially-used 16-byte shared-memory load in the Volta m8n8k4 fragment loader gets a
`.pragma "used_bytes_mask"` plus a scalar re-load of one lane. ptxas is deterministic on a
fixed PTX, so each PTX variant maps to one SASS variant; the SASS variants differ by +/-1..2
`LDS.U` traded against `NOP` padding and by register names, at identical instruction count,
REG and STACK. Master alone fails the gate against itself (4 rebuilds -> 3 SASS variants). The
feature's PTX for this kernel is byte-identical to master's (modulo the mangled name) whenever
cicc makes the same choice; the two sides only differ in how often cicc makes each choice
(master lands on variant 42260f59 in 18/23 builds, the feature in 5/23). No diff file is
delivered: there is nothing to restore identity against.

## 1. Reproduction

Probe object built per side from the compile_commands.json nvcc command (the gate's
`commands` helper output, `-lineinfo` appended where noted, `--keep` to retain the PTX),
kernel 29 compared with scripts/sass-diff-fattn-helper.py diff (gate normalization) and with
an md5 of the normalized kernel-29 SASS body (short hash used below).

Same source, same command, sm_70, kernel 29 SASS variant per build (short md5 of the body):

| side / configuration                 | builds | 42260f59 | eae89675 | 43eae0e3 | other                       |
|--------------------------------------|--------|----------|----------|----------|-----------------------------|
| master 36b101543                     | 23     | 18       | 2        | 3        | -                           |
| feature c18fac264 (unmodified)       | 23     | 5        | 15       | 2        | 59ede92b x1                 |
| feature + H1 (loader ptr `half2 *`)  | 13     | 7        | 1        | 2        | a7f19d6f x2, 6bfb7474 x1    |
| feature + V11 (no type_K/V on kernel)| 8      | 4        | 4        | 0        | -                           |

Build batches behind the table: 4 + 4 builds of the previous (aborted) attempt found in the
sm70probe-* leftovers and re-hashed; my own 3 sequential builds per side; 8 concurrent builds
per side (paths of unequal length); 8 concurrent builds per side from worktrees with
equal-length paths (sm70probe-A = master, sm70probe-B = feature), which rules out the
`__FILE__` string length of the NO_DEVICE_CODE stubs as the cause of the different
distributions; then 8 concurrent builds each of feature+H1 and feature+V11 (section 4).

Gate helper on pairs of MASTER builds (same source): master-base vs master-r1 -> "SASS-DIFF
kernel 29 (6448 vs 6448 instructions)", master-r1 vs master-r2 -> SASS-DIFF kernel 29,
master-r2 vs master-r3 -> SASS-DIFF kernel 29. Gate helper on feature-base vs master-r1
(both variant eae89675) -> rc=0, instruction-identical. Same-source rebuilds for sm_75 (3x)
and sm_86 (3x) are byte-identical objects (md5 equal), so the instability is specific to the
compute_70 compilation of this TU.

Where in the pipeline: ptxas run three times on one fixed PTX file (cuobjdump -ptx of a master
object, sm_70, -O3) gives the same kernel-29 SASS each time (42260f59, equal to the object it
came from). The PTX itself differs between builds (section 2). So the nondeterminism is in
cicc, ptxas only propagates it. Disabling ASLR (`setarch x86_64 -R`, container run with
seccomp=unconfined) does not remove it: 4 `nvcc -ptx` runs without ASLR still produced 2 PTX
variants for the DKQ 96 kernel and 4 different PTX texts for the DKQ 512 kernels (kernel 29
happened to land on its dominant variant 4/4 in that batch).

Which kernels: per-entry PTX hashes over 17 master and 24 feature builds show the same
class of flip (count of `used_bytes_mask` pragmas varies) in these entries of the object:
DKQ 80 non-softcap (kernel 29, flips in most batches), DKQ 96 non-softcap (about 1 in 8),
DKQ 112 non-softcap (1 in 33), DKQ 128 softcap (about 1 in 4), DKQ 256 softcap (about 1 in
4), DKQ 512 both variants (every build differs). For the entries other than kernel 29 ptxas
happened to fold the PTX differences into identical SASS in the gate run; DKQ 96/112/128/256
are expected to flag occasionally on sm_70 for the same reason, DKQ 512 could too.

## 2. Where the PTX diverges (kernel 29, compute_70)

Lineinfo builds, `.loc` side table, virtual registers renamed by first appearance. The whole
kernel is 5951..5955 PTX instructions depending on the variant; the variants differ at one to
three sites of the same shape. The site that flips most often (master-base = master-li vs
master-r1 = feat-base):

    A (variant d081c4 / SASS 42260f59)           B (variant a61edf / SASS eae89675)
    3159 .pragma "used_bytes_mask 61695";        3159 ld.shared.v4.u32 {%r_2600,%r_2601,%r_2602,%r_2603}, [%r_127+5760];
    3160 ld.shared.v4.u32 {%r_2600..%r_2603}, [%r_127+5760];
    3161 ld.shared.u32 %r_2604, [%r_127+5768];   3160 mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {..}, {%r_1090,%r_1091}, {%r_2600,%r_2601}, {..};
    3162 mma ... {%r_1090,%r_1091}, {%r_2600,%r_2601}, {..};
    3163 mma ... {%r_1092,%r_1093}, {%r_2604,%r_2603}, {..};
                                                 3161 mma ... {%r_1092,%r_1093}, {%r_2602,%r_2603}, {..};

    .loc for 3159..3163 on both sides:
        .loc 5 815 13, function_name $L__info_string11, inlined_at 2 863 9
        file 5 = ggml/src/ggml-cuda/common.cuh:815      ((int4 *) dst)[i] = ((const int4 *) src)[i];   (ggml_cuda_memcpy_1<16>)
        file 2 = ggml/src/ggml-cuda/mma.cuh:863         ggml_cuda_memcpy_1<4*sizeof(half2)>(t.x, xs0 + t.get_i(0)*stride);
                                                        (load_ldmatrix(tile<8,4,half2,DATA_LAYOUT_I_MAJOR_MIRRORED> &, ...))

    Identical on both sides; both files are unchanged by the feature commit (git diff --stat
    36b101543 c18fac264 -- common.cuh mma.cuh fattn-common.cuh is empty).

The other flip sites seen in kernel 29 are the same construct at [%r_127] (mask 255, then a
`ld.shared.v2.u32 [%r_127+8]`) and at [%r_3445+5760] later in the unrolled loop. mask 61695 =
0xF0FF marks bytes 8..11 of the 16-byte load as unused and the byte range is then re-loaded
with a scalar `ld.shared.u32`; the B operand of the second m8n8k4 takes {re-loaded lane 2,
lane 3}. Whether NVVM performs this split of the K-fragment load (T_A_KQ = tile<8,4,half2,
I_MAJOR_MIRRORED> on Volta, fattn-mma-f16.cuh:1193 feature / :1039 master, loaded via
ggml_cuda_fattn_smem_swizzle::load_ldmatrix at fattn-mma-f16.cuh:710/736 feature, 639/665
master) varies from run to run. The construct only exists in the Volta m8n8k4 code path; the
Turing+ path uses ldmatrix (fully used 16-byte fragments, no partial-use analysis), which is
why sm_75/sm_86/sm_120 are deterministic and identical.

Why DKQ 80 more than the others: the flip needs a partially-used 16-byte fragment load in
the unrolled KQ loop; the non-softcap DKQ 80 kernel is the shape where NVVM's choice is
least stable (flips in most batches), DKQ 96/112/128/256 flip at lower rates, DKQ 512 flips
every time. That is a property of the optimizer's internal ordering for those specific
unrolled bodies, not of any construct that is specific to head size 80 in the source.

Feature vs master PTX for kernel 29 when cicc lands on the same variant: per-entry hash after
mangled-name normalization is equal (d081c4 for master m1,m2,m3 and feature f1,f6,B2,B4,B6,B8,
C1,C2,C4,C7; 46d417 for master-r1, feat-base, feat-clean3; c1466b for master-base, master-r3,
feat-V12). I.e. the feature side's compute_70 PTX for this kernel is byte-identical to master
modulo the kernel name whenever the nondeterministic choice coincides.

## 3. SASS residual between the variants (kernel 29, all variants 6448 instructions, REG 225, STACK 48)

Register- and immediate-masked comparison and opcode histograms against variant 42260f59:

| variant  | raw differing lines | masked differing lines | opcode histogram delta        |
|----------|---------------------|------------------------|-------------------------------|
| eae89675 | 3271                | 101                    | LDS.U -1, NOP +1              |
| 43eae0e3 | 1111                | 97                     | LDS.U +1, NOP -1              |
| a7f19d6f | n/a                 | n/a                    | LDS.U +2, NOP -2              |
| 59ede92b | n/a                 | n/a                    | none (register renaming only) |
| 9c1648a2 | n/a                 | n/a                    | none (register renaming only) |

Base histogram: 143 LDS.U, 150 LDS.U.128, 254 LDS.U.64, 49 NOP, 1440 HMMA.884. The masked
differences are local reorderings of HMMA/LDS/FADD/FMNMX around the flipped load plus a
trailing NOP; register-limited occupancy is unchanged (same REG on both sides, no RES-DIFF in
the gate log). Runtime effect: at most two extra 4-byte LDS per KQ tile step in a 6448
instruction kernel, i.e. below measurement noise; nothing to fix.

## 4. Hypotheses tested (feature side edits, sm_70, kernel 29)

The gate compares one build per side; with an unstable reference the only meaningful metric
is how often a configuration lands on master's dominant variant. Single builds cannot answer
that, so the previous attempt's 12 single-build V-variants (V1..V12, all layered on H1: extra
iter parameters removed, Q_A8/dQ locals removed, finish() under if constexpr, tile_K8/dK_sm
hoisting, direct f16 load_tile call, type params dropped from the kernel) are listed as
uninformative: they produced 42260f59 x5, eae89675 x5, a7f19d6f x1, 9c1648a2 x1, which is
the same variant set. Batches of 8 concurrent builds were run for the two hypotheses that
are the closest match to the question ("another untyped pointer on the sync path", "the
extra template parameters"):

| id  | edit                                                                                   | 42260f59 (master-dominant) rate |
|-----|----------------------------------------------------------------------------------------|---------------------------------|
| -   | master 36b101543 (reference)                                                           | 18/23 = 78%                     |
| -   | feature c18fac264 unmodified                                                           | 5/23 = 22%                      |
| H1  | fattn-mma-load.cuh: f16 loader `load()` KV parameter `const half2 *` instead of `const void *` (the last untyped K/V pointer on the f16 path; 4-line diff) | 7/13 = 54% (4/8 this batch, 3/5 previous attempt) |
| V11 | drop `type_K, type_V` from the `__global__` template (mangled name equals master's)     | 4/8 = 50%                       |

Both edits move the distribution part-way toward master's but every configuration, master
included, keeps producing several variants. Expected gate mismatch probability for a single
A-vs-B comparison, from the tallies above: unmodified feature about 70%, with H1 or V11
about 55%, and even two independent master builds disagree about 35% of the time. Hence no
source edit qualifies as a fix under the gate's definition (instruction identity on sm_70),
and no diff file is produced. Two further hypotheses from the task list were not run as
batches because they cannot change the outcome: `if constexpr` shapes and `__forceinline__`
(all functions on the path are already __forceinline__ and the flip site is in unchanged
shared code that both sides inline identically, see the PTX identity in section 2).

Open (out of scope, would change master-shared code): rewriting the Volta
`load_ldmatrix(tile<8,4,half2,I_MAJOR_MIRRORED>&)` load (mma.cuh:863) so that the fragment
is loaded as two 8-byte copies or consumed without a partially-dead lane would probably
remove the partial-use decision and with it the instability; that is a change for the shared
mma.cuh, would need its own sm_70 benchmark, and is not part of this PR.

## 5. Consequence for the gate

- sm_70 cannot be a hard instruction-identity gate for this object with this toolchain: the
  reference is unstable. Options, in order of preference: (a) keep sm_75/sm_86/sm_120 as the
  hard gate (all 21 f16 instances identical there, existing logs) and treat sm_70 as
  informational; (b) if an sm_70 gate is wanted, build the master side N (>= 3) times and
  accept a kernel if it matches any of the N; (c) compare PTX instead of SASS for sm_70 with
  `.pragma "used_bytes_mask"` lines removed and the adjacent narrowed re-load folded, which is
  fragile. A register-masked SASS diff is not enough (97..101 masked lines differ).
- The kernels to expect flagged on sm_70 in this object: 29 (DKQ 80) most of the time; 24
  (DKQ 96), 19 (DKQ 112), 14 (DKQ 128 softcap), 9 (DKQ 256 softcap), 3/4 (DKQ 512) at lower
  rates. Other instance objects were not sampled repeatedly; the same construct is present
  in every Volta instance so occasional single-kernel flags there are expected too.
- The runtime residual between variants is +/-1..2 LDS.U at equal REG/STACK/instruction
  count (section 3): no occupancy or measurable throughput effect on either side.

## 6. Method notes / reproducibility

- Sequential compile of the probe object for sm_70: 26..27 s (cicc about 17 s, ptxas about
  6 s); 16 concurrent compiles: 57 s on this host.
- Hash recipe: `cuobjdump --dump-sass obj | helper.parse_sass` (gate normalization), md5 of
  the kernel-29 body; PTX per-entry md5 after removing `L9ggml_type1ELS0_1E` and mapping
  `S2_S2_S2_S2_` to `S1_S1_S1_S1_` in the entry name; PTX structural diff after renaming
  %r/%rd/%f/%p/%rs/%h registers and $L labels by first appearance and moving `.loc` lines
  into a side table.
- Fixed-PTX ptxas check: `cuobjdump -ptx obj`, cut from `.version` to the next `Fatbin`
  header line, `ptxas -arch=sm_70 -m64 -O3`.
- All sm70probe-* worktrees, build dirs, output dir and containers were removed after this
  report was written; the tallies above are the retained evidence.

## Summary (10 lines)

1. The sm_70 kernel-29 mismatch is cicc build-to-build nondeterminism, not a feature-side construct.
2. Master 36b101543 alone yields 3 SASS variants of that kernel over 23 builds; the gate fails master vs master.
3. The flip is a `.pragma "used_bytes_mask"` + scalar re-load decision on the Volta m8n8k4 K-fragment load, common.cuh:815 inlined at mma.cuh:863, files unchanged by the feature.
4. ptxas is deterministic on fixed PTX; ASLR is not the trigger; sm_75/sm_86 same-source rebuilds are byte-identical.
5. Feature PTX for the kernel is byte-identical to master's (modulo name) whenever cicc makes the same choice.
6. The sides differ only in the choice frequency: master 78% on variant 42260f59, feature 22% (65% on eae89675).
7. SASS variants differ by +/-1..2 LDS.U vs NOP and register names; 6448 instructions, REG 225, STACK 48 in all.
8. Typing the loader's KV pointer (H1) or dropping the kernel type params (V11) shifts the feature to about 50%; not a fix, no diff delivered.
9. DKQ 96/112/128/256/512 kernels of the same object show the same PTX flips at lower rates; expect occasional sm_70 flags there too.
10. Recommendation: hard gate on sm_75/86/120 only; on sm_70 accept a match against any of N>=3 master builds or treat as informational.

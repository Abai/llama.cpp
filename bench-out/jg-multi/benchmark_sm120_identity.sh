#!/bin/sh
# The 'identity' half of the throughput sweep, guarded to sm_120. Thin wrapper
# over benchmark.sh --identity; every argument is forwarded unchanged.
#
#   sh benchmark_sm120_identity.sh --models <dir> [--reps N] [--dry-run]
#
# The complementary rungs - f16 KV, everything fattn-vec handles (ub1 on
# sm_86, ub1/ub2 on sm_120, and every decode rung), deepseek on the fused
# branch (head dims 576/512 miss the fused predicate), and cap configurations
# that stay under the conversion cap. Their DEVICE code is master's, and the
# SASS gate proves it. They are MEASURED rather than asserted because
# host-side dispatch still differs - cap evaluates
# ggml_cuda_flash_attn_ext_get_f16_extra_data on every MMA op to choose
# write_partial, and both features change get_alloc_size and therefore the
# CUDA-graph shape - and the launch-bound rungs are exactly where per-op host
# overhead would surface. Expect 1.00; a deviation is a finding.
#
# --small and --identity are complementary: together they are exactly the
# unpartitioned sweep, so nothing is dropped from the final tables. The split
# only decides what is measured first. Run --small before submission and
# --identity as the confirmation pass. Add --force to re-measure the rungs this
# run targets even when they already carry a current stamp.
#
# Estimated wall clock here: ~0 h (fused: already current at this tip) / ~7.8 h (cap)
# (rungs carrying a current stamp are skipped, so a repeat run is cheap)
set -e
cd "$(dirname "$0")"
want=120
have=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ".")
[ "$have" = "$want" ] || {
    echo "ERROR: this wrapper is for sm_$want, but this GPU reports sm_$have."
    echo "       Use benchmark_sm${have}_identity.sh instead."
    exit 1; }
exec sh benchmark.sh --identity "$@"

#!/bin/sh
# The 'small' half of the throughput sweep, guarded to sm_120. Thin wrapper
# over benchmark.sh --small; every argument is forwarded unchanged.
#
#   sh benchmark_sm120_small.sh --models <dir> [--reps N] [--dry-run]
#
# Rungs whose kernels this feature actually changes: the fused MMA instances,
# or the chunked (write_partial=true) cap path. This is where a real speedup
# or regression can appear, and it is the data the PR tables rest on.
#
# --small and --identity are complementary: together they are exactly the
# unpartitioned sweep, so nothing is dropped from the final tables. The split
# only decides what is measured first. Run --small before submission and
# --identity as the confirmation pass. Add --force to re-measure the rungs this
# run targets even when they already carry a current stamp.
#
# Estimated wall clock here: ~1.6 h (fused) / ~2.6 h (cap); ~3.5 h for both, master rungs shared
# (rungs carrying a current stamp are skipped, so a repeat run is cheap)
set -e
cd "$(dirname "$0")"
want=120
have=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ".")
[ "$have" = "$want" ] || {
    echo "ERROR: this wrapper is for sm_$want, but this GPU reports sm_$have."
    echo "       Use benchmark_sm${have}_small.sh instead."
    exit 1; }
exec sh benchmark.sh --small "$@"

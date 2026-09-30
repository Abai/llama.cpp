#!/bin/sh
# The 'small' half of the throughput sweep, guarded to sm_86. Thin wrapper
# over benchmark.sh --small; every argument is forwarded unchanged.
#
#   sh benchmark_sm86_small.sh --models <dir> [--reps N] [--dry-run]
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
# Estimated wall clock here: ~6.7 h (fused) / ~8.1 h (cap); ~10.2 h for both, master rungs shared
# (rungs carrying a current stamp are skipped, so a repeat run is cheap)
set -e
cd "$(dirname "$0")"
want=86
have=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ".")
[ "$have" = "$want" ] || {
    echo "ERROR: this wrapper is for sm_$want, but this GPU reports sm_$have."
    echo "       Use benchmark_sm${have}_small.sh instead."
    exit 1; }
exec sh benchmark.sh --small "$@"

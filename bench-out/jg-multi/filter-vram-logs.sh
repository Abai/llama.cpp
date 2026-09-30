#!/bin/sh
# Shrink the *-vram-*.log files produced by `benchmark.sh --vram` to the lines
# that carry evidence, so they can be committed alongside the rendered tables.
#
# Why: llama-server at -lv 5 writes ~2400 lines per probe, ~74% of which are
# create_tensor/print_info spam; 200 probes came to 41 MB. The VRAM tables are
# rendered by mkvramtables.py, which reads exactly three things from each log:
#
#   CUDA0 compute buffer size = <N> MiB
#   CUDA0 KV buffer size      = <N> MiB
#   failed to allocate | out of memory | cudaMalloc failed      (the OOM rungs)
#
# Everything this script keeps is either one of those or provenance a reviewer
# needs to check the probe was configured as claimed (device, driver, model
# file, n_ctx/n_ubatch/flash_attn, and the KV cache types). The rendered tables
# must come out byte-identical after filtering - that is the acceptance test.
#
# Idempotent: already-filtered logs carry a marker line and are skipped. The
# rung stamps live in .log.bin and hash the SOURCE TREE, not the log, so
# filtering does not invalidate them or trigger a re-measure.
#
#   sh filter-vram-logs.sh <dir> [<dir>...]
set -e

MARKER='# [filtered by filter-vram-logs.sh: allocation-relevant lines only]'

# One alternation, built on a single line: a stray empty alternative (||) is an
# ERE that matches EVERY line, which silently turns this script into a no-op.
KEEP='ggml_cuda_init|Device [0-9]+:|CUDA0 compute buffer size|CUDA0 KV buffer size|CUDA0 model buffer size|llama_context: *(n_ctx|n_ubatch|n_batch|flash_attn|type_k|type_v)|failed to allocate|out of memory|cudaMalloc failed|srv .*load_model: loading model|srv .*listening'

total_before=0; total_after=0; n=0; skipped=0
for d in "$@"; do
    for f in "$d"/*vram*.log; do
        [ -e "$f" ] || continue
        if head -1 "$f" | grep -qF "$MARKER"; then skipped=$((skipped+1)); continue; fi
        before=$(wc -c < "$f")
        { echo "$MARKER"; grep -aE "$KEEP" "$f" || true; } > "$f.filtered"
        # Refuse to shrink a log that would stop reproducing its own table row.
        if ! grep -qE 'CUDA0 compute buffer size|failed to allocate|out of memory|cudaMalloc failed' "$f.filtered"; then
            echo "SKIP (no evidence line found, left intact): $f"
            rm -f "$f.filtered"; continue
        fi
        mv "$f.filtered" "$f"
        after=$(wc -c < "$f")
        total_before=$((total_before+before)); total_after=$((total_after+after)); n=$((n+1))
    done
done
echo "filtered $n logs ($skipped already filtered): $((total_before/1024)) KiB -> $((total_after/1024)) KiB"

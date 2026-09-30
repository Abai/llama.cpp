#!/bin/sh
# Section F merge: pull the RTX 5070 Laptop results into both trees. RUN THIS ON A MACHINE WITH
# rsync AND ssh ACCESS TO THE LAPTOP (the rig's host, not the workspace container: the container has
# no rsync, no keys, and port 22 on the laptop is filtered from it). User and port come from the
# host's ~/.ssh/config; the host alias below is all that is passed.
#
# The laptop renders its own TABLES-/PPL-RTX5070LaptopGPU-*.md, but mktables/mkppltables
# put BOTH GPUs in one file, so the rig's TABLES-/PPL-RTX3090-*.md is already complete and
# the laptop copies are duplicates carrying the laptop's stale view of the RTX 3090 rows
# (114 of 178 rows and a MISSING cell, last time). rsync applies the FIRST matching rule,
# so these two excludes must stay above the *RTX5070LaptopGPU* include.
# Filtered on purpose. An unfiltered `rsync -aH src/ dst/` transfers on any size/mtime difference
# with the SOURCE winning: the laptop's cap-src held 503 stale RTX3090 rungs and pre-fix copies of
# benchmark.sh/mktables.py with names identical to the rig's fresh files, and would overwrite them.
# Only laptop-named artifacts and the laptop monitor CSV cross over. No --delete, ever.
#
#   sh bench-out/merge-laptop.sh          # dry run: itemised list + counts, transfers nothing
#   sh bench-out/merge-laptop.sh --go     # real transfer
#
# Env: LAPTOP (ssh host alias, default nobo.local), SRC_ROOT / DST_ROOT (repo roots; SRC_ROOT
# defaults to "$LAPTOP:<repo>", override both with local paths to test the filter offline).
set -e
LAPTOP=${LAPTOP:-nobo.local}
REPO=/home/abai/Projects/abai/online/workspace/llama.cpp
SRC_ROOT=${SRC_ROOT:-$LAPTOP:$REPO}
DST_ROOT=${DST_ROOT:-$REPO}
DRY="-n -i"; [ "$1" = "--go" ] && DRY="-i"
tmp=$(mktemp)
trap 'rm -f "$tmp"' EXIT
rc_all=0
for tree in "" cap-src/; do
    src="$SRC_ROOT/${tree}bench-out/jg-multi/"
    dst="$DST_ROOT/${tree}bench-out/jg-multi/"
    echo "### ${tree:-main tree}: $src -> $dst"
    if rsync -aHm $DRY \
            --include='*/' \
            --exclude='TABLES-RTX5070LaptopGPU*' \
            --exclude='PPL-RTX5070LaptopGPU*' \
            --include='*RTX5070LaptopGPU*' \
            --include='gpu-monitor-laptop*.csv' \
            --include='perf-counters-*.txt' \
            --include='HANDBACK-*.md' \
            --exclude='*' \
            "$src" "$dst" > "$tmp" 2>&1; then rc=0; else rc=$?; fi
    cat "$tmp"
    echo "### ${tree:-main tree}: $(grep -c '^>f' "$tmp" || true) files listed, rsync exit $rc"
    [ $rc -eq 0 ] || rc_all=$rc
done
echo "### expected: every *RTX5070LaptopGPU* artifact plus gpu-monitor-laptop*.csv; nothing else"
echo "### nothing named RTX3090-*, benchmark.sh, mktables.py, RESUME-*, run-*.log may appear above"
exit $rc_all

#!/bin/sh
# Offline test of bench-out/merge-laptop.sh: run inside the toolchain image with rsync apt-installed,
# e.g. docker run --rm -v $REPO:$REPO --entrypoint sh <image> -c "apt-get -qq update && apt-get -qq install -y rsync && sh $REPO/bench-out/merge-laptop-test.sh"
set -e
REPO=/home/abai/Projects/abai/online/workspace/llama.cpp
LAP=/tmp/lap; DST=/tmp/dst
# Start from scratch: the fixtures are only ever created, never removed, so without
# this a second run sees TEST 2's output still sitting in $DST and TEST 1's
# "dry run wrote nothing" assertion fails spuriously.
rm -rf "$LAP" "$DST"
F=cuda-fattn-mma-fused-dequant; C=cuda-fattn-convert-buffer-cap; M=master-36b101543
mk() { mkdir -p "$(dirname "$1")"; printf '%s\n' "${2:-LAPTOP}" > "$1"; }
# --- laptop main tree: wanted files
for f in $F/RTX5070LaptopGPU-gemma-q80q80-branch-ub1.sql $F/RTX5070LaptopGPU-gemma-q80q80-branch-ub1.sql.bin \
         $F/RTX5070LaptopGPU-gemma-q80q80-branch-ub1.err $F/RTX5070LaptopGPU-tests-fa-branch.log \
         $F/RTX5070LaptopGPU-gemma-ppl-f16-c16384-n9-branch.txt $F/TABLES-RTX5070LaptopGPU-sm120-8GB.md \
         $F/PPL-RTX5070LaptopGPU-sm120-8GB.md $F/VALIDATE-RTX5070LaptopGPU-sm120-8GB.txt \
         $M/RTX5070LaptopGPU-gemma-q80q80-master-ub1.sql $M/RTX5070LaptopGPU-gemma-q80q80-master-ub1.sql.bin \
         $M/RTX5070LaptopGPU-gemma-q80q80-master-ub1.err gpu-monitor-laptop.csv gpu-monitor-laptop-115w.csv perf-counters-115w-BEFORE-fused.txt HANDBACK-115w-2026-09-18.md; do mk "$LAP/bench-out/jg-multi/$f"; done
# --- laptop main tree: decoys that must NOT cross
for f in benchmark.sh mktables.py mkvramtables.py RESUME-laptop.md run-sweep-fused.log run-tests-cap-run1.log gpu-monitor-rig.csv \
         $F/RTX3090-gemma-q80q80-branch-ub1.sql $F/TABLES-RTX3090-sm86-24GB.md $M/RTX3090-gemma-q80q80-master-ub1.sql \
         attic-gemma-c32768/RTX3090-gemma-ppl-f16-c32768-n9-master.log \
        cuda-fattn-x/TABLES-RTX5070LaptopGPU-sm120-8GB.md \
        cuda-fattn-x/PPL-RTX5070LaptopGPU-sm120-8GB.md; do mk "$LAP/bench-out/jg-multi/$f" STALE-LAPTOP-COPY; done
mkdir -p "$LAP/bench-out/jg-multi/emptydir"
# --- laptop cap tree
for f in $C/RTX5070LaptopGPU-llama-q80q80-branch32-ub2.sql $C/RTX5070LaptopGPU-llama-q80q80-branch32-ub2.sql.bin \
         $C/RTX5070LaptopGPU-llama-q80q80-branch32-ub2.err $C/VALIDATE-RTX5070LaptopGPU-sm120-8GB-run1.txt \
         $C/VALIDATE-RTX5070LaptopGPU-sm120-8GB.txt $C/TABLES-RTX5070LaptopGPU-sm120-8GB.md \
         $M/RTX5070LaptopGPU-llama-q80q80-master-ub2.sql; do mk "$LAP/cap-src/bench-out/jg-multi/$f"; done
for f in benchmark.sh mktables.py $C/RTX3090-llama-q80q80-branch-ub2.sql $M/RTX3090-llama-q80q80-master-ub2.sql; do
    mk "$LAP/cap-src/bench-out/jg-multi/$f" STALE-LAPTOP-COPY; done
# --- rig destination: same-named files with sentinel content, OLDER than the laptop decoys
for f in benchmark.sh mktables.py $M/RTX3090-gemma-q80q80-master-ub1.sql $F/RTX3090-gemma-q80q80-branch-ub1.sql; do mk "$DST/bench-out/jg-multi/$f" RIG-OK; done
for f in benchmark.sh mktables.py $C/RTX3090-llama-q80q80-branch-ub2.sql $M/RTX3090-llama-q80q80-master-ub2.sql; do mk "$DST/cap-src/bench-out/jg-multi/$f" RIG-OK; done
find "$DST" -type f -exec touch -d '2026-09-16 08:00' {} +
find "$LAP" -type f -exec touch -d '2026-09-17 12:00' {} +

echo "################ CONTROL: what the brief's UNFILTERED rsync would transfer (dry run, main tree)"
rsync -aHn -i "$LAP/bench-out/jg-multi/" "$DST/bench-out/jg-multi/" | grep '^>f' | sed 's/^/   /'

echo "################ TEST 1: merge-laptop.sh dry run"
SRC_ROOT=$LAP DST_ROOT=$DST sh "$REPO/bench-out/merge-laptop.sh" > /tmp/dry.out 2>&1; echo "script exit $?"
cat /tmp/dry.out
echo "--- assertions (dry run) ---"
listed=$(grep '^>f' /tmp/dry.out | awk '{print $2}' | sort)
echo "$listed" | grep -q 'RTX3090' && echo "FAIL: RTX3090 decoy listed" || echo "ok: no RTX3090 file listed"
echo "$listed" | grep -qE '^(benchmark.sh|mktables.py|mkvramtables.py|RESUME-laptop.md|run-)' && echo "FAIL: script/log decoy listed" || echo "ok: no script/notes/log decoy listed"
echo "$listed" | grep -q '^gpu-monitor-laptop.csv$' && echo "ok: monitor CSV listed" || echo "FAIL: monitor CSV missing"
grep -qE '^cd.* (attic-gemma-c32768|emptydir)/' /tmp/dry.out && echo "FAIL: pruned dir would be created" || echo "ok: attic/empty dirs pruned"
grep -q 'TABLES-RTX5070LaptopGPU' /tmp/dry.out && echo "FAIL: laptop TABLES render would transfer" || echo "ok: laptop TABLES render excluded"
grep -q 'PPL-RTX5070LaptopGPU' /tmp/dry.out && echo "FAIL: laptop PPL render would transfer" || echo "ok: laptop PPL render excluded"
echo "$listed" | grep -q "^gpu-monitor-laptop-115w.csv$" && echo "ok: 115 W monitor CSV listed" || echo "FAIL: 115 W monitor CSV missing"
echo "$listed" | grep -q "gpu-monitor-rig.csv" && echo "FAIL: rig monitor decoy listed" || echo "ok: rig-named monitor decoy excluded"
echo "$listed" | grep -q "^perf-counters-115w-BEFORE-fused.txt$" && echo "ok: counters file listed" || echo "FAIL: counters file missing"
echo "$listed" | grep -q "^HANDBACK-115w-2026-09-18.md$" && echo "ok: hand-back listed" || echo "FAIL: hand-back missing"
# 13 = 15 wanted main-tree files minus TABLES-/PPL-RTX5070LaptopGPU (now excluded:
# the rig's own render already carries both GPUs); 6 = 7 cap files minus its TABLES-.
echo "listed main+cap = $(echo "$listed" | wc -l) (expect 13 + 6 = 19)"
[ ! -e "$DST/bench-out/jg-multi/$F/RTX5070LaptopGPU-gemma-q80q80-branch-ub1.sql" ] && echo "ok: dry run wrote nothing" || echo "FAIL: dry run wrote files"

echo "################ TEST 2: merge-laptop.sh --go"
SRC_ROOT=$LAP DST_ROOT=$DST sh "$REPO/bench-out/merge-laptop.sh" --go > /tmp/go.out 2>&1; echo "script exit $?"
grep '^###' /tmp/go.out
echo "--- assertions (after --go) ---"
n_lap=$(find "$DST" -type f -name '*RTX5070LaptopGPU*' | wc -l); echo "laptop-named files in dst: $n_lap (expect 15)"
[ -e "$DST/bench-out/jg-multi/gpu-monitor-laptop-115w.csv" ] && echo "ok: 115 W monitor arrived" || echo "FAIL: 115 W monitor missing"
[ -e "$DST/bench-out/jg-multi/gpu-monitor-rig.csv" ] && echo "FAIL: rig-named decoy copied" || echo "ok: rig-named decoy not copied"
[ "$(cat $DST/bench-out/jg-multi/gpu-monitor-laptop.csv)" = LAPTOP ] && echo "ok: monitor CSV arrived" || echo "FAIL: monitor CSV"
bad=0; for f in bench-out/jg-multi/benchmark.sh bench-out/jg-multi/mktables.py bench-out/jg-multi/$M/RTX3090-gemma-q80q80-master-ub1.sql \
              bench-out/jg-multi/$F/RTX3090-gemma-q80q80-branch-ub1.sql cap-src/bench-out/jg-multi/benchmark.sh cap-src/bench-out/jg-multi/mktables.py \
              cap-src/bench-out/jg-multi/$C/RTX3090-llama-q80q80-branch-ub2.sql cap-src/bench-out/jg-multi/$M/RTX3090-llama-q80q80-master-ub2.sql; do
    [ "$(cat "$DST/$f")" = RIG-OK ] || { echo "FAIL: rig file overwritten: $f"; bad=1; }; done
[ $bad = 0 ] && echo "ok: all 8 same-named rig files intact (RIG-OK), despite newer laptop copies"
find "$DST" -type f -newer /tmp/dry.out | grep -v RTX5070LaptopGPU | grep -v "gpu-monitor-laptop\|perf-counters-\|HANDBACK-" | sed 's/^/FAIL: unexpected new file: /' || true
[ -d "$DST/bench-out/jg-multi/attic-gemma-c32768" ] && echo "FAIL: attic dir created" || echo "ok: attic dir not created"
[ -d "$DST/bench-out/jg-multi/emptydir" ] && echo "FAIL: empty dir created" || echo "ok: empty dir not created"
[ -e "$DST/bench-out/jg-multi/RESUME-laptop.md" ] && echo "FAIL: RESUME copied" || echo "ok: RESUME-laptop.md not copied"
echo "################ TEST 3: re-run --go is a no-op"
SRC_ROOT=$LAP DST_ROOT=$DST sh "$REPO/bench-out/merge-laptop.sh" --go | grep '^### .*files listed'

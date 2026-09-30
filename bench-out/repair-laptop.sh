#!/bin/sh
# Wrapper for repair-laptop.py. Run ON THE LAPTOP, after the fixed benchmark.sh
# has been pushed - it refuses to run against the buggy runner, because deleting
# cells and then re-measuring them with the old predicate reproduces the split.
#
#   sh bench-out/repair-laptop.sh          # list what would be deleted
#   sh bench-out/repair-laptop.sh --go     # delete
set -e
cd "$(dirname "$0")/.."
REPO=$(pwd); export REPO
FIXED=ab47ed3a90ea21ff004bb89ac7edcfdc
for f in bench-out/jg-multi/benchmark.sh cap-src/bench-out/jg-multi/benchmark.sh; do
    have=$(md5sum "$f" | cut -d' ' -f1)
    [ "$have" = "$FIXED" ] || {
        echo "ERROR: $f is $have, expected the fixed runner $FIXED."
        echo "       Push the fixed benchmark.sh from the rig before repairing."
        exit 1; }
done
echo "runner verified fixed on both trees"
if command -v python3 > /dev/null 2>&1; then
    exec python3 bench-out/repair-laptop.py "$@"
fi
exec docker run --rm --entrypoint python3 -v "$REPO:$REPO" -e REPO="$REPO" \
     llamacpp-bench-dev:12.8.1-gcc14 "$REPO/bench-out/repair-laptop.py" "$@"

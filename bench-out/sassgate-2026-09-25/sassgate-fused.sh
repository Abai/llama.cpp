#!/bin/sh
# SASS gate: fused 74c907e3d (build-cap, main tree) vs master 36b101543 (build-master), sm_86,
# the 21 f16 instance objects, paired by order inside each object via the repo's own helper.
BASE=/home/abai/Projects/abai/online/workspace/llama.cpp
INST=ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/template-instances
cd $BASE; rc=0; n=0
for a in build-cap/$INST/fattn-mma-f16-instance-ncols1_*-ncols2_*.cu.o; do
  case "$a" in *q4_0*|*q8_0*) continue;; esac
  b=build-master/$INST/$(basename $a); n=$((n+1))
  echo "##### $(basename $a)"
  if python3 scripts/sass-diff-fattn-helper.py diff "$a" "$b"; then echo "  -> IDENTICAL"; else echo "  -> DIFF"; rc=1; fi
done
echo "=== objects compared: $n   GATE: $([ $rc = 0 ] && echo PASS || echo FAIL)"

#!/bin/sh
# Re-probe of the issue's 180k->196k max-context claim with Q4_0 weights.
# Mirrors benchmark.sh run_vram: llama-server in the bench image, success =
# "listening on", failure = allocation error; log saved per probe.
# Usage: probe.sh <side:master|fused> <ctx>
set -e
ROOT=/home/abai/Projects/abai/online/workspace/llama.cpp
# NOTE: docker drives the HOST daemon; mounts must use host-side paths.
# The host's models dir (container-local view: /home/abai/models):
MODELS=/home/abai/Projects/abai/online/models
MODEL=$MODELS/unsloth/Qwen3.6-27B-MTP-GGUF/${3:-Qwen3.6-27B-Q4_0.gguf}
IMG=llamacpp-bench-dev:12.8.1-gcc14
side=$1; ctx=$2
tag=${3:+-$(echo "$3" | sed 's/Qwen3.6-27B-//;s/\.gguf//')}
case $side in
    master) srv=$ROOT/build-master/bin/llama-server;;
    fused)  srv=$ROOT/build-fused-probe/bin/llama-server;;
    *) echo "bad side"; exit 1;;
esac
name="$ROOT/bench-out/probe-196k-q40/RTX3090-qwen$tag-vram-q80-q80-c$ctx-$side"
cid=$(docker run -d --gpus all --entrypoint "$srv" -v "$ROOT:$ROOT" -v "$MODELS:$MODELS" "$IMG" \
    -m "$MODEL" -c "$ctx" -ngl 99 -fa on -ub 512 --no-warmup -lv 5 -ctk q8_0 -ctv q8_0)
verdict=TIMEOUT
i=0
while [ $i -lt 90 ]; do
    sleep 2; i=$((i+1))
    log=$(docker logs "$cid" 2>&1)
    if echo "$log" | grep -q "listening on"; then verdict=STARTS; break; fi
    if echo "$log" | grep -qE "failed to allocate|out of memory|cudaMalloc failed|error loading model"; then verdict=FAILS; break; fi
    [ "$(docker inspect -f '{{.State.Running}}' "$cid" 2>/dev/null)" = "true" ] || { verdict=EXITED; break; }
done
docker logs "$cid" > "$name.log" 2>&1 || true
docker rm -f "$cid" > /dev/null 2>&1 || true
compute=$(grep -E 'CUDA0 compute buffer size' "$name.log" | tail -1 | grep -oE '[0-9.]+ MiB' || echo 'n/a')
kv=$(grep -E 'CUDA0 KV buffer size' "$name.log" | tail -1 | grep -oE '[0-9.]+ MiB' || echo 'n/a')
echo "RESULT side=$side ctx=$ctx verdict=$verdict compute=$compute kv=$kv"

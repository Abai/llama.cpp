#!/bin/sh
# Does chunking change the FA kernel's launch geometry, or only how many times
# it launches? Same binary (cap branch), same workload; the only difference is
# GGML_CUDA_FATTN_CONVERT_BYTES, which decides whether the conversion is split.
#   single = 1 GiB cap  -> llama q8_0/q8_0 @ d32768 fits, one launch per FA op
#   chunked = 64 MiB cap (the default) -> ~132 MiB conversion, multiple launches
# Metrics: grid geometry and launch__waves_per_multiprocessor per FA launch.
# CUDA graphs off so ncu sees individual launches.
set -e
R=/home/abai/Projects/abai/online/workspace/llama.cpp
M=/home/abai/Projects/abai/online/models
MODEL=$M/unsloth/Llama-3.1-8B-Instruct-GGUF/Llama-3.1-8B-Instruct-Q4_0.gguf
IMG=llamacpp-bench-dev:12.8.1-gcc14
OUT=$R/cap-src/bench-out/occupancy
mode=$1; bytes=$2
docker run --rm --gpus all --cap-add=SYS_ADMIN --entrypoint /usr/local/cuda/bin/ncu \
  -v "$R:$R" -v "$M:$M" -e GGML_CUDA_DISABLE_GRAPHS=1 -e GGML_CUDA_FATTN_CONVERT_BYTES="$bytes" "$IMG" \
  --kernel-name regex:flash_attn --launch-count 12 --csv \
  --metrics launch__grid_size,launch__block_size,launch__waves_per_multiprocessor,sm__throughput.avg.pct_of_peak_sustained_elapsed \
  "$R/cap-src/build-cap/bin/llama-bench" -m "$MODEL" -fa 1 -ngl 99 -n 0 -p 256 -d 32768 -ub 8 -r 1 \
  -ctk q8_0 -ctv q8_0 > "$OUT/ncu-$mode.csv" 2> "$OUT/ncu-$mode.err"
echo "$mode: $(grep -c ',' "$OUT/ncu-$mode.csv" 2>/dev/null || echo 0) csv lines"

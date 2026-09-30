#!/bin/sh
# Convert gemma-2b + DeepSeek-V2-Lite safetensors to Q4_0 GGUFs.
# Idempotent: skips any model whose output GGUF already exists (e.g. copied
# from the other machine). Run from anywhere; requires the branch build
# (llama-quantize) and python3 with the convert requirements.
#
#   MODEL_DIR=/home/abai/models sh convert-models.sh
#
# Expects safetensors repos at:
#   $MODEL_DIR/unsloth/gemma-2b
#   $MODEL_DIR/deepseek-ai/DeepSeek-V2-Lite   (or $MODEL_DIR/DeepSeek-V2-Lite)
# Outputs:
#   $MODEL_DIR/converted/gemma-2b-Q4_0.gguf
#   $MODEL_DIR/converted/DeepSeek-V2-Lite-Q4_0.gguf
set -e
cd "$(dirname "$0")/../.."
ROOT=$(pwd)
MODEL_DIR=${MODEL_DIR:-/home/abai/models}
QUANTIZE=$ROOT/build-cap/bin/llama-quantize
OUT=$MODEL_DIR/converted
mkdir -p "$OUT"

[ -x "$QUANTIZE" ] || { echo "ERROR: build llama-quantize first (cmake --build build-cap --target llama-quantize)"; exit 1; }

PY=${PY:-python3}
deps_checked=0
ensure_deps() {
    [ $deps_checked = 1 ] && return
    if ! $PY -c 'import gguf, safetensors, transformers, sentencepiece' 2>/dev/null; then
        if $PY -m pip --version >/dev/null 2>&1; then
            $PY -m pip install -r "$ROOT/requirements/requirements-convert_hf_to_gguf.txt"
        elif command -v uv >/dev/null 2>&1; then
            uv venv /tmp/convert-venv >/dev/null 2>&1 || true
            uv pip install --python /tmp/convert-venv/bin/python -r "$ROOT/requirements/requirements-convert_hf_to_gguf.txt"
            PY=/tmp/convert-venv/bin/python
        else
            echo "ERROR: no pip and no uv available for convert deps"; exit 1
        fi
    fi
    deps_checked=1
}

convert_one() { # src_dir out_base
    src=$1; base=$2
    if [ -s "$OUT/$base-Q4_0.gguf" ]; then
        echo "$base-Q4_0.gguf exists, skipping"
        return
    fi
    [ -d "$src" ] || { echo "ERROR: $src not found"; exit 1; }
    ensure_deps
    if [ ! -s "$OUT/$base-f16.gguf" ]; then
        $PY "$ROOT/convert_hf_to_gguf.py" "$src" --outtype f16 --outfile "$OUT/$base-f16.gguf"
    fi
    "$QUANTIZE" "$OUT/$base-f16.gguf" "$OUT/$base-Q4_0.gguf" Q4_0
    rm -f "$OUT/$base-f16.gguf"
    echo "$base-Q4_0.gguf done"
}

gemma_src=$MODEL_DIR/unsloth/gemma-2b
[ -d "$gemma_src" ] || gemma_src=$MODEL_DIR/gemma-2b
convert_one "$gemma_src" gemma-2b

ds_src=$MODEL_DIR/deepseek-ai/DeepSeek-V2-Lite
[ -d "$ds_src" ] || ds_src=$MODEL_DIR/DeepSeek-V2-Lite
convert_one "$ds_src" DeepSeek-V2-Lite

echo "=== conversions done"

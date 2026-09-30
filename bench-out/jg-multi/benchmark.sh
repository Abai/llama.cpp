#!/bin/sh
# Single entry point for the FA feature-branch benchmarks.
#
#   sh benchmark.sh [--fast|--vram|--tests|--ppl|--kld] [--small|--identity]
#                   [--force] [--reps N] [--models <dir>] [--dry-run]
#                   [--render-only]
#
#   --fast          birds-eye validation sweep, 64 MiB cap only (~30-60 min);
#                   default is the full final-handoff matrix
#   --vram          measure VRAM (CUDA0 compute buffer) instead of
#                   throughput: llama-server context ladder per model/KV
#                   type, master vs branch (+branch32 on q8_0); OOM
#                   outcomes are recorded as results
#   --tests         test-backend-ops validation for both sides (FLASH_ATTN_EXT
#                   op, full suite, compute-sanitizer memcheck on the feature
#                   side; lowered conversion budgets when the feature has the
#                   cap knob) -> <feature>/VALIDATE-<tag>.txt. Needs no models.
#   --ppl           wikitext-2 perplexity per model, KV f16/q8_0/q4_0, both
#                   sides (PPL_CTX=32768, PPL_CHUNKS=9) -> <feature>/PPL-<tag>.md
#   --kld           KL divergence of q8_0/q4_0 KV vs the master f16-KV run for
#                   one model (KLD_MODEL, default llama; KLD_CTX=4096,
#                   KLD_CHUNKS=4), both sides -> same PPL-<tag>.md
#   --reps N        llama-bench repetitions per rung (default 3); part of the
#                   rung stamp, so a different N re-measures
#   --dry-run       print the rungs/commands that would run, execute nothing
#   --small         measure only the rungs whose kernels this feature actually
#                   changes (see "sweep partitions" below)
#   --identity      measure only the complementary rungs, whose kernels come
#                   from sources the feature leaves alone and which must
#                   therefore come out at 1.00; --small and --identity
#                   together are exactly the unpartitioned sweep
#   --force         ignore existing data: drop whatever THIS run targets (the
#                   chosen partition, or the whole sweep) before measuring, so
#                   every targeted rung is re-measured and its master/branch
#                   pair is fresh. Honours --dry-run (lists, deletes nothing).
#   --render-only   re-render the tables from the data already on disk and
#                   measure nothing (the two runners are hard-guarded). Use
#                   after merging another machine's rungs, or after a change
#                   to the renderer itself
#   --restamp       stamp every unstamped or legacy-stamped rung as measured
#                   from the current sources under the pre-2026-09-14 reps
#                   rule (1 at ub <= 8, else 3); measures nothing (migration)
#   --only <list>   restrict the run to these models (space-separated subset
#                   of: gemma llama deepseek qwen)
#   --models <dir>  directory containing the model GGUFs (searched
#                   recursively); default: ./models relative to the
#                   invoking directory. HOST-side path semantics: it is
#                   bind-mounted by the docker daemon (under dind, pass the
#                   host path; the invoking container never needs to see it)
#
# Generic over feature branches: run from a checkout of benchmark/<feature>
# (or feature/<feature>); the script resolves feature/<name>, its merge-base
# with master (= the master side), builds both sides in a container derived
# from .devops/cuda.Dockerfile, auto-detects GPU name/arch/VRAM, verifies
# the models the VRAM admits are present, runs the sweep (resumable: rungs
# are reused when the measuring binaries are unchanged, via .sql.bin
# fingerprints), and renders GitHub-pastable markdown to
# <feature>/TABLES-<gpu>-sm<cc>-<vram>GB.md.
#
# Container/toolchain image: ONE image for everything - builds, benchmark
# rungs, model verification, table-gen deps (and future VRAM sweeps). It
# extends .devops/cuda.Dockerfile's build-stage environment (same devel
# base, gcc pin via CC/CXX/CUDAHOSTCXX, cmake, git, python3) without
# running upstream's full packaging compile. Bench builds use:
#   GGML_CUDA=ON + CMAKE_CUDA_ARCHITECTURES=<detected> + Release
#   + GGML_CUDA_FA_ALL_QUANTS=ON: without it only the default K/V quant
#   subset compiles and other combos silently fall back to CPU (upstream
#   #15454) - load-bearing for quantized-KV FA benches. The Dockerfile's
#   own GGML_BACKEND_DL/CPU_ALL_VARIANTS/TESTS=OFF are packaging flags,
#   intentionally not used here.
# Benchmark rungs execute inside this same image (runtime libs pinned;
# only driver + GPU state are host-determined; parity-verified vs native
# execution, 0.06% delta, 2026-09-04).
#
# Models (required when the GPU's VRAM admits them, else skipped):
#   gemma 2B Q4_0    (~1.6 GiB)             >= 3 GiB
#   llama 8B Q4_0    (~4.4 GiB + KV)        >= 8 GiB   (f16 control at d16384)
#   deepseek2 16B    (8.29 GiB, MLA)        >= 11 GiB
#   qwen35 27B Q4_0  (14.9 GiB)             >= 20 GiB
#
# Host requirements: docker (with GPU access), nvidia-smi, git, POSIX
# sh/coreutils; python3 optional (render falls back to the toolchain
# image, which includes it when built fresh). Nothing is installed on the
# host; all builds and benchmark rungs run containerized.
# Env overrides: GPU_TAG, FEATURE.
set -e

usage() {
    sed -n '2,49p' "$0" | sed 's/^# \{0,1\}//'
}

INVOKE_PWD=$(pwd)
cd "$(dirname "$0")"
ROOT=$(cd ../.. && pwd)

FAST=0
VRAM=0
TESTS=0
PPL=0
KLD=0
DRY=0
RESTAMP=0
REPS=3
ONLY=""
MODELS_DIR=""
PARTITION=all
FORCE=0
RENDER=0
while [ $# -gt 0 ]; do
    case "$1" in
        --fast)     FAST=1;;
        --vram)     VRAM=1;;
        --tests)    TESTS=1;;
        --ppl)      PPL=1;;
        --kld)      KLD=1;;
        --dry-run)  DRY=1;;
        --restamp)  RESTAMP=1;;
        --render-only) RENDER=1;;
        --small)    PARTITION=small;;
        --identity) PARTITION=identity;;
        --force)    FORCE=1;;
        --only)     shift; ONLY=${1:?--only needs a model list};;
        --only=*)   ONLY=${1#--only=};;
        --reps)     shift; REPS=${1:?--reps needs a number};;
        --reps=*)   REPS=${1#--reps=};;
        --models)   shift; MODELS_DIR=${1:?--models needs a directory};;
        --models=*) MODELS_DIR=${1#--models=};;
        -h|--help)  usage; exit 0;;
        *) echo "ERROR: unknown argument '$1'"; usage; exit 1;;
    esac
    shift
done
[ $FAST = 1 ] && [ $VRAM = 1 ] && { echo "ERROR: --fast and --vram are mutually exclusive"; exit 1; }
if [ "$PARTITION" != all ]; then
    [ $VRAM = 0 ] || { echo "ERROR: the partitions apply to throughput rungs, not --vram"; exit 1; }
    { [ $TESTS = 0 ] && [ $PPL = 0 ] && [ $KLD = 0 ]; } || { echo "ERROR: the partitions apply to throughput rungs only"; exit 1; }
fi
case "$REPS" in ''|*[!0-9]*|0) echo "ERROR: --reps needs a positive integer"; exit 1;; esac
if [ $TESTS = 1 ] || [ $PPL = 1 ] || [ $KLD = 1 ]; then
    [ $FAST = 0 ] && [ $VRAM = 0 ] || { echo "ERROR: --tests/--ppl/--kld cannot combine with --fast/--vram"; exit 1; }
fi
PPL_CTX=${PPL_CTX:-32768}; PPL_CHUNKS=${PPL_CHUNKS:-9}
# perplexity context per model = min(PPL_CTX, trained context): gemma-2b is
# trained at 8192 and produces garbage beyond it (PPL ~4900 on master f16 at
# 32768, NaN on the fused q8_0 path there); the other three exceed 32768.
ppl_ctx_for() { case $1 in gemma) [ "$PPL_CTX" -gt 8192 ] && echo 8192 || echo "$PPL_CTX";; *) echo "$PPL_CTX";; esac; }
KLD_CTX=${KLD_CTX:-4096};  KLD_CHUNKS=${KLD_CHUNKS:-4}; KLD_MODEL=${KLD_MODEL:-}
if [ -z "$MODELS_DIR" ]; then
    MODELS_DIR="$INVOKE_PWD/models"
fi
case "$MODELS_DIR" in /*) ;; *) MODELS_DIR="$INVOKE_PWD/$MODELS_DIR";; esac

# ---------------------------------------------------------------- GPU facts
GPUNAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
[ -n "$GPU_TAG" ] || GPU_TAG=$(echo "$GPUNAME" | tr -cd 'A-Za-z0-9' | sed 's/NVIDIAGeForce//')
CC_SM=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
TOTMIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
VRAM_GB=$(( (TOTMIB + 512) / 1024 ))
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
LONGTAG="$GPU_TAG-sm$CC_SM-${VRAM_GB}GB"
echo "GPU: $GPUNAME (sm_$CC_SM, ${VRAM_GB} GB, driver $DRIVER) -> tag $GPU_TAG, tables $LONGTAG"

# ------------------------------------------------- toolchain image (.devops)
DF=$ROOT/.devops/cuda.Dockerfile
CUDA_VER=$(sed -n 's/^ARG CUDA_VERSION=//p' "$DF" | head -1)
UBU_VER=$(sed -n 's/^ARG UBUNTU_VERSION=//p' "$DF" | head -1)
GCC_VER=$(sed -n 's/^ARG GCC_VERSION=//p' "$DF" | head -1)
DEV_BASE="nvidia/cuda:${CUDA_VER}-devel-ubuntu${UBU_VER}"
IMG="llamacpp-bench-dev:${CUDA_VER}-gcc${GCC_VER}"
# (Re)build when missing or when it predates a required component - the
# same tag is upgraded in place, so there is always exactly one image.
if ! docker run --rm --entrypoint sh "$IMG" -c 'command -v cmake && command -v git && command -v python3' > /dev/null 2>&1; then
    echo "building toolchain image $IMG (extends the .devops/cuda.Dockerfile build-stage environment: $DEV_BASE + gcc-$GCC_VER/cmake/git/python3)"
    docker build -t "$IMG" - <<DOCKEREOF
FROM $DEV_BASE
RUN apt-get update && apt-get install -y gcc-$GCC_VER g++-$GCC_VER build-essential cmake git python3 libgomp1
ENV CC=gcc-$GCC_VER CXX=g++-$GCC_VER CUDAHOSTCXX=g++-$GCC_VER
DOCKEREOF
fi

# ------------------------------------------- model selection + verification
# Fail fast (before the build steps): every model the VRAM admits must be
# present; models the GPU cannot fit are skipped with a notice. The check
# runs inside the toolchain container with the same mount the bench rungs
# use, so --models has pure host-path semantics.
# (No separate existence check: docker auto-creates missing bind-mount
# paths, so a wrong dir simply yields every model missing below.)
# --entrypoint bypasses the NVIDIA base image banner, which prints to stdout
pick() { docker run --rm --entrypoint find -v "$MODELS_DIR:$MODELS_DIR" "$IMG" "$MODELS_DIR" -name "$1" 2>/dev/null | head -1; }
MODELS=""
MISSING=""
add_model() { # name min_gib file
    if [ "$VRAM_GB" -ge "$2" ]; then
        f=$(pick "$3")
        if [ -n "$f" ]; then MODELS="$MODELS $1"; eval "MODEL_$1=\$f"
        else MISSING="$MISSING $3"; fi
    else
        echo "SKIP $1: needs >= $2 GiB VRAM, have ${VRAM_GB}"
    fi
}
add_model gemma    3  'gemma-2b-Q4_0.gguf'
add_model llama    8  'Llama-3.1-8B-Instruct-Q4_0.gguf'
add_model deepseek 11 'DeepSeek-V2-Lite-Q4_0.gguf'
add_model qwen     20 'Qwen3.6-27B-Q4_0.gguf'
TESTS_ONLY=0
if { [ $TESTS = 1 ] || [ $RESTAMP = 1 ] || [ $RENDER = 1 ]; } && [ $PPL = 0 ] && [ $KLD = 0 ]; then TESTS_ONLY=1; fi
if [ -n "$MISSING" ] && [ $TESTS_ONLY = 0 ]; then
    echo "ERROR: this GPU (${VRAM_GB} GB) requires models that were not found under $MODELS_DIR:"
    for f in $MISSING; do echo "  - $f"; done
    echo "Place them there (subdirectories are fine), or pass --models <dir>."
    echo "Note: the path must exist on the DOCKER HOST (dind mounts host paths)."
    exit 1
fi
[ -n "$MODELS" ] || [ $TESTS_ONLY = 1 ] || { echo "ERROR: no runnable models for ${VRAM_GB} GB VRAM"; exit 1; }
echo "models:$MODELS (from $MODELS_DIR)"
if [ -n "$ONLY" ]; then
    sel=""; for m in $MODELS; do echo " $ONLY " | grep -q " $m " && sel="$sel $m"; done
    MODELS=$sel; echo "models restricted by --only:$MODELS"
    [ -n "$MODELS" ] || { echo "ERROR: --only selected none of the admitted models"; exit 1; }
fi

# ------------------------------------------------------- git sides + builds
BRANCH=$(git -C "$ROOT" rev-parse --abbrev-ref HEAD)
FEATURE=${FEATURE:-$(echo "$BRANCH" | sed 's|^benchmark/||')}
git -C "$ROOT" rev-parse --verify "feature/$FEATURE" > /dev/null 2>&1 || {
    echo "ERROR: cannot resolve feature/$FEATURE from branch '$BRANCH' (override with FEATURE=)"; exit 1; }
FEAT_COMMIT=$(git -C "$ROOT" rev-parse --short "feature/$FEATURE")
MBASE=$(git -C "$ROOT" merge-base "feature/$FEATURE" master)
MBASE_SHORT=$(git -C "$ROOT" rev-parse --short "$MBASE")
git -C "$ROOT" merge-base --is-ancestor "feature/$FEATURE" HEAD || {
    echo "ERROR: current checkout does not contain feature/$FEATURE"; exit 1; }
echo "feature/$FEATURE @ $FEAT_COMMIT vs master merge-base $MBASE_SHORT"

if [ ! -d "$ROOT/master-src" ]; then
    # a synced .git may carry a stale worktree registration for master-src
    git -C "$ROOT" worktree prune 2>/dev/null || true
    git -C "$ROOT" worktree add --detach master-src "$MBASE" || true
fi
[ "$(git -C "$ROOT/master-src" rev-parse HEAD)" = "$(git -C "$ROOT" rev-parse "$MBASE")" ] || {
    echo "ERROR: master-src worktree is not at merge-base $MBASE_SHORT (fix manually to protect existing data)"; exit 1; }

CMAKE_FLAGS="-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=$CC_SM -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA_FA_ALL_QUANTS=ON"

# Rung stamps (provenance sidecars .sql.bin / .log.bin) are source-based:
# "src:<key> r<reps>" (throughput) or "src:<key>" (VRAM probes), key = sha256
# over the built tree (feature commit tree, or the master merge-base tree),
# the cmake flags without the arch (the GPU is part of the rung name), the
# toolchain versions from .devops/cuda.Dockerfile AND the toolchain image ID.
# The image ID is load-bearing: the tag is upgraded in place, so a rebuilt
# image silently changes the compiler and the runtime libs. Rungs measured
# with different binaries are NOT comparable - a master/branch ratio across
# an image rebuild carried a 0.5-4% bias (2026-09-15, biggest on the fastest
# model, i.e. host-side launch overhead). Any source, flag, toolchain, image
# or repetition change makes the rung stale.
CMAKE_FLAGS_NOARCH=$(echo "$CMAKE_FLAGS" | sed 's/ -DCMAKE_CUDA_ARCHITECTURES=[0-9;]*//')
IMG_ID=$(docker image inspect -f '{{.Id}}' "$IMG" | cut -c8-23)
srckey() { printf 'tree=%s flags=%s cuda=%s gcc=%s img=%s\n' "$1" "$CMAKE_FLAGS_NOARCH" "$CUDA_VER" "$GCC_VER" "$IMG_ID" | sha256sum | cut -c1-16; }
MKEY=src:$(srckey "$(git -C "$ROOT" rev-parse "$MBASE^{tree}")")
BKEY=src:$(srckey "$(git -C "$ROOT" rev-parse "feature/$FEATURE^{tree}")")
legacy_reps_for() { [ "$1" -le 8 ] && echo 1 || echo 3; }  # reps rule before 2026-09-14, used by --restamp only

TARGETS="llama-bench llama-server"
[ $TESTS = 1 ] && TARGETS="$TARGETS test-backend-ops"
if [ $PPL = 1 ] || [ $KLD = 1 ]; then TARGETS="$TARGETS llama-perplexity"; fi
build_side() { # builddir srcdir
    bd=$1; src=$2
    if [ $DRY = 1 ] || [ $RESTAMP = 1 ]; then echo "skip build $bd <- $src (targets: $TARGETS)"; return; fi
    if [ -f "$ROOT/$bd/CMakeCache.txt" ] && ! grep -q "CMAKE_CUDA_ARCHITECTURES.*=$CC_SM\$" "$ROOT/$bd/CMakeCache.txt"; then
        echo "$bd: cached for a different arch, reconfiguring"; rm -rf "$ROOT/$bd"
    fi
    docker run --rm --entrypoint sh -v "$ROOT:$ROOT" -w "$ROOT" "$IMG" -c \
        "cmake -B $bd -S $src $CMAKE_FLAGS > /dev/null && cmake --build $bd --target $TARGETS -j\$(nproc) 2>&1 | tail -1"
}
if [ $RENDER = 0 ]; then
    echo "== building master side (build-master <- master-src)"; build_side build-master master-src
    echo "== building feature side (build-cap <- repo root)";   build_side build-cap .
fi

MB=$ROOT/build-master/bin/llama-bench
BB=$ROOT/build-cap/bin/llama-bench

# Output layout: feature-side rungs under <feature>/, master-side rungs under
# master-<mbase>/ shared across ALL feature branches with the same merge-base
# (the master binary is identical, so fingerprint-gated reuse is exact).
MDIR="master-$MBASE_SHORT"
FDIR="$FEATURE"
mkdir -p "$MDIR" "$FDIR"

# ------------------------------------------------------------- provenance
# Throughput tables can hold rows merged in from another machine (see
# merge-laptop.sh), but the header used to be rendered from the LOCAL GPU
# only - so a merged table announced one GPU while containing two. Each run
# drops a one-line sidecar next to its data; the renderer emits one header
# line per GPU that actually has rungs here. Keyed on GPU_TAG, which is the
# rung-name prefix, so "has rungs" is an exact file test.
write_provenance() {
    printf '%s|sm_%s|%s|%s|%s\n' "$GPUNAME" "$CC_SM" "$VRAM_GB" "$DRIVER" \
        "-DCMAKE_CUDA_ARCHITECTURES=$CC_SM" > "$FDIR/PROVENANCE-$GPU_TAG.txt"
}
# The header date must describe when the rows were MEASURED, not when this
# render happened - re-rendering must not appear to re-date the data.
# VRAM probes write .log files that filter-vram-logs.sh later rewrites, which
# resets their mtimes - so the VRAM date is pinned in a sidecar at measure time
# rather than inferred. Throughput .sql files are never rewritten, so those
# still derive from mtime.
measured_span() { # [name-glob, default *.sql] -> YYYY-MM-DD  or  YYYY-MM-DD..YYYY-MM-DD
    _g=${1:-*.sql}
    case "$_g" in *vram*) [ -s "$FDIR/MEASURED-vram.txt" ] && { cat "$FDIR/MEASURED-vram.txt"; return 0; };; esac
    _d=$(find "$FDIR" "$MDIR" -maxdepth 1 -name "$_g" -printf '%TY-%Tm-%Td\n' 2>/dev/null | sort -u)
    [ -n "$_d" ] || { date -I; return 0; }
    _a=$(echo "$_d" | head -1); _b=$(echo "$_d" | tail -1)
    if [ "$_a" = "$_b" ]; then echo "$_a"; else echo "$_a..$_b"; fi
    return 0
}
gpu_header_lines() { # -> the "**GPU**:" header block for the throughput table
    _tags=""
    for _p in "$FDIR"/PROVENANCE-*.txt; do
        [ -e "$_p" ] || continue
        _tag=$(basename "$_p"); _tag=${_tag#PROVENANCE-}; _tag=${_tag%.txt}
        ls "$FDIR/$_tag"-*.sql > /dev/null 2>&1 || continue   # no rungs -> not in the table
        _tags="$_tags $_tag"
    done
    # Single GPU: keep the historical line verbatim.
    if [ "$(echo $_tags | wc -w)" -le 1 ]; then
        echo "**GPU**: $GPUNAME (sm_$CC_SM, ${VRAM_GB} GB VRAM, driver $DRIVER) | \\"
        return 0
    fi
    echo "**GPUs** (rows from both machines, merged into one table; the GPU column selects): | \\"
    for _tag in $_tags; do
        IFS='|' read -r _nm _sm _vr _dr _fl < "$FDIR/PROVENANCE-$_tag.txt"
        echo "&nbsp;&nbsp;&bull; $_nm ($_sm, ${_vr} GB VRAM, driver $_dr), \`$_fl\` | \\"
    done
    return 0
}

# ----------------------------------------------------------------- restamp
if [ $RESTAMP = 1 ]; then
    n=0
    for f in "$MDIR"/*.sql "$FDIR"/*.sql "$MDIR"/*-vram-*.log "$FDIR"/*-vram-*.log; do
        [ -f "$f" ] || continue
        old=""; [ -f "$f.bin" ] && old=$(cat "$f.bin")
        case "$old" in src:*) continue;; esac
        case "$f" in "$MDIR"/*) key=$MKEY;; *) key=$BKEY;; esac
        case "$f" in
            *.sql) ub=$(echo "$f" | sed -n 's/.*-ub\([0-9]*\)\.sql$/\1/p'); [ -n "$ub" ] || ub=512
                   echo "$key r$(legacy_reps_for $ub)" > "$f.bin";;
            *.log) echo "$key" > "$f.bin";;
        esac
        n=$((n+1))
    done
    echo "restamped $n rungs with the source keys of the current trees (master $MKEY, feature $BKEY) under the pre-2026-09-14 reps rule; nothing measured"
    exit 0
fi

# Cap-lever sides (branch32, the --vram branch32 probes) only make sense
# when the feature under test implements the conversion-cap env knob;
# detect from the checked-out source (e.g. the fused-dequant branch does
# not have it - measuring branch32 there would duplicate branch).
if grep -rq GGML_CUDA_FATTN_CONVERT_BYTES "$ROOT/ggml/src" 2>/dev/null; then
    CAP_SIDES="branch32"
    FEATURE_KIND=cap
elif grep -q ggml_cuda_flash_attn_ext_mma_f16_fused_kv "$ROOT/ggml/src/ggml-cuda/fattn.cu" 2>/dev/null; then
    CAP_SIDES=""
    FEATURE_KIND=fused
    echo "note: feature has no GGML_CUDA_FATTN_CONVERT_BYTES - cap-lever sides disabled"
else
    CAP_SIDES=""
    FEATURE_KIND=other
    echo "note: feature has no GGML_CUDA_FATTN_CONVERT_BYTES - cap-lever sides disabled"
fi

# --------------------------------------------------------- sweep partitions
# The sweep splits into two complementary halves. A feature-side rung is
# "identity" when the kernel its launches select is compiled from sources this
# feature does not touch, so the GPU executes master's code:
#
#   f16 KV       neither feature changes the f16 instances (SASS-gated: fused
#                21/21 objects, cap 1068 kernels, 0 diff)
#   fattn-vec    neither feature changes the vec kernel, and vec is what
#                ggml_cuda_get_best_fattn_kernel picks for Q->ne[1] <= 1
#                (cc < 890) or <= 2 (cc >= 890) - i.e. the ub1/ub2 prefill
#                rungs and EVERY decode rung. Requires DKQ <= 256, so
#                deepseek (576) is excluded and always lands on MMA.
#   fused only   head dims outside {128,256} miss the fused predicate
#                ggml_cuda_flash_attn_ext_mma_f16_fused_kv entirely, as does
#                any K/V pair other than q8_0/q8_0, q4_0/q4_0, q8_0/q4_0
#   cap only     the conversion fits under the cap, so kv_per_chunk == 0 and
#                the write_partial=false instances run
#
# This is a claim about DEVICE code only, which is why the identity half is
# MEASURED rather than asserted: host-side dispatch still differs (cap
# evaluates ggml_cuda_flash_attn_ext_get_f16_extra_data on every MMA op to
# choose write_partial; both features change get_alloc_size, hence allocation
# size and CUDA-graph shape), and the launch-bound ub1/ub2/decode rungs are
# exactly where per-op host overhead would surface. The split only decides
# what runs first: --small carries the rungs where a real difference is
# expected, --identity the controls that must land at 1.00.
#
# Geometry is f16 bytes per KV position for ONE layer (K+V), read from the
# llama-server logs of the VRAM ladders (n_embd_k_gqa/n_embd_v_gqa). DeepSeek
# is MLA - V is a view of K, so only K is converted and only K is counted.
model_geo() { # model -> "DKQ DV BYTES_PER_KV_POSITION"
    case $1 in
        gemma)    echo "256 256 1024";;
        llama)    echo "128 128 4096";;
        qwen)     echo "256 256 4096";;
        deepseek) echo "576 512 1152";;
        *)        echo "";;
    esac
}
VECMAX=1; [ "$CC_SM" -ge 89 ] && VECMAX=2   # GGML_CUDA_CC_ADA_LOVELACE

rung_identical() { # side model depth ub pp tg ctk ctv -> 0 when master's code runs
    _side=$1; _mdl=$2; _d=$3; _ub=$4; _pp=$5; _tg=$6; _ctk=$7; _ctv=$8
    [ "$_side" = master ] && return 1
    set -- $(model_geo "$_mdl")
    [ $# -eq 3 ] || return 1                       # unknown model: measure it
    _dkq=$1; _dv=$2; _bpk=$3
    [ "$_ctk" = f16 ] && [ "$_ctv" = f16 ] && return 0
    _q1=$_ub; { [ "$_pp" = 0 ] && [ "$_tg" != 0 ]; } && _q1=1
    if [ "$_dkq" -le 256 ] && [ $((_dkq % 64)) -eq 0 ] && [ "$_dkq" != 192 ] && [ "$_q1" -le "$VECMAX" ]; then
        return 0
    fi
    case $FEATURE_KIND in
        fused)
            if [ "$_dkq" != 128 ] || [ "$_dv" != 128 ]; then
                if [ "$_dkq" != 256 ] || [ "$_dv" != 256 ]; then return 0; fi
            fi
            case "$_ctk-$_ctv" in q8_0-q8_0|q4_0-q4_0|q8_0-q4_0) return 1;; *) return 0;; esac;;
        cap)
            _cap=67108864; [ "$_side" = branch32 ] && _cap=33554432
            [ $((_bpk * (_d + _pp + _tg))) -le "$_cap" ] && return 0
            return 1;;
    esac
    return 1
}

# Partitioning is per CELL, not per side: a config's master, branch and (where it
# exists) branch32 must all land in the SAME partition. Classifying each side on
# its own splits a cell across two runs and leaves one of its pairs straddling
# them - gemma and deepseek at d32768 fit under the 64 MiB cap (branch identical
# to master) but chunk at 32 MiB (branch32 differs), which sent master to one
# partition and branch to the other and produced 50 pairs ~14 h apart.
# A cell is "identity" only when EVERY feature side of it is identical to master.
rung_partition() { # side model depth ub pp tg ctk ctv -> small|identity
    _s=$1; shift
    if rung_identical branch "$@"; then
        if [ -n "$CAP_SIDES" ] && ! rung_identical branch32 "$@"; then echo small; else echo identity; fi
    else
        echo small
    fi
}
PART_OTHER=0
PART_FORCED=0

MSRV=$ROOT/build-master/bin/llama-server
BSRV=$ROOT/build-cap/bin/llama-server

# ------------------------------------------------------------------ runner
run_lb() { # side name reps model_file depth ub pp tg extra...
    # --render-only: re-render the tables from existing data. Guarding the two
    # runners (rather than the sweep bodies) makes it total - no path can measure.
    [ $RENDER = 1 ] && return 0
    side=$1; name=$2; reps=$3; mf=$4; depth=$5; ub=$6; pp=$7; tg=$8; shift 8
    mdl=${name%%-*}
    bin=$MB; cap=""; key=$MKEY; dir=$MDIR
    case $side in
        branch)   bin=$BB; key=$BKEY; dir=$FDIR;;
        branch32) bin=$BB; key=$BKEY; dir=$FDIR; cap=33554432;;
    esac
    name="$dir/$GPU_TAG-${name}"
    # KV types come in as trailing llama-bench arguments; absent means f16.
    ctk=f16; ctv=f16; prev=""
    for a in "$@"; do
        case "$prev" in -ctk) ctk=$a;; -ctv) ctv=$a;; esac
        prev=$a
    done
    if [ "$PARTITION" != all ]; then
        # NB: an explicit 'return 0'. Under set -e a bare 'return' inherits the
        # status of the failed test above it and aborts the whole run.
        part=$(rung_partition "$side" "$mdl" "$depth" "$ub" "$pp" "$tg" "$ctk" "$ctv")
        if [ "$part" != "$PARTITION" ]; then PART_OTHER=$((PART_OTHER+1)); return 0; fi
    fi
    stamp="$key r$reps"
    if [ $FORCE = 1 ]; then
        # this rung is targeted by this run, so its old data goes regardless of
        # the stamp: that is the only way to guarantee a fresh master/branch pair
        if [ -s "$name.sql" ] || [ -f "$name.sql.bin" ]; then
            PART_FORCED=$((PART_FORCED+1))
            if [ $DRY = 1 ]; then echo "$name force (existing data would be dropped)"
            else rm -f "$name.sql" "$name.sql.bin" "$name.err"; fi
        fi
    elif [ -s "$name.sql" ]; then
        # missing sidecar counts as STALE: provenance unknown
        if [ -f "$name.sql.bin" ] && [ "$(cat "$name.sql.bin")" = "$stamp" ]; then
            echo "$name skipped"; return 0
        fi
        echo "$name stale (source, flags, toolchain or repetitions changed, or unstamped), re-measuring"
    fi
    if [ $DRY = 1 ]; then echo "DRY: $name  llama-bench -r $reps -ub $ub -d $depth -p $pp -n $tg $*"; return; fi
    # Execute inside the same toolchain image the binaries were built with:
    # pins the runtime libs (cudart/cublas/glibc/gomp) to the build
    # environment; only driver + GPU state remain host-determined.
    # Parity-verified vs native execution (0.06% delta, 2026-09-04).
    envargs=""; [ -n "$cap" ] && envargs="-e GGML_CUDA_FATTN_CONVERT_BYTES=$cap"
    docker run --rm --gpus all --entrypoint "$bin" $envargs -v "$ROOT:$ROOT" -v "$MODELS_DIR:$MODELS_DIR" "$IMG" \
        -m "$mf" -fa 1 -ngl 99 -n $tg -p $pp -d $depth -o sql -r $reps -ub $ub "$@" \
        > "$name.sql" 2> "$name.err" || { echo "FAILED: $name (see .err)"; rm -f "$name.sql"; return; }
    echo "$stamp" > "$name.sql.bin"
    echo "$name done $(date -Is)"
}
legacy_reps_for() { [ "$1" -le 8 ] && echo 1 || echo 3; }  # rule used before 2026-09-14 (sidecars without a reps field)
mpath() { eval "echo \$MODEL_$1"; }

# VRAM probe: launch llama-server in the image, wait for readiness or an
# allocation failure (both are results), save the log, stamp the sidecar
# with the llama-server binary fingerprint.
run_vram() { # side name model_file ctx extra(-ctk/-ctv)...
    # --render-only: re-render the tables from existing data. Guarding the two
    # runners (rather than the sweep bodies) makes it total - no path can measure.
    [ $RENDER = 1 ] && return 0
    side=$1; name=$2; mf=$3; ctx=$4; shift 4
    srv=$MSRV; cap=""; key=$MKEY; dir=$MDIR
    case $side in
        branch)   srv=$BSRV; key=$BKEY; dir=$FDIR;;
        branch32) srv=$BSRV; key=$BKEY; dir=$FDIR; cap=33554432;;
    esac
    name="$dir/$GPU_TAG-${name}"
    if [ $FORCE = 1 ]; then
        if [ -s "$name.log" ] || [ -f "$name.log.bin" ]; then
            PART_FORCED=$((PART_FORCED+1))
            if [ $DRY = 1 ]; then echo "$name force (existing data would be dropped)"
            else rm -f "$name.log" "$name.log.bin"; fi
        fi
    elif [ -s "$name.log" ]; then
        if [ -f "$name.log.bin" ] && [ "$(cat "$name.log.bin")" = "$key" ]; then
            echo "$name skipped"; return 0
        fi
        echo "$name stale (source, flags or toolchain changed, or unstamped), re-measuring"
    fi
    if [ $DRY = 1 ]; then echo "DRY: $name  llama-server -c $ctx $*"; return; fi
    envargs=""; [ -n "$cap" ] && envargs="-e GGML_CUDA_FATTN_CONVERT_BYTES=$cap"
    cid=$(docker run -d --gpus all --entrypoint "$srv" $envargs -v "$ROOT:$ROOT" -v "$MODELS_DIR:$MODELS_DIR" "$IMG"         -m "$mf" -c "$ctx" -ngl 99 -fa on -ub 512 --no-warmup -lv 5 "$@")
    ok=0
    i=0
    while [ $i -lt 120 ]; do
        sleep 2; i=$((i+1))
        if docker logs "$cid" 2>&1 | grep -qE "listening on|failed to allocate|out of memory|cudaMalloc failed|error loading model"; then
            ok=1; break
        fi
        [ "$(docker inspect -f '{{.State.Running}}' "$cid" 2>/dev/null)" = "true" ] || { ok=1; break; }
    done
    docker logs "$cid" > "$name.log" 2>&1 || true
    [ $ok = 1 ] || echo "# vram-probe: TIMEOUT" >> "$name.log"
    docker rm -f "$cid" > /dev/null 2>&1 || true
    echo "$key" > "$name.log.bin"
    echo "$name done $(date -Is)"
}

# ------------------------------------------------------------------ sweeps
# ------------------------------------------------------------------- tests
# --tests: reproduce the validation artifact the PR cites. Summary lines go
# to <feature>/VALIDATE-<tag>.txt, full logs next to it (*.log, gitignored).
run_tbo() { # label side [VAR=val ...] -- test-backend-ops args...
    label=$1; side=$2; shift 2
    envargs=""; while [ "$1" != "--" ]; do envargs="$envargs -e $1"; shift; done; shift
    bin=$ROOT/build-master/bin/test-backend-ops; [ $side = branch ] && bin=$ROOT/build-cap/bin/test-backend-ops
    log="$FDIR/$GPU_TAG-tests-$label.log"
    echo "== $label ($side): test-backend-ops $*"
    if [ $DRY = 1 ]; then echo "DRY: docker run --gpus all$envargs $bin $*"; return; fi
    docker run --rm --gpus all --entrypoint "$bin" $envargs -v "$ROOT:$ROOT" "$IMG" "$@" > "$log" 2>&1 || true
    printf '%-46s %s\n' "$label:" "$(grep -E 'tests passed' "$log" | tail -1 | sed 's/^ *//')" >> "$VOUT"
    grep -E '^ *Backend .*: (OK|FAIL)|backends passed' "$log" | sed 's/^ */    /' >> "$VOUT"
}
run_sanitizer() {
    log="$FDIR/$GPU_TAG-tests-sanitizer.log"; bin=$ROOT/build-cap/bin/test-backend-ops
    echo "== sanitizer (branch): compute-sanitizer --tool memcheck test-backend-ops test -o FLASH_ATTN_EXT -b CUDA0"
    if [ $DRY = 1 ]; then echo "DRY: docker run --gpus all compute-sanitizer --tool memcheck --error-exitcode 42 $bin test -o FLASH_ATTN_EXT -b CUDA0"; return; fi
    if docker run --rm --gpus all --entrypoint /usr/local/cuda/bin/compute-sanitizer -v "$ROOT:$ROOT" "$IMG" \
        --tool memcheck --error-exitcode 42 "$bin" test -o FLASH_ATTN_EXT -b CUDA0 > "$log" 2>&1; then rc=0; else rc=$?; fi
    printf '%-46s %s (exit %s)\n' "sanitizer memcheck FLASH_ATTN_EXT (branch):" "$(grep -E 'ERROR SUMMARY' "$log" | tail -1 | sed 's/^=* *//')" "$rc" >> "$VOUT"
}
if [ $TESTS = 1 ]; then
    VOUT="$FDIR/VALIDATE-$LONGTAG.txt"; [ $DRY = 1 ] && VOUT=/dev/null
    {
        echo "test-backend-ops validation: feature/$FEATURE ($FEAT_COMMIT) vs master ($MBASE_SHORT), $(date -I)"
        echo "GPU: $GPUNAME (sm_$CC_SM, ${VRAM_GB} GB, driver $DRIVER); image $IMG; cmake: $CMAKE_FLAGS"
        echo "commands: <side>/bin/test-backend-ops test -o FLASH_ATTN_EXT -b CUDA0 | test -b CUDA0 |"
        echo "          compute-sanitizer --tool memcheck --error-exitcode 42 <branch>/bin/test-backend-ops test -o FLASH_ATTN_EXT -b CUDA0"
        [ -n "$CAP_SIDES" ] && echo "          branch additionally with GGML_CUDA_FATTN_CONVERT_BYTES=1048576 / =262144 (multi-chunk conversion)"
        echo
    } > "$VOUT"
    run_tbo fa-master  master -- test -o FLASH_ATTN_EXT -b CUDA0
    run_tbo fa-branch  branch -- test -o FLASH_ATTN_EXT -b CUDA0
    if [ -n "$CAP_SIDES" ]; then
        run_tbo fa-branch-cap1MiB   branch GGML_CUDA_FATTN_CONVERT_BYTES=1048576 -- test -o FLASH_ATTN_EXT -b CUDA0
        run_tbo fa-branch-cap256KiB branch GGML_CUDA_FATTN_CONVERT_BYTES=262144  -- test -o FLASH_ATTN_EXT -b CUDA0
    fi
    run_tbo full-master master -- test -b CUDA0
    run_tbo full-branch branch -- test -b CUDA0
    run_sanitizer
    echo "=== done: $VOUT"; cat "$VOUT"
fi

# --------------------------------------------------------- perplexity / KLD
# --ppl / --kld: llama-perplexity on wikitext-2 (same source as
# scripts/get-wikitext-2.sh, fetched into the models dir inside the image).
# Summary lines -> <side dir>/<gpu>-<model>-{ppl,kld}-<kv>-c<ctx>-n<chunks>-<side>.txt
# (full logs next to them, gitignored); rendered to <feature>/PPL-<tag>.md.
if [ $PPL = 1 ] || [ $KLD = 1 ]; then
    MPPL=$ROOT/build-master/bin/llama-perplexity; BPPL=$ROOT/build-cap/bin/llama-perplexity
    WT=$MODELS_DIR/wikitext-2-raw/wiki.test.raw
    if [ $DRY = 1 ]; then echo "DRY: ensure $WT (download wikitext-2-raw-v1.zip from huggingface.co/datasets/ggml-org/ci)"; else
        docker run --rm -i --entrypoint python3 -v "$MODELS_DIR:$MODELS_DIR" "$IMG" - "$MODELS_DIR" <<'PYDL'
import os, sys, urllib.request, zipfile
d = sys.argv[1]; f = os.path.join(d, 'wikitext-2-raw', 'wiki.test.raw')
if not os.path.exists(f):
    z = os.path.join(d, 'wikitext-2-raw-v1.zip')
    urllib.request.urlretrieve('https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip', z)
    zipfile.ZipFile(z).extractall(d); os.remove(z)
print('wikitext:', f, os.path.getsize(f), 'bytes')
PYDL
        [ -n "$(docker run --rm --entrypoint sh -v "$MODELS_DIR:$MODELS_DIR" "$IMG" -c "ls $WT 2>/dev/null")" ] || { echo "ERROR: wikitext download failed ($WT missing)"; exit 1; }
    fi
    run_ppl() { # side model kv ctx chunks kind(ppl|kld) extra...
        side=$1; m=$2; kv=$3; ctx=$4; chunks=$5; kind=$6; shift 6
        bin=$MPPL; key=$MKEY; dir=$MDIR
        [ $side = branch ] && { bin=$BPPL; key=$BKEY; dir=$FDIR; }
        kvtag=$(echo "$kv" | tr -d '_')
        name="$dir/$GPU_TAG-$m-$kind-$kvtag-c$ctx-n$chunks-$side"
        stamp="$key c$ctx n$chunks"
        if [ -s "$name.txt" ] && [ -f "$name.txt.bin" ] && [ "$(cat "$name.txt.bin")" = "$stamp" ]; then
            echo "$name skipped"; return
        fi
        if [ $DRY = 1 ]; then echo "DRY: $name  llama-perplexity -c $ctx --chunks $chunks -b 2048 -ub 512 -fa 1 -ngl 99 -ctk $kv -ctv $kv $*"; return; fi
        docker run --rm --gpus all --entrypoint "$bin" -v "$ROOT:$ROOT" -v "$MODELS_DIR:$MODELS_DIR" "$IMG" \
            -m "$(mpath $m)" -f "$WT" -c $ctx --chunks $chunks -b 2048 -ub 512 -fa 1 -ngl 99 -ctk $kv -ctv $kv "$@" \
            > "$name.log" 2>&1 || { echo "FAILED: $name (see .log)"; return; }
        grep -E 'Final estimate|Mean +KLD|Maximum KLD|99\.0% +KLD|Median +KLD|Same top p|Mean ln\(PPL' "$name.log" > "$name.txt" \
            || { echo "FAILED: $name (no final estimate, see .log)"; rm -f "$name.txt"; return; }
        echo "$stamp" > "$name.txt.bin"
        echo "$name done $(date -Is)"
    }
    if [ $PPL = 1 ]; then
        for m in $MODELS; do
            for kv in f16 q8_0 q4_0; do
                for s in master branch; do run_ppl $s $m $kv $(ppl_ctx_for $m) $PPL_CHUNKS ppl; done
            done
        done
    fi
    if [ $KLD = 1 ]; then
        if [ -z "$KLD_MODEL" ]; then
            for c in llama gemma qwen deepseek; do echo " $MODELS " | grep -q " $c " && { KLD_MODEL=$c; break; }; done
        fi
        [ -n "$KLD_MODEL" ] || { echo "ERROR: --kld needs a model (KLD_MODEL=)"; exit 1; }
        base="$MODELS_DIR/kld-base-$KLD_MODEL-f16-c$KLD_CTX-n$KLD_CHUNKS-master.bin"
        if [ $DRY = 1 ]; then echo "DRY: base logits $base <- master f16 KV, $KLD_MODEL, -c $KLD_CTX --chunks $KLD_CHUNKS"; else
            if [ ! -s "$base" ]; then
                docker run --rm --gpus all --entrypoint "$MPPL" -v "$ROOT:$ROOT" -v "$MODELS_DIR:$MODELS_DIR" "$IMG" \
                    -m "$(mpath $KLD_MODEL)" -f "$WT" -c $KLD_CTX --chunks $KLD_CHUNKS -b 2048 -ub 512 -fa 1 -ngl 99 -ctk f16 -ctv f16 \
                    --kl-divergence-base "$base" > "$FDIR/$GPU_TAG-$KLD_MODEL-kld-base.log" 2>&1 \
                    || { echo "ERROR: KLD base run failed (see $FDIR/$GPU_TAG-$KLD_MODEL-kld-base.log)"; exit 1; }
            fi
        fi
        for kv in q8_0 q4_0; do
            for s in master branch; do run_ppl $s $KLD_MODEL $kv $KLD_CTX $KLD_CHUNKS kld --kl-divergence --kl-divergence-base "$base"; done
        done
        # the base lives at the HOST models path: delete it through the image
        [ $DRY = 1 ] || docker run --rm --entrypoint rm -v "$MODELS_DIR:$MODELS_DIR" "$IMG" -f "$base"
    fi
    OUT="$FDIR/PPL-$LONGTAG.md"; [ $DRY = 1 ] && OUT=/dev/null
    {
        echo "## Perplexity: feature/$FEATURE ($FEAT_COMMIT) vs master ($MBASE_SHORT)"
        echo
        echo "**GPU**: $GPUNAME (sm_$CC_SM, ${VRAM_GB} GB VRAM, driver $DRIVER) | \\"
        echo "**Build**: .devops/cuda.Dockerfile devel base ($DEV_BASE, gcc-$GCC_VER), \`$CMAKE_FLAGS\` | \\"
        echo "**Method**: llama-perplexity on wikitext-2 test, -b 2048 -ub 512 -fa 1 -ngl 99, containerized | **Date**: $(date -I)"
        echo
        if command -v python3 > /dev/null 2>&1; then
            python3 mkppltables.py "$MDIR" "$FDIR"
        else
            docker run --rm --entrypoint python3 -v "$ROOT:$ROOT" "$IMG" "$ROOT/bench-out/jg-multi/mkppltables.py" "$MDIR" "$FDIR"
        fi
    } > "$OUT"
    echo "=== done: $OUT (GitHub-pastable)"
fi
if [ $TESTS = 1 ] || [ $PPL = 1 ] || [ $KLD = 1 ]; then exit 0; fi

if [ $VRAM = 1 ]; then
    # VRAM ladders (allocation is deterministic: one probe per config).
    # q8_0/q8_0 + f16/f16 across the context ladder (branch32 on q8_0 to
    # show the cap knob); q4_0/q4_0 + mixed at two spot contexts. Contexts
    # that exceed the GPU record OOM - that is max-context evidence.
    for m in $MODELS; do
        mf=$(mpath $m)
        for c in 16384 32768 65536 131072; do
            for side in master branch $CAP_SIDES; do
                run_vram $side "$m-vram-q80-q80-c$c-$side" "$mf" $c -ctk q8_0 -ctv q8_0
            done
            for side in master branch; do
                run_vram $side "$m-vram-f16-c$c-$side" "$mf" $c
            done
        done
        for c in 32768 131072; do
            for side in master branch; do
                run_vram $side "$m-vram-q40-q40-c$c-$side" "$mf" $c -ctk q4_0 -ctv q4_0
                if [ $m != deepseek ]; then
                    run_vram $side "$m-vram-q80-q40-c$c-$side" "$mf" $c -ctk q8_0 -ctv q4_0
                fi
            done
        done
    done
elif [ $FAST = 1 ]; then
    # Birds-eye sweep, 64 MiB default cap only (sides: master + branch).
    # Feature-agnostic coverage per model: prefill across the microbatch
    # range at depth (vector/small/mid/large), fresh prefill, decode fresh
    # and deep, and an f16-KV control. Rung names are a strict subset of the
    # full sweep, so fast results are reused by later full runs (and vice
    # versa).
    for m in $MODELS; do
        mf=$(mpath $m)
        for ub in 1 8 64 512; do
            r=$REPS
            for s in master branch; do
                run_lb $s "$m-q80-q80-$s-ub$ub" $r "$mf" 32768 $ub 1024 0 -ctk q8_0 -ctv q8_0
            done
        done
        for s in master branch; do
            run_lb $s "$m-q80q80-ladpp-d0-$s"     $REPS "$mf" 0     512 512 0   -ctk q8_0 -ctv q8_0
            run_lb $s "$m-q80q80-ladtg-d0-$s"     $REPS "$mf" 0     512 0   128 -ctk q8_0 -ctv q8_0
            run_lb $s "$m-q80q80-ladtg-d32768-$s" $REPS "$mf" 32768 512 0   128 -ctk q8_0 -ctv q8_0
        done
        d=32768; [ $m = llama ] && d=16384
        for s in master branch; do
            run_lb $s "$m-f16-d$d-$s-ub512" $REPS "$mf" $d 512 1024 0
        done
    done
else
    # Full final-handoff matrix (as agreed): 3 quantized KV combos x ub
    # ladder, 32+64 MiB cap lever, f16 controls, supplemental chunked-regime
    # rows, and pp512/tg128 depth ladders.
    for kv in "q8_0 q8_0" "q4_0 q4_0" "q8_0 q4_0"; do
        set -- $kv; ctk=$1; ctv=$2
        kvtag=$(echo "$ctk-$ctv" | tr -d '_')
        for m in $MODELS; do
            # MLA: V is a view of K -> mixed K/V cache types cannot create a context
            if [ $m = deepseek ] && [ "$ctk" != "$ctv" ]; then continue; fi
            mf=$(mpath $m)
            for ub in 1 2 4 8 16 32 64 128 256 512; do
                r=$REPS
                for s in master branch $CAP_SIDES; do
                    run_lb $s "$m-$kvtag-$s-ub$ub" $r "$mf" 32768 $ub 1024 0 -ctk $ctk -ctv $ctv
                done
            done
        done
    done
    # f16 KV controls (untouched path): prefill at small/mid/large ub and
    # decode at depth, per model, so the neutrality claim rests on the same
    # Q4_0 model set as everything else.
    for m in $MODELS; do
        mf=$(mpath $m)
        d=32768; [ $m = llama ] && d=16384
        for ub in 4 64 512; do
            for s in master branch; do
                run_lb $s "$m-f16-d$d-$s-ub$ub" $REPS "$mf" $d $ub 1024 0
            done
        done
        for s in master branch; do
            run_lb $s "$m-f16-ladtg-d$d-$s" $REPS "$mf" $d 512 0 128
        done
    done
    for m in gemma deepseek; do
        echo "$MODELS" | grep -q "$m" || continue
        mf=$(mpath $m)
        d=131072; [ $m = deepseek ] && d=65536
        for ub in 1 2 4 8 16 32 64 128 256 512; do
            r=$REPS
            for s in master branch $CAP_SIDES; do
                run_lb $s "$m-q80q80-d$d-$s-ub$ub" $r "$mf" $d $ub 1024 0 -ctk q8_0 -ctv q8_0
            done
        done
    done
    for m in $MODELS; do
        mf=$(mpath $m)
        for d in 0 16384 32768 65536; do
            if [ $m = llama ] && [ $d = 65536 ] && [ "$TOTMIB" -lt 12000 ]; then continue; fi
            for s in master branch $CAP_SIDES; do
                run_lb $s "$m-q80q80-ladpp-d$d-$s" $REPS "$mf" $d 512 512 0   -ctk q8_0 -ctv q8_0
                run_lb $s "$m-q80q80-ladtg-d$d-$s" $REPS "$mf" $d 512 0   128 -ctk q8_0 -ctv q8_0
            done
        done
    done
fi

# ------------------------------------------------------------------ render
[ $FORCE = 1 ] && echo "=== --force: dropped $PART_FORCED previously measured rungs targeted by this run"
if [ "$PARTITION" != all ]; then
    echo "=== partition '$PARTITION': $PART_OTHER rungs belong to the other half and were not touched"
    echo "    the rendered table below is COMPLETE only after both --small and --identity have run"
fi
if [ $DRY = 1 ]; then echo "DRY: no render (existing tables untouched)"; exit 0; fi
if [ $VRAM = 1 ]; then
    # pin the measurement date before rendering (see measured_span)
    [ $RENDER = 1 ] || date -I > "$FDIR/MEASURED-vram.txt"
    OUT="$FDIR/TABLES-VRAM-$LONGTAG.md"
    {
        echo "## VRAM: feature/$FEATURE ($FEAT_COMMIT) vs master ($MBASE_SHORT)"
        echo
        echo "**GPU**: $GPUNAME (sm_$CC_SM, ${VRAM_GB} GB VRAM, driver $DRIVER) | \\"
        echo "**Method**: llama-server -ngl 99 -fa on -ub 512 --no-warmup (n_outputs_max = 1), containerized | **Measured**: $(measured_span '*vram*.log') | **Rendered**: $(date -I)"
        echo
        if command -v python3 > /dev/null 2>&1; then
            python3 mkvramtables.py "$MDIR" "$FDIR"
        else
            docker run --rm --entrypoint python3 -v "$ROOT:$ROOT" "$IMG" \
                "$ROOT/bench-out/jg-multi/mkvramtables.py" "$MDIR" "$FDIR"
        fi
    } > "$OUT"
    echo "=== done: $OUT (GitHub-pastable)"
    exit 0
fi
write_provenance
GPU_SUMMARY=$GPUNAME
[ "$(gpu_header_lines | grep -c '&bull;')" -gt 1 ] && GPU_SUMMARY="both GPUs"
OUT="$FDIR/TABLES-$LONGTAG.md"
RAW="$FDIR/.render-$LONGTAG.txt"
if command -v python3 > /dev/null 2>&1; then
    python3 mktables.py "$MDIR" "$FDIR" 2>/dev/null > "$RAW"
else
    docker run --rm --entrypoint python3 -v "$ROOT:$ROOT" "$IMG" \
        "$ROOT/bench-out/jg-multi/mktables.py" "$MDIR" "$FDIR" 2>/dev/null > "$RAW"
fi
{
    echo "## Benchmarks: feature/$FEATURE ($FEAT_COMMIT) vs master ($MBASE_SHORT)"
    echo
    gpu_header_lines
    echo "**Build**: .devops/cuda.Dockerfile devel base ($DEV_BASE, gcc-$GCC_VER), \`-DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA_FA_ALL_QUANTS=ON\` plus the per-GPU arch above | \\"
    side_legend="\`branch\` = feature branch build"
    if [ -n "$CAP_SIDES" ]; then
        side_legend="\`branch\` = default 64 MiB conversion cap, \`branch32\` = GGML_CUDA_FATTN_CONVERT_BYTES=32 MiB"
    fi
    echo "**Measured**: $(measured_span) | **Rendered**: $(date -I) | mode: $([ $FAST = 1 ] && echo 'fast' || echo 'full matrix') | reps: $REPS per rung (llama-bench -r). $side_legend."
    echo
    echo "<details>"
    echo "<summary>Performance ($GPU_SUMMARY)</summary>"
    echo
    grep -v '^WARN' "$RAW"
    echo
    echo "</details>"
} > "$OUT"
if grep -q '^WARN' "$RAW"; then
    echo "=== PROVENANCE WARNINGS (excluded from the table, investigate before using it):"
    grep '^WARN' "$RAW"
fi
rm -f "$RAW"
echo "=== done: $OUT (GitHub-pastable)"

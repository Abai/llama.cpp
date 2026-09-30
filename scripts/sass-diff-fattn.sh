#!/usr/bin/env bash
# SASS-diff gate for the fattn-mma-f16 template instances.
#
# Proves that two builds produce instruction-identical SASS for the
# fattn-mma-f16-instance-*.cu objects, even when kernel (mangled) names differ.
# Kernels are paired by order of appearance inside the same instance object.
#
# Usage:
#   scripts/sass-diff-fattn.sh <A> <B> [GLOB]
#
#   <A>, <B> each one of:
#     - a build dir (contains CMakeCache.txt): existing objects are used as-is
#       (set SASS_DIFF_REBUILD=1 to recompile the selected objects first)
#     - a source dir (contains CMakeLists.txt): configured + built
#     - a git ref: checked out via `git worktree add` and built
#   [GLOB] selects instances: fattn-mma-f16-instance-<GLOB>.cu (default '*').
#
# Env overrides:
#   SASS_DIFF_BUILD_A / SASS_DIFF_BUILD_B  build dir to use for source/ref mode
#   SASS_DIFF_CMAKE_EXTRA                  extra cmake configure flags
#
# Requires: cmake, nvcc, cuobjdump on PATH (and git for ref mode).
# Exit code: 0 = identical, 1 = divergence found, 2 = usage/infra error.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
HELPER="$SCRIPT_DIR/sass-diff-fattn-helper.py"

CMAKE_FLAGS=(-DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON
             -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
             -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
             -DLLAMA_BUILD_TOOLS=OFF ${SASS_DIFF_CMAKE_EXTRA:-})

usage() { echo "usage: $0 <build-dir|src-dir|git-ref> <build-dir|src-dir|git-ref> [instance-glob]" >&2; exit 2; }
[ $# -ge 2 ] || usage
GLOB="${3:-*}"

die() { echo "sass-diff-fattn: $*" >&2; exit 2; }

command -v cuobjdump >/dev/null || die "cuobjdump not on PATH"
command -v python3   >/dev/null || die "python3 not on PATH"

# prepare_tree <label> <arg>; sets BUILD_DIR (global) for the tree
prepare_tree() {
    local label=$1 arg=$2 src="" bdir=""
    local bdir_override
    bdir_override=$(eval echo "\${SASS_DIFF_BUILD_${label^^}:-}")

    if [ -f "$arg/CMakeCache.txt" ]; then
        bdir=$(cd -- "$arg" && pwd)
        if [ "${SASS_DIFF_REBUILD:-0}" = "1" ]; then
            build_objects "$bdir"
        fi
        BUILD_DIR=$bdir
        return
    fi

    if [ -f "$arg/CMakeLists.txt" ]; then
        src=$(cd -- "$arg" && pwd)
        bdir=${bdir_override:-$src/build-sassgate}
    else
        # treat as git ref
        command -v git >/dev/null || die "git needed for ref mode"
        git -C "$REPO_ROOT" rev-parse --verify --quiet "$arg^{commit}" >/dev/null \
            || die "'$arg' is not a build dir, a source dir, or a git ref"
        src="$REPO_ROOT/build-sassgate-$label-src"
        if [ ! -d "$src" ]; then
            git -C "$REPO_ROOT" worktree add --detach "$src" "$arg"
        fi
        bdir=${bdir_override:-$REPO_ROOT/build-sassgate-$label}
    fi

    command -v cmake >/dev/null || die "cmake not on PATH"
    command -v nvcc  >/dev/null || die "nvcc not on PATH"
    if [ ! -f "$bdir/CMakeCache.txt" ]; then
        cmake -S "$src" -B "$bdir" "${CMAKE_FLAGS[@]}" >"$bdir.configure.log" 2>&1 \
            || { tail -20 "$bdir.configure.log" >&2; die "cmake configure failed, see $bdir.configure.log"; }
    fi
    build_objects "$bdir"
    BUILD_DIR=$bdir
}

# compile only the selected instance objects, using the exact commands cmake would use
build_objects() {
    local bdir=$1
    [ -f "$bdir/compile_commands.json" ] || die "$bdir/compile_commands.json missing (configure with -DCMAKE_EXPORT_COMPILE_COMMANDS=ON)"
    local t0 t1
    t0=$(date +%s)
    python3 "$HELPER" commands "$bdir/compile_commands.json" "$GLOB" > "$bdir/.sassgate-cmds" \
        || die "no fattn-mma-f16-instance-$GLOB.cu entries in $bdir/compile_commands.json"
    local n
    n=$(wc -l < "$bdir/.sassgate-cmds")
    echo "[$bdir] compiling $n instance object(s) with -j$(nproc) ..."
    xargs -a "$bdir/.sassgate-cmds" -d '\n' -P "$(nproc)" -I{} sh -c '{}' \
        || die "compilation failed in $bdir"
    t1=$(date +%s)
    echo "[$bdir] build took $((t1 - t0)) s"
}

find_objects() { # <build-dir> -> sorted list of instance objects
    find "$1" -path '*/ggml-cuda.dir/template-instances/*' \
         -name "fattn-mma-f16-instance-$GLOB.cu.o" | sort
}

prepare_tree a "$1"; BUILD_A=$BUILD_DIR
prepare_tree b "$2"; BUILD_B=$BUILD_DIR

mapfile -t OBJS_A < <(find_objects "$BUILD_A")
mapfile -t OBJS_B < <(find_objects "$BUILD_B")
[ "${#OBJS_A[@]}" -gt 0 ] || die "no objects matching fattn-mma-f16-instance-$GLOB.cu.o under $BUILD_A"
[ "${#OBJS_A[@]}" -eq "${#OBJS_B[@]}" ] || die "object count mismatch: ${#OBJS_A[@]} vs ${#OBJS_B[@]}"

for i in "${!OBJS_A[@]}"; do
    [ "$(basename "${OBJS_A[$i]}")" = "$(basename "${OBJS_B[$i]}")" ] \
        || die "object basename mismatch: ${OBJS_A[$i]} vs ${OBJS_B[$i]}"
done

echo "comparing ${#OBJS_A[@]} instance object(s): $BUILD_A vs $BUILD_B"
FAIL=0
for i in "${!OBJS_A[@]}"; do
    name=$(basename "${OBJS_A[$i]}" .cu.o)
    if python3 "$HELPER" diff "${OBJS_A[$i]}" "${OBJS_B[$i]}"; then
        echo "PASS $name"
    else
        echo "FAIL $name"
        FAIL=1
    fi
done

if [ "$FAIL" -ne 0 ]; then
    echo "RESULT: SASS DIVERGENCE DETECTED"
    exit 1
fi
echo "RESULT: all instance objects instruction-identical"

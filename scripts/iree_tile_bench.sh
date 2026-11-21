#!/bin/bash
# Tile-level comparison: ukr-mlir micro-kernels vs IREE's arm64 f32 mmt4d tile
# functions (plan.md, "Level 1"). IREE's ukernels are plain C, so they are
# compiled straight from an IREE checkout; no IREE build is needed.

set -euo pipefail

# Default values
IREE_SRC="external/iree"
LLVM_DIR=""
BUILD_DIR="tmp/iree_tile"
ARCH="firestorm"
VEC_LEN=128
NUM_VECS=32
FMA_UNITS=4
FMA_LATENCY=4
CPU="native"
OUTPUT=""
BENCH_ARGS=()

function usage() {
    echo "Usage: $0 [options] [-- benchmark args]" >&2
    echo "Options:" >&2
    echo "  -i <path>    IREE source checkout (default: external/iree)" >&2
    echo "  -l <path>    Path to LLVM build dir (default: tools on PATH)" >&2
    echo "  -a <string>  Architecture name passed to the generator (default: firestorm)" >&2
    echo "  -v <int>     Vector length in bits (default: 128)" >&2
    echo "  -r <int>     Vector registers (default: 32)" >&2
    echo "  -u <int>     Vector FMA pipes (default: 4, Apple M4)" >&2
    echo "  -t <int>     FMA latency in cycles (default: 4, Apple M4)" >&2
    echo "  -c <cpu>     -mcpu for clang and llc (default: native)" >&2
    echo "  -o <file>    Also write the CSV to this file" >&2
    echo "  -h           Show help" >&2
    echo "Benchmark args (after --): --k=64,256,1024 --modes=hot,warm,cold --reps=15" >&2
    echo "  --sample-ms=20 --arena-mb=256 --check-only" >&2
    exit 1
}

while getopts "i:l:a:v:r:u:t:c:o:h" opt; do
    case $opt in
        i) IREE_SRC=$OPTARG ;;
        l) LLVM_DIR=$OPTARG ;;
        a) ARCH=$OPTARG ;;
        v) VEC_LEN=$OPTARG ;;
        r) NUM_VECS=$OPTARG ;;
        u) FMA_UNITS=$OPTARG ;;
        t) FMA_LATENCY=$OPTARG ;;
        c) CPU=$OPTARG ;;
        o) OUTPUT=$OPTARG ;;
        h) usage ;;
        *) usage ;;
    esac
done
shift $((OPTIND - 1))
BENCH_ARGS=("$@")

LLVM_BIN=""
if [[ -n "$LLVM_DIR" ]]; then
    LLVM_BIN="${LLVM_DIR%%/}/bin/"
fi

UK_SRC="$IREE_SRC/runtime/src/iree/builtins/ukernel"
if [[ ! -d "$UK_SRC" ]]; then
    echo "IREE sources not found at $IREE_SRC. Clone them with:" >&2
    echo "  git clone --depth 1 https://github.com/iree-org/iree $IREE_SRC" >&2
    exit 1
fi

mkdir -p "$BUILD_DIR/include/iree/builtins/ukernel/arch/arm_64"
CFLAGS="-O3 -mcpu=$CPU -DNDEBUG"
INCLUDES="-I$BUILD_DIR/include -I$IREE_SRC/runtime/src"

# -------------------------------------------------------------------
# IREE tile functions
# -------------------------------------------------------------------
echo ">> Compiling IREE arm64 mmt4d tile functions ($(git -C "$IREE_SRC" log -1 --format=%h 2>/dev/null || echo unknown))" >&2
grep -v cmakedefine "$UK_SRC/arch/arm_64/config_arm_64.h.in" \
    > "$BUILD_DIR/include/iree/builtins/ukernel/arch/arm_64/config_arm_64.h"
clang $CFLAGS $INCLUDES -c "$UK_SRC/arch/arm_64/mmt4d_arm_64_base.c" -o "$BUILD_DIR/iree_mmt4d_arm_64_base.o"

# -------------------------------------------------------------------
# ukr-mlir kernels (same lowering as blis_ukernel_generator.sh, plus -inline so
# the bare-pointer entry absorbs the kernel and its memref descriptors)
# -------------------------------------------------------------------
generate_ukr () {
    local NAME=$1 MR=$2 NR=$3; shift 3
    local D="$BUILD_DIR/$NAME"
    python3 scripts/blis_ukernel_generator.py $MR $NR f32 $ARCH row f32 --name=$NAME "$@" > $D.mlir
    ./build/bin/transform-opt \
        -transform=test/ukernel_transform.mlir \
        --is-rvv=false \
        --vec-len=$VEC_LEN \
        --num-vecs=$NUM_VECS \
        --fma-units=$FMA_UNITS \
        --fma-latency=$FMA_LATENCY \
        $D.mlir > ${D}_transformed.mlir
    ${LLVM_BIN}mlir-opt \
        -inline \
        -canonicalize \
        -one-shot-bufferize="bufferize-function-boundaries" \
        -convert-linalg-to-loops \
        -lower-vector-multi-reduction \
        -convert-vector-to-scf \
        -convert-vector-to-llvm \
        -convert-scf-to-cf \
        -expand-strided-metadata \
        -cse \
        -loop-invariant-code-motion \
        -memref-expand \
        -normalize-memrefs \
        -lower-affine \
        -convert-math-to-llvm \
        -convert-arith-to-llvm \
        -convert-cf-to-llvm \
        -convert-ub-to-llvm \
        -finalize-memref-to-llvm \
        -convert-func-to-llvm \
        -reconcile-unrealized-casts \
        -canonicalize \
        ${D}_transformed.mlir > ${D}_lowered.mlir
    ${LLVM_BIN}mlir-translate -mlir-to-llvmir ${D}_lowered.mlir > $D.ll
    ${LLVM_BIN}llc -O3 -mcpu=$CPU -filetype=obj $D.ll -o $D.o
}

# Shapes must match UKR_SHAPES / UKR_V1_SHAPES in test/iree_tile_bench.c.
echo ">> Generating ukr-mlir kernels" >&2
UKR_OBJS=()
# Current kernels: bare-pointer entry, K split chosen by the generator.
for SHAPE in 1x8 2x8 4x8 8x8 8x12; do
    MR=${SHAPE%x*}; NR=${SHAPE#*x}
    generate_ukr ukr_${SHAPE}_b0 $MR $NR --alpha=one --beta=zero --edges=false --entry=bare
    generate_ukr ukr_${SHAPE}_b1 $MR $NR --alpha=one --beta=one --edges=false --entry=bare
    UKR_OBJS+=("$BUILD_DIR/ukr_${SHAPE}_b0.o" "$BUILD_DIR/ukr_${SHAPE}_b1.o")
done
# v1 baseline: MLIR C interface, no K split (the first comparison).
for SHAPE in 1x8 2x8 4x8 8x8; do
    MR=${SHAPE%x*}; NR=${SHAPE#*x}
    generate_ukr ukr_v1_${SHAPE}_b0 $MR $NR --alpha=one --beta=zero --edges=false --k-split=1
    generate_ukr ukr_v1_${SHAPE}_b1 $MR $NR --alpha=one --beta=one --edges=false --k-split=1
    UKR_OBJS+=("$BUILD_DIR/ukr_v1_${SHAPE}_b0.o" "$BUILD_DIR/ukr_v1_${SHAPE}_b1.o")
done
generate_ukr ukr_8x8_blis 8 8 --alpha=dynamic --beta=dynamic --edges=true
UKR_OBJS+=("$BUILD_DIR/ukr_8x8_blis.o")

# -------------------------------------------------------------------
# Harness
# -------------------------------------------------------------------
echo ">> Building harness" >&2
clang $CFLAGS $INCLUDES test/iree_tile_bench.c "$BUILD_DIR/iree_mmt4d_arm_64_base.o" "${UKR_OBJS[@]}" \
    -o "$BUILD_DIR/iree_tile_bench"

echo ">> Running" >&2
if [[ -n "$OUTPUT" ]]; then
    mkdir -p "$(dirname "$OUTPUT")"
    "$BUILD_DIR/iree_tile_bench" ${BENCH_ARGS[@]+"${BENCH_ARGS[@]}"} | tee "$OUTPUT"
else
    "$BUILD_DIR/iree_tile_bench" ${BENCH_ARGS[@]+"${BENCH_ARGS[@]}"}
fi

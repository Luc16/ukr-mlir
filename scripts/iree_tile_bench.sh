#!/bin/bash
# Tile-level comparison: ukr-mlir micro-kernels vs IREE's f32 mmt4d tile
# functions (plan.md, "Level 1"), on arm64 NEON, x86-64 AVX2+FMA or AVX-512.
# IREE's ukernels are plain C, so they are compiled straight from an IREE
# checkout; no IREE build is needed.

set -euo pipefail

# Default values
IREE_SRC="external/iree"
LLVM_DIR=""
BUILD_DIR=""
ISA=""
ARCH=""
VEC_LEN=""
NUM_VECS=""
FMA_UNITS=""
FMA_LATENCY=""
CPU="native"
OUTPUT=""
BENCH_ARGS=()

function usage() {
    echo "Usage: $0 [options] [-- benchmark args]" >&2
    echo "Options:" >&2
    echo "  -i <path>    IREE source checkout (default: external/iree)" >&2
    echo "  -l <path>    Path to LLVM build dir (default: tools on PATH)" >&2
    echo "  -I <isa>     neon | avx2 | avx512 (default: neon on arm64; on x86-64," >&2
    echo "               avx512 if the CPU has it, else avx2)" >&2
    echo "  -b <path>    Build directory (default: tmp/iree_tile/build_<isa>)" >&2
    echo "  -a <string>  Architecture name passed to the generator (default per ISA)" >&2
    echo "  -v <int>     Vector length in bits (default per ISA: 128 / 256 / 512)" >&2
    echo "  -r <int>     Vector registers (default per ISA: 32 / 16 / 32)" >&2
    echo "  -u <int>     Vector FMA pipes (default per ISA: 4 / 2 / 2)" >&2
    echo "  -t <int>     FMA latency in cycles (default per ISA: 4 / 4 / 4)" >&2
    echo "  -c <cpu>     Target CPU: -mcpu (arm64) or -march (x86-64) for clang," >&2
    echo "               -mcpu for llc (default: native)" >&2
    echo "  -o <file>    Also write the CSV to this file" >&2
    echo "  -h           Show help" >&2
    echo "Benchmark args (after --): --k=64,256,1024 --modes=hot,warm,cold --reps=15" >&2
    echo "  --sample-ms=20 --arena-mb=256 --check-only" >&2
    exit 1
}

while getopts "i:l:I:b:a:v:r:u:t:c:o:h" opt; do
    case $opt in
        i) IREE_SRC=$OPTARG ;;
        l) LLVM_DIR=$OPTARG ;;
        I) ISA=$OPTARG ;;
        b) BUILD_DIR=$OPTARG ;;
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

if [[ -z "$ISA" ]]; then
    case "$(uname -m)" in
        arm64|aarch64) ISA=neon ;;
        x86_64) if grep -qw avx512f /proc/cpuinfo 2>/dev/null; then ISA=avx512; else ISA=avx2; fi ;;
        *) echo "Unknown host $(uname -m); pass -I" >&2; exit 1 ;;
    esac
fi

# Per-ISA machine model, IREE source and kernel shapes. The shapes must match
# IREE_SHAPES / UKR_EXTRA_SHAPES / BLIS_M0,N0 in test/iree_tile_bench.c.
case "$ISA" in
    neon)
        D_ARCH=firestorm; D_VEC=128; D_REGS=32; D_UNITS=4; D_LAT=4
        IREE_ARCH=arm_64; IREE_FILE=mmt4d_arm_64_base.c; IREE_CFLAGS=""
        IREE_SHAPES="1x8 2x8 4x8 8x8"; EXTRA_SHAPES="8x12"; BLIS_SHAPE=8x8
        ISA_DEF=""; CPU_FLAG="-mcpu" ;;
    avx2)
        D_ARCH=haswell; D_VEC=256; D_REGS=16; D_UNITS=2; D_LAT=4
        IREE_ARCH=x86_64; IREE_FILE=mmt4d_x86_64_avx2_fma.c; IREE_CFLAGS="-mavx2 -mfma -mf16c"
        IREE_SHAPES="1x8 2x8 4x8 8x8"; EXTRA_SHAPES="6x16"; BLIS_SHAPE=8x8
        ISA_DEF="-DUKR_ISA_AVX2"; CPU_FLAG="-march" ;;
    avx512)
        D_ARCH=skx; D_VEC=512; D_REGS=32; D_UNITS=2; D_LAT=4
        IREE_ARCH=x86_64; IREE_FILE=mmt4d_x86_64_avx512_base.c
        IREE_CFLAGS="-mavx512f -mavx512bw -mavx512dq -mavx512vl -mavx512cd -mfma"
        IREE_SHAPES="1x16 2x16 4x16 8x16 16x16"; EXTRA_SHAPES="12x32 14x32"; BLIS_SHAPE=16x16
        ISA_DEF="-DUKR_ISA_AVX512"; CPU_FLAG="-march" ;;
    *) echo "Unknown ISA: $ISA" >&2; usage ;;
esac
ARCH=${ARCH:-$D_ARCH}; VEC_LEN=${VEC_LEN:-$D_VEC}; NUM_VECS=${NUM_VECS:-$D_REGS}
FMA_UNITS=${FMA_UNITS:-$D_UNITS}; FMA_LATENCY=${FMA_LATENCY:-$D_LAT}
BUILD_DIR=${BUILD_DIR:-tmp/iree_tile/build_$ISA}
echo ">> ISA $ISA: vec $VEC_LEN, $NUM_VECS regs, $FMA_UNITS FMA units, latency $FMA_LATENCY, cpu $CPU" >&2

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

mkdir -p "$BUILD_DIR/include/iree/builtins/ukernel/arch/$IREE_ARCH"
CFLAGS="-O3 $CPU_FLAG=$CPU -DNDEBUG $ISA_DEF"
INCLUDES="-I$BUILD_DIR/include -I$IREE_SRC/runtime/src"

# -------------------------------------------------------------------
# IREE tile functions
# -------------------------------------------------------------------
echo ">> Compiling IREE $IREE_FILE ($(git -C "$IREE_SRC" log -1 --format=%h 2>/dev/null || echo unknown))" >&2
# Drop the #cmakedefine lines: the optional paths (fp16/bf16/dotprod/i8mm on arm,
# the other x86 variants) stay off; f32 does not use them.
grep -v cmakedefine "$UK_SRC/arch/$IREE_ARCH/config_$IREE_ARCH.h.in" \
    > "$BUILD_DIR/include/iree/builtins/ukernel/arch/$IREE_ARCH/config_$IREE_ARCH.h"
IREE_OBJ="$BUILD_DIR/iree_${IREE_FILE%.c}.o"
clang $CFLAGS $IREE_CFLAGS $INCLUDES -c "$UK_SRC/arch/$IREE_ARCH/$IREE_FILE" -o "$IREE_OBJ"

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

echo ">> Generating ukr-mlir kernels" >&2
UKR_OBJS=()
# Current kernels: bare-pointer entry, K split chosen by the generator.
for SHAPE in $IREE_SHAPES $EXTRA_SHAPES; do
    MR=${SHAPE%x*}; NR=${SHAPE#*x}
    generate_ukr ukr_${SHAPE}_b0 $MR $NR --alpha=one --beta=zero --edges=false --entry=bare
    generate_ukr ukr_${SHAPE}_b1 $MR $NR --alpha=one --beta=one --edges=false --entry=bare
    UKR_OBJS+=("$BUILD_DIR/ukr_${SHAPE}_b0.o" "$BUILD_DIR/ukr_${SHAPE}_b1.o")
done
# v1 baseline: MLIR C interface, no K split (the first comparison).
for SHAPE in $IREE_SHAPES; do
    MR=${SHAPE%x*}; NR=${SHAPE#*x}
    generate_ukr ukr_v1_${SHAPE}_b0 $MR $NR --alpha=one --beta=zero --edges=false --k-split=1
    generate_ukr ukr_v1_${SHAPE}_b1 $MR $NR --alpha=one --beta=one --edges=false --k-split=1
    UKR_OBJS+=("$BUILD_DIR/ukr_v1_${SHAPE}_b0.o" "$BUILD_DIR/ukr_v1_${SHAPE}_b1.o")
done
generate_ukr ukr_blis ${BLIS_SHAPE%x*} ${BLIS_SHAPE#*x} --alpha=dynamic --beta=dynamic --edges=true
UKR_OBJS+=("$BUILD_DIR/ukr_blis.o")

# -------------------------------------------------------------------
# Harness
# -------------------------------------------------------------------
echo ">> Building harness" >&2
clang $CFLAGS $INCLUDES test/iree_tile_bench.c "$IREE_OBJ" "${UKR_OBJS[@]}" \
    -lm -o "$BUILD_DIR/iree_tile_bench"

echo ">> Running" >&2
if [[ -n "$OUTPUT" ]]; then
    mkdir -p "$(dirname "$OUTPUT")"
    "$BUILD_DIR/iree_tile_bench" ${BENCH_ARGS[@]+"${BENCH_ARGS[@]}"} | tee "$OUTPUT"
else
    "$BUILD_DIR/iree_tile_bench" ${BENCH_ARGS[@]+"${BENCH_ARGS[@]}"}
fi

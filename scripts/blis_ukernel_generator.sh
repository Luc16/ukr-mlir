#!/bin/bash

# Default values
MR=6
NR=8
PRECISION="float"
OUTPUT_PRECISION=""
ARCH="haswell"
LLVM_DIR=""
LAYOUT="row"
VEC_LEN=256

function usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -m <int>     MR dimension (default: 6)"
    echo "  -n <int>     NR dimension (default: 8)"
    echo "  -p <type>    Precision: float | double | half (default: float)"
    echo "  -o <type>    Output Precision: float | double | half (default: same as input)"
    echo "  -a <string>  Architecture name (default: haswell)"
    echo "  -l <path>    Path to LLVM build directory"
    echo "  -L <string>  Layout (row or column, default: row)"
    echo "  -v <int>     Vector length in bits (default: 256)"
    echo "  -h           Help"
    exit 1
}

while getopts "m:n:p:a:l:o:L:v:h" opt; do
    case $opt in
        m) MR=$OPTARG ;;
        n) NR=$OPTARG ;;
        p) PRECISION=$OPTARG ;;
        a) ARCH=$OPTARG ;;
        l) LLVM_DIR=$OPTARG ;;
        o) OUTPUT_PRECISION=$OPTARG ;;
        L) LAYOUT=$OPTARG ;;
        v) VEC_LEN=$OPTARG ;;
        h) usage ;;
        *) usage ;;
    esac
done

echo ">> Generating kernels MR=$MR NR=$NR Precision=$PRECISION Arch=$ARCH Layout=$LAYOUT OUT=$OUTPUT_PRECISION LLVM_DIR=$LLVM_DIR "

# Setup LLVM binaries path prefix if a directory was provided
LLVM_BIN=""
if [[ -n "$LLVM_DIR" ]]; then
    LLVM_BIN="${LLVM_DIR%%/}/bin/"
fi

generate_kernel () {
TYPE=$1
MLIR_TYPE=$2
OUT_MLIR_TYPE=$3
OBJ_NAME=$4

echo "Generating $TYPE micro-kernel..."

python3 scripts/blis_ukernel_generator.py $MR $NR $MLIR_TYPE $ARCH $LAYOUT $OUT_MLIR_TYPE > tmp/blis_ukernel_$TYPE.mlir

./build/bin/transform-opt \
    -transform=test/ukernel_transform.mlir \
    --is-rvv=$(echo "$ARCH" | grep -q "rv64" && echo "true" || echo "false") \
    --vec-len=$VEC_LEN \
    tmp/blis_ukernel_$TYPE.mlir \
    > tmp/blis_transformed_$TYPE.mlir

${LLVM_BIN}mlir-opt \
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
    tmp/blis_transformed_$TYPE.mlir \
    > tmp/blis_lowered_$TYPE.mlir

  ${LLVM_BIN}mlir-translate \
    -mlir-to-llvmir \
    tmp/blis_lowered_$TYPE.mlir \
    > tmp/blis_ukernel_$TYPE.ll

TARGET_FLAGS="-mcpu=native"
if [[ "$ARCH" == *"rv64"* ]]; then
  TARGET_FLAGS="-mtriple=riscv64-unknown-linux-gnu -mattr=+m,+a,+f,+d,+v,+zvfh,+zvl${VEC_LEN}b"
fi

${LLVM_BIN}llc -O3 $TARGET_FLAGS -filetype=obj \
  tmp/blis_ukernel_$TYPE.ll \
  -o tmp/$OBJ_NAME

}

mkdir -p tmp test

# mlir precision mapping
IN_PRECISION=""
case "$PRECISION" in
    float|f32) IN_PRECISION="f32" ;;
    double|f64) IN_PRECISION="f64" ;;
    half|f16) IN_PRECISION="f16" ;;
    bf16) IN_PRECISION="bf16" ;;
    i8) IN_PRECISION="i8" ;;
    i16) IN_PRECISION="i16" ;;
    i32) IN_PRECISION="i32" ;;
    ui8) IN_PRECISION="ui8" ;;
    ui16) IN_PRECISION="ui16" ;;
    ui32) IN_PRECISION="ui32" ;;
    si8) IN_PRECISION="si8" ;;
    si16) IN_PRECISION="si16" ;;
    si32) IN_PRECISION="si32" ;;
esac

PRECISION_LETTER="x"
case "$PRECISION" in
    float|f32) PRECISION_LETTER="s" ;;
    double|f64) PRECISION_LETTER="d" ;;
    half|f16) PRECISION_LETTER="h" ;;
esac

echo "Input MLIR type: $IN_PRECISION"

OUT_MLIR_TYPE=$IN_PRECISION
if [[ -n "$OUTPUT_PRECISION" ]]; then
  case "$OUTPUT_PRECISION" in
    float|f32) OUT_MLIR_TYPE="f32" ;;
    double|f64) OUT_MLIR_TYPE="f64" ;;
    half|f16) OUT_MLIR_TYPE="f16" ;;
    bf16) OUT_MLIR_TYPE="bf16" ;;
    i8) OUT_MLIR_TYPE="i8" ;;
    i16) OUT_MLIR_TYPE="i16" ;;
    i32) OUT_MLIR_TYPE="i32" ;;
    ui8) OUT_MLIR_TYPE="ui8" ;;
    ui16) OUT_MLIR_TYPE="ui16" ;;
    ui32) OUT_MLIR_TYPE="ui32" ;;
    si8) OUT_MLIR_TYPE="si8" ;;
    si16) OUT_MLIR_TYPE="si16" ;;
    si32) OUT_MLIR_TYPE="si32" ;;
  esac
    generate_kernel $PRECISION_LETTER $IN_PRECISION $OUT_MLIR_TYPE "bli_gemm_${ARCH}_${MR}x${NR}_${LAYOUT}__in_${IN_PRECISION}_out_${OUT_MLIR_TYPE}_ukernel.o"
else
    generate_kernel $PRECISION_LETTER $IN_PRECISION $OUT_MLIR_TYPE "bli_${PRECISION_LETTER}gemm_${ARCH}_${MR}x${NR}_${LAYOUT}_ukernel.o"
    echo "Generated kernel: bli_${PRECISION_LETTER}gemm_${ARCH}_${MR}x${NR}_${LAYOUT}_ukernel.o"
fi

echo "Compilation complete! .o files generated in tmp/."

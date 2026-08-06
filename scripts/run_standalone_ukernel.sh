#!/bin/bash

# Default values
MR=6
NR=8
K=256
ITERATIONS=100
PRECISION="float"
ARCH="haswell"
LLVM_DIR=""
LAYOUT="col"
VEC_LEN=256

function usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -m <int>      MR dimension (default: 6)"
    echo "  -n <int>      NR dimension (default: 8)"
    echo "  -k <int>      Test K dimension (default: 256)"
    echo "  -i <int>      Iterations (default: 100)"
    echo "  -p <type>     Precision: 'float' or 'double' (default: float)"
    echo "  -a <string>   Architecture (default: haswell)"
    echo "  -l <path>     Path to LLVM build dir (Optional)"
    echo "  -L <string>   Layout: 'row' or 'col' (default: col)"
    echo "  -v <int>      Vector length in bits (default: 256)"
    echo "  -h            Show help"
    exit 1
}

while getopts "m:n:k:i:p:a:l:L:v:h" opt; do
    case $opt in
        m) MR=$OPTARG ;;
        n) NR=$OPTARG ;;
        k) K=$OPTARG ;;
        i) ITERATIONS=$OPTARG ;;
        p) PRECISION=$OPTARG ;;
        a) ARCH=$OPTARG ;;
        l) LLVM_DIR=$OPTARG ;;
        L) LAYOUT=$OPTARG ;;
        v) VEC_LEN=$OPTARG ;;
        h) usage ;;
        *) usage ;;
    esac
done

mkdir -p tmp

echo "=========================================================="
echo ">> Naked Test: MR=$MR, NR=$NR | Test Size: M=$M, N=$N, K=$K"
echo ">> Precision=$PRECISION | Arch=$ARCH | Layout=$LAYOUT"
echo "=========================================================="

# -------------------------------------------------------------------
# Path Configurations (LLVM ONLY - No BLIS required)
# -------------------------------------------------------------------
LLVM_FLAG=""
RUNNER_LIB_FLAGS="-L./test -Wl,-rpath,./test"

if [[ -n "$LLVM_DIR" ]]; then
    LLVM_FLAG="-l $LLVM_DIR"
    RUNNER_LIB_PATH="${LLVM_DIR%%/}/lib"
    RUNNER_LIB_FLAGS="-L$RUNNER_LIB_PATH -Wl,-rpath,$RUNNER_LIB_PATH"
fi

# -------------------------------------------------------------------
# Determine Object File Names & Precision Macros
# -------------------------------------------------------------------
if [[ "$PRECISION" == "float" ]]; then
    PREFIX="s"
    CPP_IN="32"
    CPP_OUT="32"
else
    PREFIX="d"
    CPP_IN="64"
    CPP_OUT="64"
fi

FUNC_NAME="bli_${PREFIX}gemm_${ARCH}_${MR}_${NR}_ukernel"
OBJ_FILE="tmp/bli_${PREFIX}gemm_${ARCH}_${MR}x${NR}_${LAYOUT}_ukernel.o"

CXX=g++
MARCH_FLAGS="-march=native"
if [[ "$ARCH" == *"rv64"* ]]; then
    MARCH_FLAGS="-march=rv64gcv"
fi
CXX_FLAGS="-O3 -std=c++17 $MARCH_FLAGS"

# -------------------------------------------------------------------
# Generation & Compilation
# -------------------------------------------------------------------
echo -e "\n>> Step 1: Checking for $PRECISION MLIR micro-kernel..."

if [[ -f "$OBJ_FILE" ]]; then
    echo "   [CACHED] $OBJ_FILE already exists. Skipping MLIR generation."
else
    echo "   [BUILDING] Generating micro-kernel..."
    ./scripts/blis_ukernel_generator.sh -m $MR -n $NR -p $PRECISION -a $ARCH -L $LAYOUT $LLVM_FLAG -v $VEC_LEN
fi

echo -e "\n>> Step 2: Compiling Naked C++ Tester..."

LAYOUT_DEF=$([[ "$LAYOUT" == "row" ]] && echo "-DIS_ROW_MAJOR" || echo "")
DEFS="-DTEST_MR=$MR -DTEST_NR=$NR -DTYPE_IN=$CPP_IN -DTYPE_OUT=$CPP_OUT -DMLIR_CIFACE_FUNC=_mlir_ciface_${FUNC_NAME} $LAYOUT_DEF"

$CXX $CXX_FLAGS -c test/test_ukernel.cpp -o tmp/test_ukernel.o $DEFS

echo -e "\n>> Step 3: Linking..."

$CXX tmp/test_ukernel.o "$OBJ_FILE" -o test_ukernel_exe \
    $CXX_FLAGS \
    $RUNNER_LIB_FLAGS \
    -lmlir_c_runner_utils

# -------------------------------------------------------------------
# Execution
# -------------------------------------------------------------------
echo -e "\n>> Step 4: Running Correctness & Benchmark..."

./test_ukernel_exe "$MR" "$NR" "$K" "$ITERATIONS"

# Cleanup
rm -f tmp/test_ukernel.o test_ukernel_exe

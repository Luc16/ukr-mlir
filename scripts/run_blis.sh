#!/bin/bash

# Default values
MR=6
NR=8
M=1920
N=1920
K=1920
ITERATIONS=5
PRECISION="double"
ARCH="haswell"
LLVM_DIR=""
BLIS_DIR=""
LAYOUT="row"
VEC_LEN=256

function usage() {
    echo "Usage: $0 [options]" >&2
    echo "Options:" >&2
    echo "  -m <int>      MR dimension (default: 6)" >&2
    echo "  -n <int>      NR dimension (default: 8)" >&2
    echo "  -x <int>      Global M dimension (default: 1920)" >&2
    echo "  -y <int>      Global N dimension (default: 1920)" >&2
    echo "  -z <int>      Global K dimension (default: 1920)" >&2
    echo "  -i <int>      Iterations (default: 5)" >&2
    echo "  -p <type>     Precision: 'float' or 'double' (default: double)" >&2
    echo "  -a <string>   Architecture (default: haswell)" >&2
    echo "  -l <path>     Path to LLVM build dir (Optional)" >&2
    echo "  -L <string>   Layout: 'row' or 'column' (default: row)" >&2
    echo "  -b <path>     Path to uninstalled BLIS root dir (Optional)" >&2
    echo "  -v <int>      Vector length in bits (default: 256)" >&2
    echo "  -h            Show help" >&2
    exit 1
}

while getopts "m:n:x:y:z:i:p:a:l:L:b:v:h" opt; do
    case $opt in
        m) MR=$OPTARG ;;
        n) NR=$OPTARG ;;
        x) M=$OPTARG ;;
        y) N=$OPTARG ;;
        z) K=$OPTARG ;;
        i) ITERATIONS=$OPTARG ;;
        p) PRECISION=$OPTARG ;;
        a) ARCH=$OPTARG ;;
        l) LLVM_DIR=$OPTARG ;;
        L) LAYOUT=$OPTARG ;;
        v) VEC_LEN=$OPTARG ;;
        b) BLIS_DIR=$OPTARG ;;
        h) usage ;;
        *) usage ;;
    esac
done

mkdir -p tmp

# Redirect informational output to stderr so stdout only prints CSV data
echo "==========================================================" >&2
echo ">> Config: MR=$MR, NR=$NR | Global: M=$M, N=$N, K=$K | Precision=$PRECISION | Arch=$ARCH" >&2
echo "==========================================================" >&2

# -------------------------------------------------------------------
# Path Configurations (LLVM & BLIS)
# -------------------------------------------------------------------
LLVM_FLAG=""
RUNNER_LIB_FLAGS="-L./test -Wl,-rpath,./test"

if [[ -n "$LLVM_DIR" ]]; then
    LLVM_FLAG="-l $LLVM_DIR"
    RUNNER_LIB_PATH="${LLVM_DIR%%/}/lib"
    RUNNER_LIB_FLAGS="-L$RUNNER_LIB_PATH -Wl,-rpath,$RUNNER_LIB_PATH"
fi

BLIS_CFLAGS=""
BLIS_LDFLAGS="-lblis" # Default to system-installed BLIS

if [[ -n "$BLIS_DIR" ]]; then
    BLIS_CFLAGS="-I${BLIS_DIR%%/}/include/${ARCH} -I${BLIS_DIR}/include/linux-x86_64"
    if [[ -f "${BLIS_DIR}/lib/${ARCH}/libblis.a" ]]; then
        BLIS_LDFLAGS="${BLIS_DIR}/lib/${ARCH}/libblis.a"
    else
        BLIS_LDFLAGS="${BLIS_DIR}/lib/libblis.a"
    fi
    echo ">> Using Custom BLIS Path: $BLIS_DIR" >&2
fi

# -------------------------------------------------------------------
# Determine Object File Names
# -------------------------------------------------------------------
if [[ "$PRECISION" == "float" ]]; then
    TYPE_DEF="-DUSE_FLOAT"
    PREFIX="s"
else
    TYPE_DEF=""
    PREFIX="d"
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
echo -e "\n>> Step 1: Checking for $PRECISION MLIR micro-kernel..." >&2

if [[ -f "$OBJ_FILE" ]]; then
    echo "   [CACHED] $OBJ_FILE already exists. Skipping MLIR generation." >&2
else
    echo "   [BUILDING] Generating micro-kernel..." >&2
    ./scripts/blis_ukernel_generator.sh -m $MR -n $NR -p $PRECISION -a $ARCH -L $LAYOUT $LLVM_FLAG -v $VEC_LEN >&2
fi

echo -e "\n>> Step 2: Compiling Benchmark Runner..." >&2

# Pass the custom definition flags needed by the C++ wrapper
DEFS="-DRUN_CUSTOM -DTEST_MR=$MR -DTEST_NR=$NR -DMR=$MR -DNR=$NR -DTEST_KERNEL_FUNC=$FUNC_NAME $TYPE_DEF"

# Compile the newly modified C++ benchmark file
$CXX $CXX_FLAGS $BLIS_CFLAGS -c test/benchmark_runner.cpp -o tmp/runner.o $DEFS -I./test

echo -e "\n>> Step 3: Linking..." >&2

$CXX tmp/runner.o "$OBJ_FILE" -o run_blis_exe \
    $CXX_FLAGS \
    $RUNNER_LIB_FLAGS \
    -lmlir_c_runner_utils \
    $BLIS_LDFLAGS

# -------------------------------------------------------------------
# Execution
# -------------------------------------------------------------------
echo -e "\n>> Step 4: Running Custom Kernel Benchmark..." >&2

# The '1' specifies that this is a custom kernel run for the is_custom parameter
./run_blis_exe "$M" "$N" "$K" "$ITERATIONS" 1 "$MR" "$NR" "$ARCH" "$PRECISION" "$LAYOUT"

# Cleanup
rm -f tmp/runner.o run_blis_exe

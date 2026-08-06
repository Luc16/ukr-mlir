#!/bin/bash

MR=16
NR=12
K=256
ITERATIONS=100
VEC_LEN=256     # Default vector length
ARCH="skx"
LLVM_DIR=""
LAYOUT="col"
TYPE_IN="f32"   # Default
TYPE_OUT="f32"  # Default

function usage() {
    echo "Usage: $0 [options]"
    echo "  -m <int>      MR dimension"
    echo "  -n <int>      NR dimension"
    echo "  -z <int>      k dimension"
    echo "  -i <int>      Iterations (default: 100)"
    echo "  -v <int>      Vector length in bits (default: 256)"
    echo "  -I <string> Input Precision (f32, bf16, i8, i32)"
    echo "  -O <string> Output Precision (f32, i32)"
    echo "  -L <row|col>  Layout"
    echo "  -a <arch>     Architecture"
    echo "  -l <path>     LLVM Path"
    exit 1
}

# Added 'v:' to the getopts string
while getopts "m:n:x:y:z:i:v:I:O:L:a:l:h" opt; do
    case $opt in
        m) MR=$OPTARG ;;
        n) NR=$OPTARG ;;
        z) K=$OPTARG ;;
        i) ITERATIONS=$OPTARG ;;
        v) VEC_LEN=$OPTARG ;; # <--- Mapped the vector length variable
        I) TYPE_IN=$OPTARG ;;
        O) TYPE_OUT=$OPTARG ;;
        L) LAYOUT=$OPTARG ;;
        a) ARCH=$OPTARG ;;
        l) LLVM_DIR=$OPTARG ;;
        h) usage ;;
        *) usage ;;
    esac
done

echo "=========================================================="
echo ">> Naked Test: MR=$MR, NR=$NR | K=$K | Iterations=$ITERATIONS"
echo ">> Precision: In=$TYPE_IN, Out=$TYPE_OUT | Arch=$ARCH | VecLen=$VEC_LEN"
echo "=========================================================="

if [[ -n "$LLVM_DIR" ]]; then
  LLVM_FLAG="-l $LLVM_DIR"
  RUNNER_LIB_FLAGS="-L${LLVM_DIR%%/}/lib -Wl,-rpath,${LLVM_DIR%%/}/lib"
else 
  LLVM_FLAG=""
  RUNNER_LIB_FLAGS="-L./test -Wl,-rpath,./test"
fi

# 1. Map String Inputs to BLIS Prefixes and C++ Macros
if [[ "$TYPE_IN" == "f32" || "$TYPE_IN" == "float" || "$TYPE_IN" == "32" ]]; then
    PREFIX="s"; CPP_IN="32"
elif [[ "$TYPE_IN" == "f64" || "$TYPE_IN" == "double" || "$TYPE_IN" == "64" ]]; then
    PREFIX="d"; CPP_IN="64"
elif [[ "$TYPE_IN" == "f16" || "$TYPE_IN" == "half" || "$TYPE_IN" == "16" ]]; then
    PREFIX=""; CPP_IN="16"
elif [[ "$TYPE_IN" == "bf16" || "$TYPE_IN" == "bfloat16" ]]; then
    PREFIX="x"; CPP_IN="17" 
elif [[ "$TYPE_IN" == "i8" || "$TYPE_IN" == "int8" || "$TYPE_IN" == "8" ]]; then
    PREFIX="x"; CPP_IN="8"  
elif [[ "$TYPE_IN" == "i32" || "$TYPE_IN" == "int32" ]]; then
    PREFIX=""; CPP_IN="33" 
else
    PREFIX="s"; CPP_IN="32" # Default
fi

# 2. Map Output Precision to C++ Macros
if [[ "$TYPE_OUT" == "f32" || "$TYPE_OUT" == "float" || "$TYPE_OUT" == "32" ]]; then
    CPP_OUT="32"
elif [[ "$TYPE_OUT" == "bf16" || "$TYPE_OUT" == "bfloat16" ]]; then
    CPP_OUT="17"
elif [[ "$TYPE_OUT" == "i32" || "$TYPE_OUT" == "int32" ]]; then
    CPP_OUT="33" 
else
    CPP_OUT="32" # Default
fi

if [[ "$TYPE_IN" == "$TYPE_OUT" ]]; then
    OBJ_FILE="tmp/bli_${PREFIX}gemm_${ARCH}_${MR}x${NR}_${LAYOUT}_ukernel.o"
    GEN_CMD="./scripts/blis_ukernel_generator.sh -m $MR -n $NR -p $TYPE_IN -a $ARCH -L $LAYOUT -v $VEC_LEN $LLVM_FLAG"
else
    OBJ_FILE="tmp/bli_gemm_${ARCH}_${MR}x${NR}_${LAYOUT}__in_${TYPE_IN}_out_${TYPE_OUT}_ukernel.o"
    GEN_CMD="./scripts/blis_ukernel_generator.sh -m $MR -n $NR -p $TYPE_IN -o $TYPE_OUT -a $ARCH -L $LAYOUT -v $VEC_LEN $LLVM_FLAG"
fi

FUNC_NAME="bli_${PREFIX}gemm_${ARCH}_${MR}_${NR}_ukernel"

# 4. Generate MLIR
# if [[ -f "$OBJ_FILE" ]]; then
#     echo "   [CACHED] $OBJ_FILE exists. Skipping MLIR generation."
# else
#     echo "   [BUILDING] Generating MLIR..."
#     $GEN_CMD
#
#     MLIR_OUTPUT=$(ls -t tmp/*.o 2>/dev/null | head -n 1)
#     if [[ -n "$MLIR_OUTPUT" && "$MLIR_OUTPUT" != "$OBJ_FILE" ]]; then
#         mv "$MLIR_OUTPUT" "$OBJ_FILE"
#     fi
# fi
#
echo "   [BUILDING] Generating MLIR..."
$GEN_CMD

MLIR_OUTPUT=$(ls -t tmp/*.o 2>/dev/null | head -n 1)
if [[ -n "$MLIR_OUTPUT" && "$MLIR_OUTPUT" != "$OBJ_FILE" ]]; then
    mv "$MLIR_OUTPUT" "$OBJ_FILE"
fi

# 5. Compile C++ Tester
echo "   [BUILDING] Compiling C++ Tester..."
CXX=clang++
MARCH_FLAGS=$([[ "$ARCH" == *"rv64"* ]] && echo "-march=rv64gcv" || echo "-march=native")
#
CXX_FLAGS="-O3 -std=c++17 $MARCH_FLAGS"
LAYOUT_DEF=$([[ "$LAYOUT" == "row" ]] && echo "-DIS_ROW_MAJOR" || echo "")
#
# # Inject the numeric macros mapping into C++
DEFS="-DTEST_MR=$MR -DTEST_NR=$NR -DTYPE_IN=$CPP_IN -DTYPE_OUT=$CPP_OUT -DMLIR_CIFACE_FUNC=_mlir_ciface_${FUNC_NAME} $LAYOUT_DEF -Wno-narrowing"
#
$CXX $CXX_FLAGS -c test/test_widening.cpp -o tmp/test_ukernel.o $DEFS
#
# # 6. Link & Run
$CXX tmp/test_ukernel.o "$OBJ_FILE" -o test_exe $CXX_FLAGS $RUNNER_LIB_FLAGS -lmlir_c_runner_utils
#
echo -e "\n>> Running Test..."
./test_exe $K $ITERATIONS 
#
rm -f tmp/test_ukernel.o test_exe

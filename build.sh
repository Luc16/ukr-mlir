#!/bin/bash

# Default values
BUILD_TYPE="Release"
LLVM_PROJ_DIR=""

function usage() {
    echo "Usage: $0 -l <llvm_build_dir> [-t build_type]"
    echo "Options:"
    echo "  -l <path>    Path to the LLVM project build directory (Required)"
    echo "  -t <type>    CMake build type: Debug | Release | RelWithDebInfo (Default: Release)"
    echo "  -h           Show help"
    exit 1
}

while getopts "l:t:h" opt; do
    case $opt in
        l) LLVM_PROJ_DIR=$OPTARG ;;
        t) BUILD_TYPE=$OPTARG ;;
        h) usage ;;
        *) usage ;;
    esac
done

if [[ -z "$LLVM_PROJ_DIR" ]]; then
    echo "Error: LLVM build directory is required."
    usage
fi

# Ensure the paths are absolute
LLVM_PROJ_DIR=$(realpath "$LLVM_PROJ_DIR")

if [[ ! -d "$LLVM_PROJ_DIR/lib/cmake/mlir" ]]; then
    echo "Error: Could not find MLIR CMake config in $LLVM_PROJ_DIR/lib/cmake/mlir"
    exit 1
fi

echo ">> Configuring build ($BUILD_TYPE mode)..."

mkdir -p build && cd build

cmake -G Ninja .. \
   -DCMAKE_C_COMPILER=clang \
   -DCMAKE_CXX_COMPILER=clang++ \
   -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
   -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
   -DMLIR_DIR="$LLVM_PROJ_DIR/lib/cmake/mlir" \
   -DLLVM_DIR="$LLVM_PROJ_DIR/lib/cmake/llvm"

echo ">> Building transform-opt..."
cmake --build . --target transform-opt -j$(nproc)

cd ..
echo ">> Build complete!"

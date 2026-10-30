# ukr-mlir: Automated GEMM Micro-Kernel Generation

[![Paper](https://img.shields.io/badge/Paper-Cross--Platform_MLIR_Based_GEMM_Micro--Kernel_Generation-blue)](#)

`ukr-mlir` is an infrastructure for the analytical modeling and automatic generation of matrix multiplication (GEMM) micro-kernels using MLIR and LLVM. It provides a full pipeline to generate high-performance micro-kernels for multiple architectures, evaluate their numerical correctness against a naive baseline, and benchmark them natively within the BLIS framework.

This repository contains the official infrastructure and artifact for the paper: **Cross-Platform MLIR Based GEMM Micro-Kernel Generation**.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Build Instructions](#build-instructions)
- [Running Experiments & Usage Examples](#running-experiments--usage-examples)
  - [1. BLIS Micro-Kernel Generation (`blis_ukernel_generator.sh`)](#1-blis-micro-kernel-generation-blis_ukernel_generatorsh)
  - [2. Standalone Micro-Kernel Validation (`run_standalone_ukernel.sh`)](#2-standalone-micro-kernel-validation-run_standalone_ukernelsh)
  - [3. BLIS Integration Benchmark (`run_blis.sh`)](#3-blis-integration-benchmark-run_blissh)
  - [4. Mixed-Precision / Widening (`run_widening_ukernel.sh`)](#4-mixed-precision--widening-run_widening_ukernelsh)
- [Project Structure](#project-structure)
- [Citation](#citation)
- [Authors](#authors)
- [License](#license)

---

## Prerequisites

To build and run the experiments, ensure you have the following dependencies available on your system:

- **LLVM & MLIR:** A local build of the LLVM Project with MLIR enabled (compiled from source with CMake targets available).
- **BLIS:** A local build of the BLIS instantiation framework (required for macro-kernel benchmarks).
- **C++17 Compiler:** `clang++` or `g++`.
- **Build Tools:** `cmake` (3.15 or higher) and `ninja` or `make`.
- **Python 3:** Required for the BLIS generator scripts.

---

## Build Instructions

Before running any experiments, you must compile the custom MLIR transform optimizer (`transform-opt`) included in this repository.

### Quick Build (Recommended)
Use the provided `build.sh` script, pointing it to your local LLVM build directory:

```bash
# Clone the repository
git clone https://github.com/your-username/ukr-mlir.git
cd ukr-mlir

# Build the transform-opt tool
./build.sh -l /path/to/your/llvm-project/build
```

### Manual CMake Build
Alternatively, run CMake manually specifying your LLVM and MLIR installation paths:

```bash
mkdir build && cd build
cmake .. \
  -DLLVM_DIR=/path/to/llvm-project/build/lib/cmake/llvm \
  -DMLIR_DIR=/path/to/llvm-project/build/lib/cmake/mlir
make -j$(nproc)
```

---

## Running Experiments & Usage Examples

The `scripts/` directory contains the primary entry points for generating, validating, and benchmarking micro-kernels.

### 1. BLIS Micro-Kernel Generation (`blis_ukernel_generator.sh`)

This script drives the end-to-end generation and compilation pipeline for BLIS-compatible micro-kernels. It invokes `blis_ukernel_generator.py` to create the initial MLIR dialect representation, applies vector optimizations using `transform-opt` and `mlir-opt`, translates the lowered MLIR to LLVM IR, and compiles it into a target-specific object file (`.o`) using `llc`. All intermediate files and final `.o` object files are emitted to the `tmp/` directory.

**Example Command:**

```bash
./scripts/blis_ukernel_generator.sh -a haswell -L row -p float -v 256 -m 6 -n 8 -l /path/to/your/llvm-project/build
```

**Parameters:**
- `-m 6`: Micro-kernel row dimension ($MR$). Default is `6`.
- `-n 8`: Micro-kernel column dimension ($NR$). Default is `8`.
- `-p float`: Input precision (`float`, `double`, `half`, `bf16`, `i8`, `i16`, `i32`, `ui8`, `ui16`, `ui32`, `si8`, `si16`, `si32`). Default is `float`.
- `-o float`: Output/accumulation precision (`float`, `double`, `half`, `bf16`, etc.). Default is the same as input precision.
- `-a haswell`: Target architecture name (e.g., `haswell`, `firestorm`, `arm`, `rv64`). Default is `haswell`.
- `-L row`: Matrix layout (`row` or `col`). Default is `row`.
- `-v 256`: Vector register length in bits (e.g., `128`, `256`, `512`). Default is `256`.
- `-l /path/to/llvm-project/build`: Path to compiled LLVM build directory (used to resolve `mlir-opt`, `mlir-translate`, and `llc`).

---

### 2. Standalone Micro-Kernel Validation (`run_standalone_ukernel.sh`)

This script generates the MLIR micro-kernel, lowers it to an object file, and links it against a standalone C++ runner. It compares the output of the generated kernel against a naive nested-loop C++ implementation to verify numerical correctness, and outputs raw GFLOPS.

**Example Command:**

```bash
./scripts/run_standalone_ukernel.sh -a firestorm -L row -p float -v 128 -l ../llvm-project/build/ -k 512 -m 20 -n 4 -i 10000000
```

**Parameters:**
- `-a firestorm`: Target architecture (e.g., `firestorm`, `haswell`, `arm`, `rv64`).
- `-L row`: Matrix layout (`row` or `col`).
- `-p float`: Data precision.
- `-v 128`: Vector length in bits (e.g., `128` for NEON, `256` for AVX2).
- `-l ../llvm-project/build/`: Path to your compiled LLVM project build directory.
- `-m 20 -n 4`: Micro-kernel dimensions ($MR \times NR$).
- `-k 512`: The $K$ dimension for the standalone dot-product loop.
- `-i 10000000`: Number of iterations for stable benchmarking.

---

### 3. BLIS Integration Benchmark (`run_blis.sh`)

This script integrates your custom MLIR micro-kernel directly into the BLIS framework context and benchmarks it using `bli_gemm_ex`. It outputs a comparison between the native BLIS micro-kernel and the custom MLIR-generated one.

**Example Command:**

```bash
./scripts/run_blis.sh -a firestorm -L row -p float -v 128 -l ../llvm-project/build/ -b ../blis -x 1024 -y 1024 -z 1024 -m 20 -n 4
```

**Parameters:**
- `-b ../blis`: Path to your local BLIS installation. The script will automatically locate the correct include and lib directories.
- `-x 1024 -y 1024 -z 1024`: Global macro-matrix dimensions ($M \times N \times K$).
- *(Inherits `-a`, `-L`, `-p`, `-v`, `-l`, `-m`, `-n` identically to the standalone script).*

---

### 4. Mixed-Precision / Widening (`run_widening_ukernel.sh`)

For testing mixed-precision workloads (e.g., `f16` or `bf16` inputs accumulating into `f32`), use this script. It functions similarly to the standalone runner but allows for mismatched input and output types.

**Example Command:**

```bash
./scripts/run_widening_ukernel.sh -a rv64 -L col -I f16 -O f32 -v 512 -l ../llvm-project/build/ -m 16 -n 12 -z 256 -i 100
```

**Parameters:**
- `-I f16`: Input precision type (e.g., `f16`, `bf16`).
- `-O f32`: Output/accumulation precision type.
- *(Inherits target, layout, vector length, and dimension flags from standalone runner).*

---

## Project Structure

- **`include/` & `lib/`**: Contains the C++ source code and MLIR TableGen definitions (`Ukernel.td`, `Ukernel.h`, `Ukernel.cpp`).
- **`scripts/`**: Automation scripts for code generation, execution wrappers, and BLIS framework integration.
- **`test/`**: C++ unit tests, benchmark runners, runner wrappers (`wrapper.h`), and MLIR transform dialect configurations (`ukernel_transform.mlir`).
- **`transform-opt/`**: Implementation of the custom MLIR optimization binary (`transform-opt.cpp`).

---

## Citation

If you use this infrastructure or code in your research, please cite our paper:

```bibtex
@inproceedings{ribas2026microkernels,
  title={Cross-Platform MLIR Based GEMM Micro-Kernel Generation},
  author={Ribas, Luc Joffily and Araujo, Guido},
  year={2026}
}
```

---

## Authors

- **Luc Joffily Ribas** - *Universidade Estadual de Campinas (Unicamp)*
- **Guido Araujo** - *[Celera Systems](https://celera.ai/) & Universidade Estadual de Campinas (Unicamp)*

---

## License

This project is licensed under the terms of the `LICENSE` file included in the root directory.

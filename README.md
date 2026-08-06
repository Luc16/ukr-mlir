# ukr-mlir: Automated GEMM Micro-Kernel Generation

[![Paper](https://img.shields.io/badge/Paper-Cross--Platform_MLIR_Based_GEMM_Micro--Kernel_Generation-blue)](#)

This repository contains the official infrastructure and artifact for the paper: **Cross-Platform MLIR Based GEMM Micro-Kernel Generation**.

`ukr-mlir` is an infrastructure for the analytical modeling and automatic generation of matrix multiplication (GEMM) micro-kernels using MLIR and LLVM. It provides a full pipeline to generate high-performance micro-kernels for multiple architectures, evaluate their numerical correctness against a naive baseline, and benchmark them natively within the BLIS framework.

## Prerequisites

To build and run the experiments, you will need the following dependencies:
*   **LLVM & MLIR:** A local build of the LLVM Project with MLIR enabled.
*   **BLIS:** A local build of the BLIS instantiation framework (required for macro-kernel benchmarks).
*   **C++17 Compiler:** `clang++` or `g++`.
*   **Build Tools:** `cmake` and `ninja`.

## Build Instructions

Before running any experiments, you must compile the custom MLIR transform optimizer (`transform-opt`) included in this repository. 

Use the provided `build.sh` script, pointing it to your local LLVM build directory:

```bash
# Clone the repository
git clone [https://github.com/your-username/ukr-mlir.git](https://github.com/your-username/ukr-mlir.git)
cd ukr-mlir

# Build the transform-opt tool
./build.sh -l /path/to/your/llvm-project/build
```

## Running Experiments

The `scripts/` directory contains the primary entry points for generating, validating, and benchmarking the micro-kernels.

### 1. Standalone Micro-Kernel Validation (`run_standalone_ukernel.sh`)

This script generates the MLIR micro-kernel, lowers it to an object file, and links it against a standalone C++ runner. It compares the output of the generated kernel against a naive nested-loop C++ implementation to verify numerical correctness, and outputs raw GFLOPS.

**Example Command:**

```bash
./scripts/run_standalone_ukernel.sh -a firestorm -L row -p float -v 128 -l ../llvm-project/build/ -k 512 -m 20 -n 4 -i 10000000
```

- `-a firestorm`: Target architecture (e.g., firestorm, haswell, arm, rv64).
- `-L row`: Matrix layout (`row` or `col`).
- `-p float`: Data precision.
- `-v 128`: Vector length in bits (e.g., 128 for NEON, 256 for AVX2).
- `-l ../llvm-project/build/`: Path to your compiled LLVM project.
- `-m 20 -n 4`: Micro-kernel dimensions (MR x NR).
- `-k 512`: The K dimension for the standalone dot-product loop.
- `-i 10000000`: Number of iterations for stable benchmarking.

### 2. BLIS Integration Benchmark (`run_blis.sh`)

This script integrates your custom MLIR micro-kernel directly into the BLIS framework context and benchmarks it using `bli_gemm_ex`. It outputs a comparison between the native BLIS micro-kernel and the custom MLIR generated one.

**Example Command:**

```bash
./scripts/run_blis.sh -a firestorm -L row -p float -v 128 -l ../llvm-project/build/ -b ../blis -x 1024 -y 1024 -z 1024 -m 20 -n 4
```

- `-b ../blis`: Path to your local BLIS installation. The script will automatically locate the correct include and lib directories.
- `-x 1024 -y 1024 -z 1024`: Global macro-matrix dimensions (M x N x K).
- *(Inherits `-a`, `-L`, `-p`, `-v`, `-l`, `-m`, `-n` identically to the standalone script).*

### 3. Mixed-Precision / Widening (`run_widening_ukernel.sh`)

For testing mixed-precision workloads (e.g., `f16` inputs accumulating into `f32`), use this script. It functions similarly to the standalone runner but allows for mismatched input/output types.

**Example Command:**

```bash
./scripts/run_widening_ukernel.sh -a rv64 -L col -I f16 -O f32 -v 512 -l ../llvm-project/build/ -m 16 -n 12 -z 256 -i 100
```

- `-I bf16`: Input precision type.
- `-O f32`: Output/accumulation precision type.

## Citation

If you use this infrastructure or code in your research, please cite our paper:

<!-- ```bibtex -->
<!-- @inproceedings{ribas2026microkernels, -->
<!--   title={Cross-Platform MLIR Based GEMM Micro-Kernel Generation}, -->
<!--   author={Ribas, Luc Joffily and Araujo, Guido}, -->
<!--   year={2026} -->
<!--   booktitle={...} % Update with conference/journal details -->
<!-- } -->
<!-- ``` -->

## Authors

* **Luc Joffily Ribas** - *Universidade Estadual de Campinas (Unicamp)* 
* **Guido Araujo** - *[Celera Systems](https://celera.ai/) \& Universidade Estadual de Campinas (Unicamp)*  


## License

This project is licensed under the terms of the LICENSE file included in the root directory.

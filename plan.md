# ukr-mlir vs IREE: micro-benchmark and integration plan (Nov 2026)

Goal: compare the ukr-mlir micro-kernels (now with Arm SME and strides) against IREE's CPU microkernels, then integrate them into IREE.

## Key mapping (why this is easy for f32)
IREE data-tiles matmul into pack -> linalg.mmt4d -> unpack. For f32, K0 = 1, so one mmt4d tile is:
- LHS panel: [K1][M0] contiguous = BLIS packed A (k x mr)
- RHS panel: [K1][N0] contiguous = BLIS packed B (k x nr)
- Acc tile: M0 x N0 row-major, contiguous = C with ldc = N0
- No alpha/beta: only an "accumulate" flag (beta in {0,1}, alpha = 1)
=> ukr-mlir kernel with mr = M0, nr = N0, C stride = N0 is a drop-in tile function. IREE pads in pack, so masked edges are not needed inside IREE; narrow-M tiles (M0 = 1, 2, 4) are where "MLIR Exact" kernels matter.
(verify) IREE default f32 tiles: arm64 8x8x1, AVX2 8x8x1, AVX-512 16x16x1. Paper's best M4 shape is 8x12 -> testing it needs a change to IREE's encoding tile-size table.

## Level 1: tile-level micro-benchmark (no IREE fork)
- Build IREE runtime only (no compiler) with tests/benchmarks; check `mmt4d_benchmark` in runtime/src/iree/builtins/ukernel/tools (verify target name and flags).
- Own harness: same buffers, same outer M1 x N1 loop; call IREE's `iree_uk_mmt4d(params)` vs `ukr_mmt4d(params)` (same loop, ukr-mlir tile kernel). Pin core, median of >= 20 runs, GFLOP/s and % of peak.
- Sweep: K in {64, 256, 1024}; tiles in L1 (hot) and an L2/DRAM pool (cold, like the RVV widening experiment); M0 in {1, 2, 4, 8(,16)}; ukr-mlir at IREE's shape and at its own best shape.
- Emit ukr-mlir with bare-pointer calling convention (no memref descriptors) and alwaysinline so it inlines like IREE's bitcode ukernels.

## Level 2: dispatch / end-to-end inside IREE
Configs on the same linalg.matmul shapes (square sweep + ResNet50 IM2ROW shapes from the paper):
A. data tiling + codegen (ukernels off)
B. data tiling + IREE ukernels (`--iree-llvmcpu-enable-ukernels=mmt4d`, verify flag spelling)
C. data tiling + ukr-mlir tile function
D. (Arm) `+sme` codegen path, no data tiling
Run with iree-benchmark-module, local-sync for 1 thread, local-task with pinned CPU ids for MT. Also report pack/unpack dispatch share (Tracy).

## Integration routes (ranked)
1. Fork hack for numbers: add a tile function in runtime/src/iree/builtins/ukernel/arch/<arch>/ that calls ukr-mlir bitcode (mlir-translate -> .bc) linked into the ukernel bitcode build.
2. No fork: custom dispatch via `hal.executable.objects` + preprocessing transform spec (samples/custom_dispatch/cpu/embedded/example_transform_spec.mlir). IREE stops tiling/threading that op -> suits whole-GEMM GAMM comparison more than the micro-kernel.
3. Upstream: new LLVMCPU C-bitcode ukernel framework (Benoit Jacob, June 2026: #24567, #24568, #24569, tracking #24574): `iree_codegen.inner_tiled` + `#iree_cpu.ukernel_provider`, flag `--iree-llvmcpu-enable-llvm-ukernels=inner_tiled`, one ukernel per MMA intrinsic (`iree_uk_mma_<intrinsic>`), pointer+offset ABI, bitcode resolved from objects attached to the op first. x86 seeds only so far. Pitch: MLIR-generated bitcode for arm64/RVV/SME intrinsics.

## SME notes
- No SME tile function found in IREE's legacy ukernel library; SME in IREE goes through codegen (2D scalable tiling, `--iree-llvmcpu-force-arm-streaming`, `--iree-llvmcpu-disable-arm-sme-tiling`) (verify).
- Streaming-mode entry/exit (smstart/smstop, ZA setup) per M0 x N0 tile call is too expensive: the SME kernel must own the M1 x N1 loop (replace whole mmt4d, streaming-locally + new-za), not just the tile.
- Apple M4: SVL = 512 bits. Check early that IREE's runtime on macOS runs a streaming dispatch correctly.

## Research angles vs IREE
- SME microkernel (IREE has codegen only).
- Strides: skip LHS pack / output unpack while weights stay packed at compile time; measure pack/unpack share first.
- Shape freedom (8x12 NEON, 16x13 RVV) vs IREE's fixed tile table.
- IREE also belongs as an MLIR-compiler baseline in the GAMM paper.

## Findings so far (2026-11-18, Apple M4, IREE 3.14.0rc20261115, pip wheels)
- Verified: arm64 f32 data-tiled mmt4d tile is 8x8x1 (`--iree-opt-data-tiling`, host CPU).
- Verified flags: `--iree-llvmcpu-enable-ukernels=mmt4d` (links `iree_uk_mmt4d`), `--iree-llvmcpu-enable-llvm-ukernels=inner_tiled`, `--iree-llvmcpu-force-arm-streaming`, `--iree-llvmcpu-disable-arm-sme-tiling`.
- Level 2 harness: `scripts/iree_bench.py` (configs A, B, D; C pending). 1 thread, f32 square: A ~= B ~= 109 GFLOP/s at 512-1024 (90 at 128); 4 threads 1024^3: 348 GFLOP/s.
- SME (config D) only kicks in with `--iree-llvmcpu-target-cpu=apple-m4 --iree-llvmcpu-target-cpu-features=+sme,+sme2 --iree-llvmcpu-enable-scalable-vectorization` (`host` + features is rejected). Runs correctly on macOS, but 1.4-2.7 GFLOP/s: the LHS column for each fmopa is gathered element by element through a stack spill (32 strided scalar loads per k step). `--iree-llvmcpu-vscale-range=4,4` helps (139 -> 97 ms at 512^3) but does not fix it.
- No SME with data tiling (mmt4d stays NEON) nor for an A-transposed linalg.generic (NEON). `+sve` + `--iree-llvmcpu-force-arm-streaming` crashes/hangs at runtime on M4.
- => IREE has no usable f32 SME GEMM on M4: strong case for the ukr-mlir SME kernel with packed A.

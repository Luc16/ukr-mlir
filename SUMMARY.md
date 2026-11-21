# ukr-mlir vs IREE: status and next steps (2026-11-21)

Branch `iree_compare` (nothing committed yet). The background plan and Level-2 findings are in `plan.md`. This file covers what was built, the current Apple M4 results, and what to do next.

## 1. What was built

### Generator (`lib/Ukernel.cpp`, `include/Ukernel.h`)
- **Epilogue rewrite** (`EpilogueConfig`, `ScalarKind {Dynamic, Zero, One}`):
  - `beta = zero` never reads C.
  - Dynamic beta uses a single `scf.if (beta == 0)` around the whole epilogue.
  - The edge path is masked on both read and write.
  - Integer types use `muli`/`addi`.
- **Function attributes** read by `generateBLISUkernel`:
  - `ukernel_alpha` = `one | dynamic`
  - `ukernel_beta` = `zero | one | dynamic`
  - `ukernel_edges` = bool, default true
  - `ukernel_k_split` = 1..16
- **K-split:** `kSplit × numAccs` accumulators. Unrolled step `u` updates set `u % kSplit`, and a pairwise reduction tree runs after the K loop.
  - It is chosen automatically: double `kSplit` (up to 8) while `kSplit·numAccs < fma_units·fma_latency` and the registers fit.
  - `--k-split=1` restores the old behaviour.
- **New `transform-opt` options:** `--num-vecs` (default 16), `--fma-units` (default 2), `--fma-latency` (default 5). For M4, use 32 / 4 / 4.

### Generator front-ends
- `scripts/blis_ukernel_generator.py` options:
  - `--alpha=`, `--beta=`, `--edges=`, `--name=`, `--k-split=`
  - `--entry=bare`: emits `@{name}_tile(ptr out, ptr lhs, ptr rhs, i64 k)`. After `mlir-opt -inline -canonicalize`, it inlines the kernel, so no memref descriptors survive. It requires `alpha=one` and a static beta.
- `scripts/blis_ukernel_generator.sh` options: `-A` (alpha), `-B` (beta), `-E` (edges).

### Tile-level comparison with IREE ("Level 1")
- **`scripts/iree_tile_bench.sh`:** compiles IREE's arm64 mmt4d tile functions straight from `external/iree` (plain C, no IREE build needed), generates the ukr kernels, and builds and runs the harness.
  - Machine-model flags: `-r` vector registers, `-u` FMA pipes, `-t` FMA latency, `-c` mcpu.
  - Benchmark arguments go after `--`.
- **`test/iree_tile_bench.c`:**
  - Uses the same outer loop as IREE's `iree_uk_mmt4d_using_tile_func`.
  - Correctness check runs first.
  - Measured FMA peak: 20 chains, about 141 GFLOP/s.
  - Three modes: hot (1 tile), warm (8×8 tiles reused), cold (256 MB arena).
  - Samples are interleaved round-robin and reported as median, min, max and MAD. Output is CSV.
- **Kernel variants compared:**
  - `iree`: IREE 1/2/4/8×8
  - `ukr`: bare entry plus K-split, 1/2/4/8×8 and 8×12
  - `ukr-v1`: C interface, no K-split
  - `ukr-blis`: 8×8 with dynamic alpha/beta and edges
- **Plots:** `plots/plot_iree_tile.py --runs <dir> --out <dir>`, run with `.venv/bin/python`.
  - `plots/iree_tile/`: current results, from `tmp/iree_tile/runs_v2`
  - `plots/iree_tile_v1/`: first comparison, from `tmp/iree_tile/runs`
  - `tmp/` is gitignored, so copy the CSVs somewhere tracked if you want to keep them.

Reproduce one run:
```
scripts/iree_tile_bench.sh -o tmp/iree_tile/runs_v2/run_1.csv -- --k=64,256,1024 --modes=hot,warm,cold --reps=15
```

## 2. Current results on Apple M4

Setup:
- IREE `8071457` (2026-11-18), f32, 1 thread, QoS user-interactive.
- 5 independent runs × 15 interleaved samples each. Values are the median of the per-run medians, in GFLOP/s.
- Measured FMA peak: 134–141 GFLOP/s.
- Run-to-run spread: at most 4.7%, and 23 of 252 rows are above 3%.

### Warm mode, no accumulate (the headline)

| Tile | K | IREE | ukr v1 (no K-split) | ukr (current) | ukr / IREE |
|---|---|---|---|---|---|
| 1×8 | 64 / 256 / 1024 | 35.3 / 24.8 / 21.0 | 32.7 / 23.9 / 18.7 | 68.6 / 70.4 / 51.7 | **1.9× / 2.8× / 2.5×** |
| 2×8 | 64 / 256 / 1024 | 57.6 / 42.3 / 36.7 | 50.2 / 40.7 / 36.1 | 101.1 / 110.0 / 89.2 | **1.8× / 2.6× / 2.4×** |
| 4×8 | 64 / 256 / 1024 | 91.2 / 72.7 / 68.4 | 80.4 / 70.3 / 67.1 | 116.4 / 116.9 / 114.9 | **1.28× / 1.6× / 1.7×** |
| 8×8 | 64 / 256 / 1024 | 119.3 / 117.3 / 115.9 | 116.6 / 116.5 / 114.8 | 117.9 / 116.6 / 116.1 | ~1.0× (tie) |
| 8×12 | 64 / 256 / 1024 | (vs IREE 8×8) | n/a | 126.8 / 127.1 / 126.5 | **1.06–1.10×** vs IREE 8×8 |

`ukr-blis` 8×8 (dynamic alpha/beta, edges on) gets 115–117 GFLOP/s warm. The generic BLIS-style signature costs almost nothing at 8×8.

### Cold mode, no accumulate (256 MB arena)

| Tile | K = 64 / 256 / 1024: IREE | ukr |
|---|---|---|
| 1×8 | 32.5 / 23.6 / 20.3 | 56.9 / 56.7 / 47.0 |
| 2×8 | 52.7 / 40.4 / 35.7 | 85.0 / 90.1 / 83.3 |
| 4×8 | 78.5 / 68.8 / 67.3 | 96.6 / 106.8 / 111.5 |
| 8×8 | 109.6 / 113.1 / 112.6 | 108.5 / 112.5 / 113.0 |
| 8×12 | n/a | 117.0 / 122.4 / 123.9 |

### Accumulate path
- Warm and cold modes look the same as without accumulate.
- **Hot mode, accumulate:** IREE's narrow tiles drop sharply (for 1×8, 34.8 → 19.2 at K=64; 4×8: 89.6 → 66.1). The likely cause is a store-to-load dependency between consecutive calls on the same tile. ukr does not drop, because it reads C once in the epilogue. In the heatmap this is marked with `*`.

### Why (diagnosis)
- M4 FMA latency is 4 cycles and there are 4 FMA pipes, so about 16–20 independent chains are needed to reach peak.
- Measured: 1 / 2 / 4 / 8 / 16 / 20 chains gave 8.8 / 17.5 / 35 / 70 / 126 / 142 GFLOP/s.
- IREE's 1×8, 2×8 and 4×8 tiles have only 2, 4 and 8 accumulators, so they are latency-bound at about 17.5, 35 and 70 GFLOP/s per K-step chain. Old ukr was latency-bound the same way.
- **K-split explains almost all of the gain.**
  - Attribution test: the bare entry alone gives +7–16% on narrow tiles at K=64, and nothing elsewhere.
  - 8×12 (24 accumulators) and 8×8 (16) do not get a K-split. They already have enough chains.
- 1×8 at K=1024 warm (52 vs 71 hot) is limited by the memory and L1 working set, not by FMAs.

### Caveats
- K-split changes the order of FP summation along K. Results are not bitwise identical to IREE, but they stay within tolerance. This needs a note in any upstream pitch.
- Default BLIS kernels can now get a K-split too (for example, the haswell 6×8 gets 2 sets). Check that the paper numbers still reproduce, or pass `--k-split=1`.
- These are tile-level results only. In a real IREE matmul the narrow tiles cover only the M remainder. The end-to-end gain depends on the shape mix (small-M, GEMV-like, decode-style).

## 3. Setting up IREE for the tests

There are two IREE pieces, and the two test levels need different ones:

| Test | Needs | Builds IREE? |
|---|---|---|
| Tile level (`scripts/iree_tile_bench.sh`) | IREE **source** (only the C ukernel files) | No |
| Dispatch level (`scripts/iree_bench.py`, configs A/B/D) | IREE **compiler + runtime tools** (pip wheels) | No |
| Fork prototype (config C, step 4.C) | IREE source **built from scratch** | Yes |

Both test levels also need this repo's `transform-opt` (`./build.sh -l <llvm-build>`) and the LLVM tools from the same LLVM build (`mlir-opt`, `mlir-translate`, `llc`). Pass them with `-l`, or put them on `PATH`.

### 3.1 IREE source (tile-level comparison)
```bash
git clone --depth 1 https://github.com/iree-org/iree external/iree
# Results in this file used 8071457 (2026-11-18). To pin it:
#   git -C external/iree fetch --depth 1 origin 80714577309e0530127e7e3943ff281d6253a209
#   git -C external/iree checkout 80714577309e0530127e7e3943ff281d6253a209
```
- No submodules and no CMake. The script only compiles `runtime/src/iree/builtins/ukernel/arch/arm_64/mmt4d_arm_64_base.c`.
- It writes `config_arm_64.h` itself: it copies `config_arm_64.h.in` with the `#cmakedefine` lines removed, which turns off the optional fp16/bf16/dotprod/i8mm paths. f32 does not use them.
- `external/` is gitignored.

Run:
```bash
scripts/iree_tile_bench.sh -l /path/to/llvm-project/build -- --check-only   # correctness only
for i in 1 2 3 4 5; do
  scripts/iree_tile_bench.sh -l /path/to/llvm-project/build \
      -o tmp/iree_tile/runs_v2/run_$i.csv -- --k=64,256,1024 --modes=hot,warm,cold --reps=15
done
.venv/bin/python plots/plot_iree_tile.py --runs tmp/iree_tile/runs_v2 --out plots/iree_tile
```
- Close other applications while it runs. The harness raises its QoS class on macOS. Each full run takes a few minutes.
- If the measured peak printed at the top of the log is far below about 140 GFLOP/s, the machine was busy or throttled. Rerun.
- If you add a kernel shape, add it in **both** places: `scripts/iree_tile_bench.sh` (generation) and `UKR_SHAPES` / `UKR_V1_SHAPES` in `test/iree_tile_bench.c`.

### 3.2 IREE tools (dispatch-level comparison)
```bash
python3 -m venv ~/iree-venv
~/iree-venv/bin/pip install --pre -f https://iree.dev/pip-release-links.html \
    iree-base-compiler iree-base-runtime numpy
python3 scripts/iree_bench.py --iree-bin ~/iree-venv/bin -c A,B,D -s 256,512,1024 -o tmp/iree/results.csv
```
- The `plan.md` numbers used `3.14.0rc20261115`. Pin it with `iree-base-compiler==3.14.0rc20261115 iree-base-runtime==3.14.0rc20261115`.
- **Local quirk:** this repo's `.venv` was copied from `~/Documents/UKernels/.venv`. Its `pip` still installs into the old venv, and `.venv/bin/python` cannot `import iree`. The IREE tools live in `~/Documents/UKernels/.venv/bin`. Pass that as `--iree-bin`, or make a fresh venv as above. Plotting only needs `.venv` (matplotlib, pandas).
- Check that the flags still exist after an upgrade: `iree-compile --help | grep -E 'enable-ukernels|enable-llvm-ukernels|data-tiling'`.

### 3.3 x86 on sorgan (for step 4.B)
- Needed on the cluster: an LLVM/MLIR build (the same LLVM revision this repo builds against), `transform-opt` built with it, and the IREE source checkout from 3.1. The pip wheels are only needed for the dispatch level (Linux x86_64 wheels exist).
- `iree_tile_bench.sh` and `test/iree_tile_bench.c` are arm64-only for now:
  - IREE file: `arch/x86_64/mmt4d_x86_64_avx2_fma.c` (AVX2, 8x8 tiles) and `mmt4d_x86_64_avx512_base.c` (AVX-512, 16x16 tiles plus 1/2/4/8x16 narrow ones). Generate `config_x86_64.h` from its `.in` the same way as the arm one.
  - Tile entry names: `iree_uk_mmt4d_tile_f32f32f32_8x8x1_x86_64_avx2_fma` and so on.
  - The NEON peak loop needs an AVX2/AVX-512 version.
  - Replace the macOS QoS call with `sched_setaffinity`.
- Run through Slurm on a whole node (`--exclusive`) so that other jobs don't skew the numbers. Record the CPU model, and whether turbo or frequency scaling is on.

## 4. Next steps (in order)

### A. Short-term (this machine)
1. **Commit the current work** (generator, scripts, harness, plots). Store the run CSVs somewhere tracked (for example `results/iree_tile_m4/`).
2. **Exact narrow shapes:** generate 3×8, 5×8, 6×8 and 7×8, and compare them with IREE's decomposition into 1/2/4×8 tiles (shape freedom argument).
3. **Level 2, mmt4d-only:** compare IREE codegen (A) vs IREE ukernel (B) on the mmt4d dispatch alone (`scripts/iree_bench.py`), so the tile results can be read in context. Measure the pack/unpack share with Tracy.
4. **K-split for IREE's own tile functions:** a fair control. Patch IREE's `mmt4d_arm_64_base.c` 1/2/4×8 tiles with 2–4 accumulator sets. If they catch up, the contribution is the generator's machine model that finds this automatically, not the kernel itself.
5. Tune the K-split heuristic, for example the cap of 8, or extending it to 8×8 at small K (IREE has a slight edge at K=64).

### B. x86 on sorgan (LSC Unicamp cluster)
- SSH works: `ssh sorgan` (jump host `lsc`), with Slurm. Partitions seen: `agpu`, `gpu`, `ib`. Run `sinfo` to find the Haswell and Skylake nodes.
- Port the harness:
  - Compile IREE's `arch/x86_64` mmt4d tiles (AVX2 8×8, AVX-512 16×16).
  - Replace the macOS QoS call with `taskset` / `sched_setaffinity`.
  - Peak loop: 2 FMA ports × 4–5 cycles latency means about 8–10 chains.
  - Generator flags: `-v 256 -r 16 -u 2 -t 5` (Haswell), `-v 512 -r 32 -u 2 -t 4` (Skylake-SP).
- Expect smaller K-split gains on narrow tiles: fewer chains are needed, but there are also fewer registers.

### C. Config C: ukr-mlir inside IREE (fork prototype)
1. Add a tile function in `runtime/src/iree/builtins/ukernel/arch/arm_64/` (and `arch/x86_64/`) that calls the ukr-mlir object or bitcode (`mlir-translate` → `.bc`, linked into the ukernel bitcode build).
2. Add 8×12 and exact narrow tiles to `enumerateMatmulTileArm64` (`CPUEncodingExternalModels.cpp:1370`). Pack has fast paths only for N0 = 8/1, so check pack cost for N0 = 12.
3. Watch out for `ALLOW_GENERIC_FALLBACK_TILE_FUNCTION`: a missing tile function silently falls back to scalar code. Assert in the harness that your function is the one being called.
4. Run end-to-end on the matmul sweep and the ResNet50 IM2ROW shapes: A vs B vs C, on the M4 and (with the x86 tile functions from step 4.B) on Haswell and Skylake.

### D. Upstream route
- Target the new `inner_tiled` LLVM-ukernel framework (`#iree_cpu.ukernel_provider`, `--iree-llvmcpu-enable-llvm-ukernels=inner_tiled`, PRs #24567–24569, tracking issue #24574). Its pointer-plus-offset ABI maps closely onto `--entry=bare`.
- Pitch: MLIR-generated bitcode for arm64, RVV and SME shapes that IREE does not hand-write. Use the narrow-tile and 8×12 results as evidence.
- Open a discussion with maintainers (Benoit Jacob) before writing a large PR.

### E. Research angles
- **SME:** IREE has no usable f32 SME GEMM on M4 (see `plan.md`, 1.4–2.7 GFLOP/s). The SME kernel must own the M1×N1 loop, because streaming-mode entry per tile is too expensive.
- **RVV:** riscv64 uses codegen unless `+xsmtvdot` is enabled, so there is room for generated kernels (for example 16×13).
- **Strides:** skip the LHS pack and output unpack while weights stay packed.
- Add IREE as an MLIR-compiler baseline in the GAMM paper.

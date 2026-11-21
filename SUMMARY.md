# ukr-mlir vs IREE: status and next steps (2026-11-21)

Branch `iree_compare`. The M4 work is committed (`adcab79`); the x86 port, its results and the cross-architecture plots are not yet. The background plan and Level-2 findings are in `plan.md`. This file covers what was built, the tile-level results on Apple M4 and on two x86 machines, and what to do next.

## 0. Takeaways so far

Tile-level only (one micro-kernel call inside IREE's M1×N1 outer loop), f32, 1 thread, IREE `8071457`, on three ISAs:

- **K-split on narrow tiles is the strongest and most consistent result.** IREE's 1/2/4-row tiles have too few accumulators to hide the FMA latency. The generator splits K over several accumulator sets, chosen from the machine model (FMA units × latency).
  - In warm mode, 1-row tiles get about 1.9–2.8× on M4, 1.8–2.1× on AVX2 and 1.6–1.7× on AVX-512.
  - ukr-v1 (no K-split) is at or below IREE on every tile, so the gain comes from the K-split.
- **Larger shapes than IREE ships beat IREE's largest tile:**
  - 8×12 vs 8×8: 1.06–1.10× on M4.
  - 6×16 vs 8×8: 1.2–1.35× on AVX2.
  - 12×32 vs 16×16: 1.4–1.6× on AVX-512, reaching 93–95% of the FMA peak.
  - On x86 the main cause is the ratio of loads to FMAs per K step, not register use as such:
    - AVX2 8×8 does 9 loads per 8 FMAs; 6×16 does 8 per 12.
    - x86 has only 2 load ports.
- **At the same shape ukr is also faster on AVX-512:** 16×16 is 1.18–1.32× IREE's 16×16 in warm mode. IREE's 16×16 drops sharply as K grows; the cause has not been measured.
- **What this does not show yet:**
  - End-to-end benefit. IREE picks tile shapes for more than the kernel: pack/unpack cost, padding waste, and splitting the M remainder into power-of-two narrow tiles. Some of the tile-level gain may disappear once those count.
  - Whether IREE could simply add a K-split by hand (step 4.A.4).
  - The claim worth making: "the generator approach, with a machine-model-driven K-split and free choice of shape, beats IREE's hand-written tiles where they are weakest". The paper's original generator alone (ukr-v1) does not beat IREE.

Plots across all three architectures are in `plots/iree_tile_arch/`. Generate them with `python3 plots/plot_iree_tile_arch.py` (no venv needed, only matplotlib):
- `1_ksplit_narrow.png`: IREE vs ukr-v1 vs ukr on narrow tiles. One row per architecture, one column per K, warm mode.
- `2_larger_shapes.png`: IREE's largest tile vs ukr at the same shape and at the larger shape, warm and cold.
- `3_speedup_heatmap.png`: ukr / IREE for every tile, K and mode.

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
  - It is chosen automatically: double `kSplit` (up to 8) while `kSplit·numAccs < fma_units·fma_latency` and `2·kSplit·numAccs <= num_vecs − vecsPerStep − 2`.
  - `--k-split=1` restores the old behaviour.
  - On x86 (8 chains needed), every narrow tile reaches 8 accumulators. For example, 1×8 AVX2 gets kSplit = 8, and 4×16 AVX-512 gets kSplit = 2.
- **New `transform-opt` options:** `--num-vecs` (default 16), `--fma-units` (default 2), `--fma-latency` (default 5).
  - M4: 32 / 4 / 4
  - AVX2 (Coffee Lake): 16 / 2 / 4
  - AVX-512 (Cascade Lake): 32 / 2 / 4

### Generator front-ends
- `scripts/blis_ukernel_generator.py` options:
  - `--alpha=`, `--beta=`, `--edges=`, `--name=`, `--k-split=`
  - `--entry=bare`: emits `@{name}_tile(ptr out, ptr lhs, ptr rhs, i64 k)`. After `mlir-opt -inline -canonicalize`, it inlines the kernel, so no memref descriptors survive. It requires `alpha=one` and a static beta.
- `scripts/blis_ukernel_generator.sh` options: `-A` (alpha), `-B` (beta), `-E` (edges).

### Tile-level comparison with IREE ("Level 1")
- **`scripts/iree_tile_bench.sh`:** compiles IREE's mmt4d tile functions straight from `external/iree` (plain C, no IREE build needed), generates the ukr kernels, and builds and runs the harness.
  - **`-I neon|avx2|avx512`** picks the ISA. It is auto-detected if omitted. Each ISA sets its own:

    | ISA | IREE file | Shapes | Defaults `-a/-v/-r/-u/-t` |
    |---|---|---|---|
    | neon | `arm_64/mmt4d_arm_64_base.c` | IREE 1/2/4/8×8, extra 8×12, BLIS 8×8 | firestorm / 128 / 32 / 4 / 4 |
    | avx2 | `x86_64/mmt4d_x86_64_avx2_fma.c` | IREE 1/2/4/8×8, extra 6×16, BLIS 8×8 | haswell / 256 / 16 / 2 / 4 |
    | avx512 | `x86_64/mmt4d_x86_64_avx512_base.c` | IREE 1/2/4/8/16×16, extra 12×32 and 14×32, BLIS 16×16 | skx / 512 / 32 / 2 / 4 |

  - `-c` is the target CPU: `-mcpu` for clang on arm64, `-march` for clang on x86, and `-mcpu` for llc in both cases.
  - `-b` sets the build directory (default `tmp/iree_tile/build_<isa>`).
  - Benchmark arguments go after `--`.
- **`scripts/iree_tile_bench.sbatch`** (sorgan):
  - Takes the whole node with `--exclusive`.
  - Writes `machine.txt` (lscpu, governor, turbo, git revisions).
  - Does `RUNS` (default 5) runs into `results/iree_tile_x86/<node>/`.
- **`test/iree_tile_bench.c`:**
  - Uses the same outer loop as IREE's `iree_uk_mmt4d_using_tile_func`.
  - The ISA is chosen at compile time: `UKR_ISA_AVX2`, `UKR_ISA_AVX512`, or NEON on aarch64. Shape lists are the X-macros `IREE_SHAPES`, `UKR_EXTRA_SHAPES` and `BLIS_M0/N0`. If you add a shape, add it in both this file and the script.
  - Correctness check runs first.
  - FMA peak loop per ISA:
    - NEON: 20 chains, about 141 GFLOP/s.
    - AVX2: 12 ymm chains.
    - AVX-512: 16 zmm chains.
  - On Linux the process is pinned to one CPU with `sched_setaffinity` (the first allowed CPU, or `UKR_PIN_CPU`). On macOS it raises its QoS class instead.
  - Three modes: hot (1 tile), warm (8×8 tiles reused), cold (256 MB arena).
  - Samples are interleaved round-robin and reported as median, min, max and MAD. Output is CSV.
- **Kernel variants compared:**
  - `iree`: IREE's tiles
  - `ukr`: bare entry plus K-split, same shapes as IREE plus the extra ones
  - `ukr-v1`: C interface, no K-split
  - `ukr-blis`: dynamic alpha/beta and edges
- **Plots:**
  - `plots/plot_iree_tile.py --runs <dir> --out <dir>` (M4 only):
    - `plots/iree_tile/`: current results, from `tmp/iree_tile/runs_v2`
    - `plots/iree_tile_v1/`: first comparison, from `tmp/iree_tile/runs`
  - `plots/plot_iree_tile_arch.py`: all architectures (see section 0). The data directories are listed in `DEFAULT_ARCHS`.
- **Data:**
  - `results/iree_tile_m4/run_{1..5}.csv` holds the M4 runs behind section 2, copied from the Mac's `tmp/iree_tile/runs_v2`.
  - `results/iree_tile_x86/<node>/` holds the x86 runs, plus `machine.txt` and the logs.
  - The first M4 comparison (`plots/iree_tile_v1/`) is still only in `tmp/iree_tile/runs`.

Reproduce one M4 run:
```
scripts/iree_tile_bench.sh -o tmp/iree_tile/runs_v2/run_1.csv -- --k=64,256,1024 --modes=hot,warm,cold --reps=15
```

## 2. Results on Apple M4

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

Hot mode, no accumulate (K = 64 / 256 / 1024):
- **ukr / IREE:**
  - 1×8: 1.74 / 2.80 / 3.39×
  - 2×8: 1.74 / 2.61 / 3.11×
  - 4×8: 1.29 / 1.61 / 1.66×
  - 8×8: tie
  - 8×12 vs IREE 8×8: 1.07 / 1.09 / 1.09×
- **Why the gain grows with K in hot mode:** in warm mode, 1×8 drops at K=1024 because of the working set. In hot mode the working set stays small, so ukr 1×8 keeps rising (60.5 → 69.1 → 71.4 GFLOP/s) while IREE falls (34.8 → 24.7 → 21.1).

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

## 3. Results on x86 (sorgan, 2026-11-21)

Setup:
- Same IREE commit, harness and aggregation as on M4: 5 runs × 15 samples, median of per-run medians.
- Built with clang 19 and the sorgan MLIR build (`~/llvm-project/build`, 2026-03-30).
- Whole node (`--exclusive`), pinned to CPU 0.
- Governor `performance`, turbo on.

| Node | CPU | ISA, flags | Measured FMA peak |
|---|---|---|---|
| sorgan-cpu2 | Xeon Gold 6252 (Cascade Lake, 24C) | AVX-512, `-c cascadelake`, 32 regs / 2 FMA / lat 4 | 219–221 GFLOP/s (about 3.45 GHz AVX-512 clock) |
| sorgan-cpu6 | Xeon E-2224 (Coffee Lake, 4C) | AVX2, `-c skylake`, 16 regs / 2 FMA / lat 4 | 146 GFLOP/s (about 4.57 GHz) |
| sorgan-cpu5 | Xeon E-2244G (Coffee Lake, 4C/8T) | same as cpu6 | 143 GFLOP/s; ratios agree with cpu6 within a few % |

Run-to-run spread: the medians are stable. A few rows (hot mode, K=64) have one outlier run with up to 13–20% spread.

### AVX-512 (cpu2), warm mode, no accumulate, K = 64 / 256 / 1024

| Tile | IREE | ukr-v1 | ukr | ukr / IREE |
|---|---|---|---|---|
| 1×16 | 42.7 / 31.2 / 28.8 | 22.9 / 26.6 / 27.9 | 69.3 / 49.8 / 48.3 | **1.6–1.7×** |
| 2×16 | 68.8 / 57.9 / 55.4 | 44.4 / 52.8 / 55.7 | 97.8 / 88.3 / 84.9 | **1.4–1.5×** |
| 4×16 | 112.8 / 108.6 / 109.7 | 84.6 / 101.6 / 101.2 | 121.2 / 120.8 / 102.2 | 1.07 / 1.11 / 0.93× |
| 8×16 | 128.6 / 128.8 / 126.9 | 128.6 / 150.1 / 143.3 | 150.6 / 156.7 / 145.7 | 1.15–1.22× |
| 16×16 | 141.7 / 133.4 / 115.8 | 147.6 / 158.2 / 152.0 | 167.7 / 166.8 / 152.8 | 1.18–1.32× |
| **12×32** | (vs IREE 16×16) | n/a | **204.0 / 209.4 / 163.1** | **1.41–1.57×** |
| 14×32 | (vs IREE 16×16) | n/a | 155.7 / 158.8 / 141.1 | spills (see below) |

`ukr-blis` 16×16: 146–159 GFLOP/s warm, about the same as ukr-v1.

### AVX2 (cpu6), warm mode, no accumulate, K = 64 / 256 / 1024

| Tile | IREE | ukr-v1 | ukr | ukr / IREE |
|---|---|---|---|---|
| 1×8 | 29.0 / 19.9 / 18.6 | 14.8 / 17.2 / 18.0 | 61.6 / 39.6 / 33.6 | **1.8–2.1×** |
| 2×8 | 46.6 / 38.2 / 36.6 | 29.3 / 34.3 / 35.9 | 82.1 / 65.1 / 56.8 | **1.55–1.76×** |
| 4×8 | 76.3 / 72.1 / 69.7 | 58.2 / 68.2 / 70.0 | 98.6 / 88.6 / 74.8 | 1.07–1.29× |
| 8×8 | 114.5 / 106.9 / 93.1 | 100.6 / 110.1 / 95.6 | 120.4 / 114.6 / 96.8 | 1.04–1.07× |
| **6×16** | (vs IREE 8×8) | n/a | **138.8 / 140.0 / 125.5** | **1.21–1.35×** |

`ukr-blis` 8×8: 96–110 GFLOP/s warm, about the same as ukr-v1.

### Other modes and notes
- **Hot mode** shows the largest narrow-tile gains: up to 3.4× for 1×8 and 3.2× for 1×16 at K=256. 6×16 reaches 141–144 GFLOP/s, 97–98% of the AVX2 peak.
- **Cold mode** shrinks the gains to about 1.1–1.3×. AVX2 1×8 stays at 1.5–1.7× and 2×8 at about 1.45×.
- **Accumulate:** IREE's 1×16 drops in hot mode (33.8 → 27.8 at K=64), the same store-to-load effect as on M4 but weaker. ukr doesn't drop.
- **Narrow tiles at K=1024:** they become L1/L2-bandwidth bound, even in hot mode. 1×16 falls from 95 to 49 GFLOP/s, and 1×8 AVX2 from 64 to 39. On M4, 1×8 drops only in warm mode (70 → 52) and keeps rising in hot mode, so M4's bigger L1 (128 KB vs 32 KB) holds the K=1024 panel.
- **4×16 at K=1024 is about 6% slower than IREE** (kSplit = 2). This was judged acceptable for now; revisit if it shows up in full-matrix tests.
- **14×32 spills one accumulator inside the K loop.** 28 accumulators fit the generator's register budget (`num_vecs − vecsPerStep − 2`), but LLVM needs one more register, so 14×32 is slower than 12×32. The other kernels keep all accumulators in registers; their only stack traffic is setup in the C-interface wrapper, outside the K loop.
- **ukr-v1 is slower than IREE on narrow x86 tiles** (for example, 1×16 at K=64 warm: 22.9 vs 42.7). Part of that is the C-interface call overhead per tile, which the bare entry removes.

### Diagnosis (x86)
- Both machines measured an FMA latency of 4 cycles with 2 FMA ports, so 8 independent chains are needed.
- Microbenchmark on cpu5/6: 1 / 2 / 4 / 6 / 8 / 10 chains gave 0.25 / 0.50 / 1.00 / 1.50 / 2.00 / 2.00 FMA/cycle. A single long chain of back-to-back FMAs in one asm block reads 4.5–5 cycles on Coffee Lake, but kernels never run that pattern.
- IREE's narrow tiles have 1, 2 or 4 accumulators, so they are latency-bound at 1/8, 2/8 and 4/8 of peak.
- IREE's largest tiles have 8 accumulators (AVX2 8×8) and 16 (AVX-512 16×16). Each K step does one load per FMA, a broadcast plus the RHS load, so they run close to the 2-load-port limit.
- 6×16 and 12×32 use 2 RHS loads plus 6 or 12 broadcasts for 12 or 24 FMAs.

## 4. Setting up IREE for the tests

There are two IREE pieces, and the two test levels need different ones:

| Test | Needs | Builds IREE? |
|---|---|---|
| Tile level (`scripts/iree_tile_bench.sh`) | IREE **source** (only the C ukernel files) | No |
| Dispatch level (`scripts/iree_bench.py`, configs A/B/D) | IREE **compiler + runtime tools** (pip wheels) | No |
| Fork prototype (config C, step 5.C) | IREE source **built from scratch** | Yes |

Both test levels also need this repo's `transform-opt` (`./build.sh -l <llvm-build>`) and the LLVM tools from the same LLVM build (`mlir-opt`, `mlir-translate`, `llc`). Pass them with `-l`, or put them on `PATH`.

### 4.1 IREE source (tile-level comparison)
```bash
git clone --depth 1 https://github.com/iree-org/iree external/iree
# Results in this file used 8071457 (2026-11-18). To pin it:
#   git -C external/iree fetch --depth 1 origin 80714577309e0530127e7e3943ff281d6253a209
#   git -C external/iree checkout 80714577309e0530127e7e3943ff281d6253a209
# Or a small sparse checkout (what sorgan uses):
#   git clone --filter=blob:none --no-checkout https://github.com/iree-org/iree external/iree
#   git -C external/iree sparse-checkout set runtime/src/iree/builtins/ukernel runtime/src/iree/base runtime/src/iree/schemas
#   git -C external/iree checkout 80714577309e0530127e7e3943ff281d6253a209
```
- No submodules and no CMake. The script compiles only the one mmt4d file for the chosen ISA.
- It writes `config_<arch>.h` itself: it copies the `.h.in` with the `#cmakedefine` lines removed, which turns off the optional paths (fp16/bf16/dotprod/i8mm on arm, VNNI/BF16 on x86). f32 does not use them.
- `external/` is gitignored.

Run on M4:
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
- If you add a kernel shape, add it in **both** places: the per-ISA `case` in `scripts/iree_tile_bench.sh` and the X-macros at the top of `test/iree_tile_bench.c`.

### 4.2 IREE tools (dispatch-level comparison)
```bash
python3 -m venv ~/iree-venv
~/iree-venv/bin/pip install --pre -f https://iree.dev/pip-release-links.html \
    iree-base-compiler iree-base-runtime numpy
python3 scripts/iree_bench.py --iree-bin ~/iree-venv/bin -c A,B,D -s 256,512,1024 -o tmp/iree/results.csv
```
- The `plan.md` numbers used `3.14.0rc20261115`. Pin it with `iree-base-compiler==3.14.0rc20261115 iree-base-runtime==3.14.0rc20261115`.
- **Local quirk (Mac):** this repo's `.venv` was copied from `~/Documents/UKernels/.venv`. Its `pip` still installs into the old venv, and `.venv/bin/python` cannot `import iree`. The IREE tools live in `~/Documents/UKernels/.venv/bin`. Pass that as `--iree-bin`, or make a fresh venv as above. Plotting only needs matplotlib, and the system `python3` on the Linux desktop works.
- Check that the flags still exist after an upgrade: `iree-compile --help | grep -E 'enable-ukernels|enable-llvm-ukernels|data-tiling'`.

### 4.3 x86 on sorgan (LSC Unicamp cluster)
- Access: `ssh sorgan` (jump host `lsc`), Slurm. CPU nodes are in partition `cpu`; cpu2 is also alone in `debug`.
  - cpu1–3: Xeon Gold 6252 (Cascade Lake, AVX-512). cpu3 also has an MI210 and is often allocated.
  - cpu5: Xeon E-2244G. cpu6: Xeon E-2224. Both are Coffee Lake, AVX2 only. **There is no Haswell node.**
  - bigmem: 32 cores.
- Already set up:
  - `~/ukr-mlir` (branch `iree_compare`; the x86 changes were copied over and are uncommitted there too)
  - sparse IREE at `8071457` in `external/iree`
  - `transform-opt` built against `~/llvm-project/build`
- The `module` command exists only on the login node, under `bash -lc`. Compute nodes don't have it, so the sbatch script puts spack's clang 19 on `PATH` directly.
- Run:
  ```bash
  cd ~/ukr-mlir
  sbatch -w sorgan-cpu2 --export=ALL,ISA=avx512,CPU=cascadelake scripts/iree_tile_bench.sbatch
  sbatch -w sorgan-cpu6 --export=ALL,ISA=avx2,CPU=skylake       scripts/iree_tile_bench.sbatch
  # results in results/iree_tile_x86/<node>/; copy back with
  #   rsync -a sorgan:ukr-mlir/results/iree_tile_x86/ results/iree_tile_x86/
  ```
  One job (5 runs) takes about 8–10 minutes. A correctness-only check: `srun -p cpu -w <node> bash -c "export PATH=<clang dir>:\$PATH; scripts/iree_tile_bench.sh -l ~/llvm-project/build -I avx2 -c skylake -- --check-only"`.

## 5. Next steps (in order)

### A. Short-term
1. **Commit the x86 work:** `test/iree_tile_bench.c`, `scripts/iree_tile_bench.sh`, `scripts/iree_tile_bench.sbatch`, `plots/plot_iree_tile_arch.py`, `plots/iree_tile_arch/`, `results/`. Also delete the stray `plots/__pycache__/*.pyc` from `adcab79` and gitignore it.
2. **Re-check the NEON build** of the ported harness on the Mac (`scripts/iree_tile_bench.sh -- --check-only`). It was not rebuilt after the x86 changes.
3. **Exact narrow shapes:** generate 3×8, 5×8, 6×8 and 7×8 (and the ×16 equivalents on AVX-512), and compare them with IREE's decomposition into power-of-two tiles (shape freedom argument).
4. **K-split for IREE's own tile functions:** a fair control. Patch IREE's 1/2/4-row tiles (`mmt4d_arm_64_base.c`, `mmt4d_x86_64_avx2_fma.c`, `mmt4d_x86_64_avx512_base.c`) with 2–8 accumulator sets. If they catch up, the contribution is the generator's machine model that finds this automatically, not the kernel itself.
5. **Register budget:** reserve one more register in the K-split/shape budget, or detect spills, so that 14×32-style shapes don't spill. Then retry 14×32 on AVX-512.
6. **Level 2, mmt4d-only:** compare IREE codegen (A) vs IREE ukernel (B) on the mmt4d dispatch alone (`scripts/iree_bench.py`), so the tile results can be read in context. Measure the pack/unpack share with Tracy, and look at what padding to 12×32 or 6×16 would cost.
7. Leave 4×16 at K=1024 (−6%) and the K-split cap of 8 alone unless full-matrix tests show they matter.

### B. x86 follow-ups
- Run the dispatch-level comparison (`scripts/iree_bench.py`) on sorgan. Linux x86_64 IREE wheels exist.
- Optional: compare IREE's AVX2 tiles with ukr on the Cascade Lake node (AVX2 code on an AVX-512 machine). This would separate the ISA effect from the microarchitecture effect.

### C. Config C: ukr-mlir inside IREE (fork prototype)
1. Add a tile function in `runtime/src/iree/builtins/ukernel/arch/arm_64/` and `arch/x86_64/` that calls the ukr-mlir object or bitcode (`mlir-translate` → `.bc`, linked into the ukernel bitcode build).
2. Add the larger tiles (8×12 on arm64, 6×16 for AVX2, 12×32 for AVX-512) and exact narrow tiles to `enumerateMatmulTileArm64` (`CPUEncodingExternalModels.cpp:1370`) and the x86 equivalent. Pack has fast paths only for some N0 (on arm, 8 and 1), so check pack cost for the new N0 values.
3. Watch out for `ALLOW_GENERIC_FALLBACK_TILE_FUNCTION`: a missing tile function silently falls back to scalar code. Assert in the harness that your function is the one being called.
4. Run end-to-end on the matmul sweep and the ResNet50 IM2ROW shapes: A vs B vs C, on the M4, Coffee Lake and Cascade Lake.

### D. Upstream route
- Target the new `inner_tiled` LLVM-ukernel framework (`#iree_cpu.ukernel_provider`, `--iree-llvmcpu-enable-llvm-ukernels=inner_tiled`, PRs #24567–24569, tracking issue #24574). Its pointer-plus-offset ABI maps closely onto `--entry=bare`.
- Pitch: MLIR-generated bitcode for shapes IREE does not hand-write (larger x86 and arm64 tiles, RVV, SME), with a machine-model-driven K-split. Use the three-ISA tile results as evidence, and the end-to-end numbers once Config C exists.
- Caveat for the pitch: the K-split changes the order of FP summation along K. Results are not bitwise identical to IREE, but they stay within tolerance.
- Open a discussion with maintainers (Benoit Jacob) before writing a large PR.

### E. Research angles
- **SME:** IREE has no usable f32 SME GEMM on M4 (see `plan.md`, 1.4–2.7 GFLOP/s). The SME kernel must own the M1×N1 loop, because streaming-mode entry per tile is too expensive.
- **RVV:** riscv64 uses codegen unless `+xsmtvdot` is enabled, so there is room for generated kernels (for example 16×13).
- **Strides:** skip the LHS pack and output unpack while weights stay packed.
- Add IREE as an MLIR-compiler baseline in the GAMM paper.

### Caveats
- K-split changes the FP summation order along K (see 5.D).
- Default BLIS kernels can now get a K-split too (for example, the haswell 6×8 gets 2 sets). Check that the paper numbers still reproduce, or pass `--k-split=1`.
- These are tile-level results only. In a real IREE matmul the narrow tiles cover only the M remainder. The end-to-end gain depends on the shape mix (small-M, GEMV-like, decode-style) and on pack/padding costs for new shapes.

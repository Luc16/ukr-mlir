#!/bin/bash
# Slurm wrapper for scripts/iree_tile_bench.sh on the sorgan cluster.
# Takes the whole node (--exclusive) so other jobs do not skew the numbers,
# records the CPU and frequency settings, then does RUNS independent runs.
#
#   sbatch -w sorgan-cpu2 --export=ALL,ISA=avx512,CPU=cascadelake scripts/iree_tile_bench.sbatch
#   sbatch -w sorgan-cpu6 --export=ALL,ISA=avx2,CPU=skylake       scripts/iree_tile_bench.sbatch
#
# Environment: ISA (avx2|avx512), CPU (-march/-mcpu), RUNS (default 5),
# LLVM_DIR (default ~/llvm-project/build), CLANG_BIN (dir with clang),
# OUT (default results/iree_tile_x86/<node>), BENCH_ARGS.
#SBATCH -J iree-tile
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --exclusive
#SBATCH -t 02:00:00
#SBATCH -o tmp/slurm-%x-%N-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

ISA=${ISA:?set ISA=avx2|avx512}
CPU=${CPU:-native}
RUNS=${RUNS:-5}
LLVM_DIR=${LLVM_DIR:-$HOME/llvm-project/build}
CLANG_BIN=${CLANG_BIN:-/spack/opt/spack/linux-x86_64_v3/llvm-19.1.7-eyv47zbamu4w2lz2lgyaqdd6mbwppf3i/bin}
NODE=$(hostname -s)
OUT=${OUT:-results/iree_tile_x86/$NODE}
BENCH_ARGS=${BENCH_ARGS:---k=64,256,1024 --modes=hot,warm,cold --reps=15}
export PATH="$CLANG_BIN:$PATH"

mkdir -p "$OUT"
{
    echo "node: $NODE  isa: $ISA  cpu: $CPU  date: $(date -Is)"
    echo "ukr-mlir: $(git rev-parse --short HEAD)$(git diff --quiet || echo -dirty)"
    echo "iree: $(git -C external/iree rev-parse --short HEAD 2>/dev/null || echo unknown)"
    echo "clang: $(clang --version | head -1)"
    echo "governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo n/a)"
    echo "no_turbo: $(cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null || echo n/a)"
    lscpu
} > "$OUT/machine.txt"

for i in $(seq 1 "$RUNS"); do
    echo ">> run $i/$RUNS on $NODE" >&2
    # shellcheck disable=SC2086
    scripts/iree_tile_bench.sh -l "$LLVM_DIR" -I "$ISA" -c "$CPU" \
        -b "tmp/iree_tile/build_${ISA}_$NODE" -o "$OUT/run_$i.csv" -- $BENCH_ARGS \
        2> >(tee "$OUT/run_$i.log" >&2) > /dev/null
done
echo ">> done: $OUT" >&2

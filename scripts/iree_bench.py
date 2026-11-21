"""Level-2 IREE baseline: compile linalg.matmul with several IREE CPU configs,
check results against numpy and report GFLOP/s (see plan.md, "Level 2").

Configs:
  A  data tiling + codegen (ukernels off)
  B  data tiling + IREE mmt4d ukernels
  D  Arm SME codegen (no data tiling, scalable vectorization, +sme)
  C  (ukr-mlir tile function) is not wired up yet.

Usage:
  python3 scripts/iree_bench.py --iree-bin /path/to/venv/bin -c A,B,D -s 256,512,1024
  python3 scripts/iree_bench.py --shapes-file shapes.csv -o results/iree.csv
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys

import numpy as np

TARGET = [
    "--iree-hal-target-device=local",
    "--iree-hal-local-target-device-backends=llvm-cpu",
]

CONFIGS = {
    "A": TARGET + [
        "--iree-llvmcpu-target-cpu=host",
        "--iree-opt-data-tiling",
        "--iree-llvmcpu-enable-ukernels=none",
    ],
    "B": TARGET + [
        "--iree-llvmcpu-target-cpu=host",
        "--iree-opt-data-tiling",
        "--iree-llvmcpu-enable-ukernels=mmt4d",
    ],
    # 'host' cannot be combined with explicit features, so name the CPU.
    # Without --iree-llvmcpu-enable-scalable-vectorization IREE emits NEON.
    # vscale-range=4,4 matches the M4's 512-bit SVL.
    "D": TARGET + [
        "--iree-llvmcpu-target-cpu=apple-m4",
        "--iree-llvmcpu-target-cpu-features=+sme,+sme2",
        "--iree-llvmcpu-enable-scalable-vectorization",
        "--iree-llvmcpu-vscale-range=4,4",
        "--iree-opt-data-tiling=false",
    ],
}

MLIR_TYPES = {"f32": np.float32, "f16": np.float16}


def matmul_mlir(m, n, k, dtype):
    return f"""
func.func @mm(%a: tensor<{m}x{k}x{dtype}>, %b: tensor<{k}x{n}x{dtype}>) -> tensor<{m}x{n}x{dtype}> {{
  %z = arith.constant 0.0 : {dtype}
  %e = tensor.empty() : tensor<{m}x{n}x{dtype}>
  %c = linalg.fill ins(%z : {dtype}) outs(%e : tensor<{m}x{n}x{dtype}>) -> tensor<{m}x{n}x{dtype}>
  %r = linalg.matmul ins(%a, %b : tensor<{m}x{k}x{dtype}>, tensor<{k}x{n}x{dtype}>)
                     outs(%c : tensor<{m}x{n}x{dtype}>) -> tensor<{m}x{n}x{dtype}>
  return %r : tensor<{m}x{n}x{dtype}>
}}
"""


def tool(bin_dir, name):
    path = os.path.join(bin_dir, name) if bin_dir else shutil.which(name)
    if not path or not os.path.exists(path):
        sys.exit(f"error: {name} not found (pass --iree-bin or put it on PATH)")
    return path


def run(cmd, timeout):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def device_flags(threads):
    if threads == 1:
        return ["--device=local-sync"]
    return ["--device=local-task", f"--task_topology_max_group_count={threads}"]


def parse_shapes(args):
    shapes = []
    if args.shapes_file:
        with open(args.shapes_file) as f:
            for row in csv.reader(f):
                if not row or row[0].strip().startswith("#"):
                    continue
                shapes.append(tuple(int(x) for x in row[:3]))
    for s in args.square:
        shapes.append((s, s, s))
    return shapes


def bench_one(args, cfg, m, n, k, workdir):
    tag = f"{cfg}_{m}x{n}x{k}_{args.dtype}"
    src = os.path.join(workdir, f"mm_{m}x{n}x{k}_{args.dtype}.mlir")
    vmfb = os.path.join(workdir, f"{tag}.vmfb")
    with open(src, "w") as f:
        f.write(matmul_mlir(m, n, k, args.dtype))

    r = run([tool(args.iree_bin, "iree-compile"), src, *CONFIGS[cfg], "-o", vmfb], args.timeout)
    if r.returncode != 0:
        return {"status": "compile_error", "detail": r.stderr.strip().splitlines()[-1:]}

    rng = np.random.default_rng(0)
    np_t = MLIR_TYPES[args.dtype]
    a = rng.uniform(-1, 1, (m, k)).astype(np_t)
    b = rng.uniform(-1, 1, (k, n)).astype(np_t)
    a_npy, b_npy = os.path.join(workdir, f"a_{tag}.npy"), os.path.join(workdir, f"b_{tag}.npy")
    out_npy = os.path.join(workdir, f"out_{tag}.npy")
    np.save(a_npy, a)
    np.save(b_npy, b)
    inputs = [f"--input=@{a_npy}", f"--input=@{b_npy}"]
    common = [f"--module={vmfb}", "--function=mm", *device_flags(args.threads), *inputs]

    r = run([tool(args.iree_bin, "iree-run-module"), *common, f"--output=@{out_npy}"], args.timeout)
    if r.returncode != 0:
        return {"status": "run_error", "detail": r.stderr.strip().splitlines()[-1:]}
    ref = a.astype(np.float64) @ b.astype(np.float64)
    out = np.load(out_npy).astype(np.float64)
    max_err = float(np.max(np.abs(out - ref)) / max(1.0, float(np.max(np.abs(ref)))))
    tol = 1e-4 if args.dtype == "f32" else 1e-2

    r = run([tool(args.iree_bin, "iree-benchmark-module"), *common,
             f"--benchmark_repetitions={args.reps}",
             "--benchmark_report_aggregates_only=true",
             "--benchmark_format=json"], args.timeout)
    if r.returncode != 0:
        return {"status": "bench_error", "detail": r.stderr.strip().splitlines()[-1:]}
    report = json.loads(r.stdout[r.stdout.index("{"):])
    median = next(b for b in report["benchmarks"] if b.get("aggregate_name") == "median")
    secs = median["real_time"] * {"ns": 1e-9, "us": 1e-6, "ms": 1e-3, "s": 1.0}[median["time_unit"]]

    return {
        "status": "ok" if max_err <= tol else "WRONG",
        "max_rel_err": max_err,
        "median_ms": secs * 1e3,
        "gflops": 2.0 * m * n * k / secs / 1e9,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--iree-bin", default=os.environ.get("IREE_BIN", ""),
                   help="directory with iree-compile & co (default: $IREE_BIN or PATH)")
    p.add_argument("-c", "--configs", default="A,B,D", help="comma-separated subset of A,B,D")
    p.add_argument("-s", "--square", default="", help="comma-separated square sizes, e.g. 256,512,1024")
    p.add_argument("--shapes-file", help="CSV with M,N,K per line ('#' comments allowed)")
    p.add_argument("-p", "--dtype", default="f32", choices=sorted(MLIR_TYPES))
    p.add_argument("-t", "--threads", type=int, default=1)
    p.add_argument("-r", "--reps", type=int, default=20, help="benchmark repetitions (median reported)")
    p.add_argument("-w", "--workdir", default="tmp/iree")
    p.add_argument("-o", "--output", help="CSV file for results")
    p.add_argument("--timeout", type=int, default=600, help="per-command timeout in seconds")
    args = p.parse_args()
    args.square = [int(x) for x in args.square.split(",") if x]

    shapes = parse_shapes(args)
    if not shapes:
        args.square = [256, 512, 1024]
        shapes = parse_shapes(args)
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    for c in configs:
        if c not in CONFIGS:
            sys.exit(f"error: unknown config {c!r} (available: {', '.join(CONFIGS)})")
    os.makedirs(args.workdir, exist_ok=True)

    fields = ["config", "M", "N", "K", "dtype", "threads", "status", "median_ms", "gflops", "max_rel_err"]
    rows = []
    print(f"{'cfg':<4}{'M':>6}{'N':>6}{'K':>6}  {'status':<14}{'ms':>10}{'GFLOP/s':>10}")
    for (m, n, k) in shapes:
        for cfg in configs:
            try:
                res = bench_one(args, cfg, m, n, k, args.workdir)
            except subprocess.TimeoutExpired:
                res = {"status": "timeout"}
            row = {"config": cfg, "M": m, "N": n, "K": k, "dtype": args.dtype,
                   "threads": args.threads, **res}
            rows.append(row)
            ms = f"{res['median_ms']:.3f}" if "median_ms" in res else "-"
            gf = f"{res['gflops']:.1f}" if "gflops" in res else "-"
            print(f"{cfg:<4}{m:>6}{n:>6}{k:>6}  {res['status']:<14}{ms:>10}{gf:>10}"
                  + (f"  {res['detail']}" if "detail" in res else ""), flush=True)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"results written to {args.output}")


if __name__ == "__main__":
    main()

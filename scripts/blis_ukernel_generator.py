import sys

# Options: --alpha=one|dynamic, --beta=zero|one|dynamic, --edges=true|false,
# --name=<func name> (default: bli_<prefix>_<arch>_<mr>_<nr>_ukernel),
# --k-split=<n> (accumulator sets for the K loop; default: chosen by the generator),
# --entry=bare (also emit <name>_tile(out, lhs, rhs, k) taking bare pointers, see below)
opts = {"alpha": "dynamic", "beta": "dynamic", "edges": "true", "name": "",
        "k-split": "", "entry": ""}
args = []
for arg in sys.argv[1:]:
    if arg.startswith("--"):
        key, _, val = arg[2:].partition("=")
        if key not in opts:
            print(f"Unknown option: {arg}")
            sys.exit(1)
        opts[key] = val
    else:
        args.append(arg)

if len(args) < 4:
    print("Usage: python script.py <mr> <nr> <elem_type> <arch> [layout] [out_type] "
          "[--alpha=one|dynamic] [--beta=zero|one|dynamic] [--edges=true|false] [--name=<func>] "
          "[--k-split=<n>] [--entry=bare]")
    sys.exit(1)

mr = int(args[0])
nr = int(args[1])
elem_type = args[2]
arch = args[3]
layout = args[4] if len(args) > 4 else "row"
out_type = args[5] if len(args) > 5 else elem_type

prefix = "gemm" # Default for mixed precision like i8->i32
if elem_type == "f32" and out_type == "f32":
    prefix = "sgemm"
elif elem_type == "f64" and out_type == "f64":
    prefix = "dgemm"
elif elem_type == "f16" and out_type == "f16":
    prefix = "hgemm"

func_name = opts["name"] or f"bli_{prefix}_{arch}_{mr}_{nr}_ukernel"

k_split_attr = f", ukernel_k_split = {int(opts['k-split'])} : i64" if opts["k-split"] else ""

if opts["entry"] not in ("", "bare"):
    print(f"Unknown entry kind: {opts['entry']}")
    sys.exit(1)
if opts["entry"] == "bare" and (opts["alpha"] != "one" or opts["beta"] == "dynamic"):
    print("--entry=bare needs --alpha=one and --beta=zero|one (the entry has no alpha/beta)")
    sys.exit(1)

is_int = elem_type.startswith('i')

op_ext = "arith.extsi" if is_int else "arith.extf"
op_mul = "arith.muli"   if is_int else "arith.mulf"
op_add = "arith.addi"   if is_int else "arith.addf"

# Handle optional MLIR widening for the dummy body
widen_op = ""
mul_type = elem_type
mul_a, mul_b = "%a", "%b"

if elem_type != out_type:
    widen_op = f"""
        %a_ext = {op_ext} %a : {elem_type} to {out_type}
        %b_ext = {op_ext} %b : {elem_type} to {out_type}
"""
    mul_a, mul_b = "%a_ext", "%b_ext"
    mul_type = out_type

# Bare-pointer entry for IREE-style callers: out tile M0 x N0 row-major, LHS
# panel [K][M0], RHS panel [K][N0]. It wraps the pointers into memref
# descriptors and calls the kernel; with -inline before lowering, the kernel is
# inlined and the descriptors fold away, leaving plain pointer arithmetic.
bare_entry = ""
if opts["entry"] == "bare":
    desc2 = "!llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>"
    desc1 = "!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>"

    def panel(name, ptr, width):
        return f"""
    %{name}0 = llvm.mlir.undef : {desc2}
    %{name}1 = llvm.insertvalue %{ptr}, %{name}0[0] : {desc2}
    %{name}2 = llvm.insertvalue %{ptr}, %{name}1[1] : {desc2}
    %{name}3 = llvm.insertvalue %z, %{name}2[2] : {desc2}
    %{name}4 = llvm.insertvalue %k, %{name}3[3, 0] : {desc2}
    %{name}5 = llvm.insertvalue %w{width}, %{name}4[3, 1] : {desc2}
    %{name}6 = llvm.insertvalue %w{width}, %{name}5[4, 0] : {desc2}
    %{name}7 = llvm.insertvalue %one, %{name}6[4, 1] : {desc2}
    %{name} = builtin.unrealized_conversion_cast %{name}7 : {desc2} to memref<?x{width}x{elem_type}>"""

    widths = sorted({mr, nr})
    bare_entry = f"""
  func.func @{func_name}_tile(%out: !llvm.ptr {{llvm.noalias}}, %lhs: !llvm.ptr {{llvm.noalias}},
                              %rhs: !llvm.ptr {{llvm.noalias}}, %k: i64) {{
    %z = llvm.mlir.constant(0 : i64) : i64
    %one = llvm.mlir.constant(1 : i64) : i64
{chr(10).join(f"    %w{w} = llvm.mlir.constant({w} : i64) : i64" for w in widths)}
    %wc = llvm.mlir.constant({mr * nr} : i64) : i64
{panel("a", "lhs", mr)}
{panel("b", "rhs", nr)}
    %c0 = llvm.mlir.undef : {desc1}
    %c1 = llvm.insertvalue %out, %c0[0] : {desc1}
    %c2 = llvm.insertvalue %out, %c1[1] : {desc1}
    %c3 = llvm.insertvalue %z, %c2[2] : {desc1}
    %c4 = llvm.insertvalue %wc, %c3[3, 0] : {desc1}
    %c5 = llvm.insertvalue %one, %c4[4, 0] : {desc1}
    %c = builtin.unrealized_conversion_cast %c5 : {desc1} to memref<?x{out_type}>
    %alpha = memref.alloca() : memref<{out_type}>
    %beta = memref.alloca() : memref<{out_type}>
    %null = llvm.mlir.zero : !llvm.ptr
    %ki = arith.index_cast %k : i64 to index
    %m = arith.constant {mr} : index
    %n = arith.constant {nr} : index
    %cs = arith.constant 1 : index
    call @{func_name}(%m, %n, %ki, %alpha, %a, %b, %beta, %c, %n, %cs, %null, %null)
      : (index, index, index, memref<{out_type}>, memref<?x{mr}x{elem_type}>, memref<?x{nr}x{elem_type}>,
         memref<{out_type}>, memref<?x{out_type}>, index, index, !llvm.ptr, !llvm.ptr) -> ()
    return
  }}
"""

print(f"""
#map_a = affine_map<(d0, d1, d2) -> (d2, d0)>
#map_b = affine_map<(d0, d1, d2) -> (d2, d1)>
#map_c = affine_map<(d0, d1, d2) -> (d0, d1)>

module {{
  func.func @{func_name}(
    %m: index, %n: index, %k: index,
    %alpha_mem: memref<{out_type}>,
    %a_mem: memref<?x{mr}x{elem_type}>,
    %b_mem: memref<?x{nr}x{elem_type}>,
    %beta_mem: memref<{out_type}>,
    %c_mem: memref<?x{out_type}>,
    %rs_c: index, %cs_c: index,
    %data: !llvm.ptr, %cntx: !llvm.ptr
  ) attributes {{llvm.emit_c_interface, ukernel_layout = "{layout}", ukernel_alpha = "{opts['alpha']}", ukernel_beta = "{opts['beta']}", ukernel_edges = {opts['edges']}{k_split_attr}}} {{

    %alpha = memref.load %alpha_mem[] : memref<{out_type}>
    %beta  = memref.load %beta_mem[]  : memref<{out_type}>

    %c_2d = memref.reinterpret_cast %c_mem to offset: [0], sizes: [{mr}, {nr}], strides: [%rs_c, %cs_c] : memref<?x{out_type}> to memref<{mr}x{nr}x{out_type}, strided<[?, ?], offset: ?>>

    linalg.generic {{
        indexing_maps = [#map_a, #map_b, #map_c],
        iterator_types = ["parallel", "parallel", "reduction"]
      }}
      ins(%a_mem, %b_mem : memref<?x{mr}x{elem_type}>, memref<?x{nr}x{elem_type}>)
      outs(%c_2d : memref<{mr}x{nr}x{out_type}, strided<[?, ?], offset: ?>>)
    {{
    ^bb0(%a: {elem_type}, %b: {elem_type}, %c: {out_type}):{widen_op}
      %mul = {op_mul} {mul_a}, {mul_b} : {mul_type}
      %add = {op_add} %c, %mul : {out_type}
      linalg.yield %add : {out_type}
    }}

    return
  }}
{bare_entry}}}
""")


import sys

if len(sys.argv) < 5:
    print("Usage: python script.py <mr> <nr> <elem_type> <arch> [layout] [out_type]")
    sys.exit(1)

mr = int(sys.argv[1])
nr = int(sys.argv[2])
elem_type = sys.argv[3]
arch = sys.argv[4]
layout = sys.argv[5] if len(sys.argv) > 5 else "row"
out_type = sys.argv[6] if len(sys.argv) > 6 else elem_type

prefix = "gemm" # Default for mixed precision like i8->i32
if elem_type == "f32" and out_type == "f32":
    prefix = "sgemm"
elif elem_type == "f64" and out_type == "f64":
    prefix = "dgemm"
elif elem_type == "f16" and out_type == "f16":
    prefix = "hgemm"

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

print(f"""
#map_a = affine_map<(d0, d1, d2) -> (d2, d0)>
#map_b = affine_map<(d0, d1, d2) -> (d2, d1)>
#map_c = affine_map<(d0, d1, d2) -> (d0, d1)>

module {{
  func.func @bli_{prefix}_{arch}_{mr}_{nr}_ukernel(
    %m: index, %n: index, %k: index,
    %alpha_mem: memref<{out_type}>,
    %a_mem: memref<?x{mr}x{elem_type}>,
    %b_mem: memref<?x{nr}x{elem_type}>,
    %beta_mem: memref<{out_type}>,
    %c_mem: memref<?x{out_type}>,
    %rs_c: index, %cs_c: index,
    %data: !llvm.ptr, %cntx: !llvm.ptr
  ) attributes {{llvm.emit_c_interface, ukernel_layout = "{layout}"}} {{

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
}}
""")


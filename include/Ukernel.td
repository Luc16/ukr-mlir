//===-- Ukernel.td - Transform dialect Ukernel -----------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines Transform dialect extension operations used in the Ukernel.
//
//===----------------------------------------------------------------------===//

#ifndef UKERNEL
#define UKERNEL

include "mlir/Dialect/Linalg/TransformOps/LinalgTransformEnums.td"
include "mlir/Dialect/Transform/IR/TransformAttrs.td"
include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.td"
include "mlir/Dialect/Transform/IR/TransformTypes.td"
include "mlir/Dialect/SCF/IR/DeviceMappingInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/RegionKindInterface.td"

// Define the sgemm operation. By convention, prefix its name with the name of the 
// extension, "structured.". The full op name will be further prefixed with "transform.".
def UkernelOp : Op<Transform_Dialect, "structured.ukernel",
       [DeclareOpInterfaceMethods<TransformOpInterface>,
        DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {

  let summary = "Generate SGemm micro-kernels and associated loops.";
  let description = [{
    This operation generates optimized SGemm micro-kernels along with their associated loops
  }];

  // The argument include the handle to the payload operation, with optional mK_info, 
  // & arch_info configurations:
  //   mK_info = { nrows, ncols }
  //   arch_info = { l1size, l2size, l3size, cache_line }

  // The handle must implement TransformHandleTypeInterface.   
  let arguments = (ins
    TransformHandleTypeInterface:$target,
    OptionalAttr<ArrayAttr>:$mK_info,
    OptionalAttr<ArrayAttr>:$arch_info
  );

  let results = (outs
    TransformHandleTypeInterface:$microkernel,     // container of micro-Kernels
    Variadic<TransformHandleTypeInterface>:$loops   // container of associated loops
  );

  let assemblyFormat = "$target attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;

}


#endif // UKERNEL

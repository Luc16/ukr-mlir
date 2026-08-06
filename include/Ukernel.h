//===-- Ukernel.h - Transform dialect Extension Ukernel -------------*- c++ -*-===//
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

#ifndef UKERNEL_H
#define UKERNEL_H

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgMatchOps.h"
#include "mlir/Dialect/Linalg/TransformOps/Syntax.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include <mlir/Dialect/Transform/IR/TransformAttrs.h>
#include <mlir/Dialect/Utils/StructuredOpsUtils.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/RegionKindInterface.h>

namespace mlir {
class CallOpInterface;
class RewriterBase;

namespace linalg {
class GenericOp;
class LinalgOp;
} // namespace linalg

namespace transform {
class AnyOpType;
class AnyValueType;
class OperationType;
class TransformHandleTypeInterface;
} // namespace transform
} // namespace mlir

namespace mlir {
class DialectRegistry;
} // namespace mlir

// To debug info, use: transform-opt your_parameters -debug-only=Ukernel
#define DEBUG_TYPE "sgemm"
#define DBGS() (llvm::dbgs() << "\n[" DEBUG_TYPE "] ")

#define GET_OP_CLASSES
#include "Ukernel.h.inc"

// Registers our Transform dialect extension.
void registerUkernel(::mlir::DialectRegistry &registry);

//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//

namespace {

using namespace mlir;

/// Create a named location from a name and another location.
static inline Location CreateNameLoc(const Twine &name, Location loc) {
#ifndef NDEBUG
  return NameLoc::get(StringAttr::get(loc.getContext(), name), loc);
#else
  return loc;
#endif
}

/// Set named location to an existing value
static inline Value SetNameLoc(Value value, const Twine &name) {
#ifndef NDEBUG
  value.setLoc(CreateNameLoc(name, value.getLoc()));
#endif
  return value;
}

/// Set named location to an existing op with a single result
static inline Operation *SetNameLoc(Operation *op, const Twine &name) {
#ifndef NDEBUG
  assert(op->getNumResults() == 1 && "Operation must have a single result");
  SetNameLoc(op->getResult(0), name);
#endif
  return op;
}

} // namespace

//===----------------------------------------------------------------------===//
// Inline functions
//===----------------------------------------------------------------------===//

static inline mlir::Location operator<<(mlir::Location loc,
                                        const mlir::Twine &name) {
  return CreateNameLoc(name, loc);
}

static inline mlir::Value operator<<(mlir::Value value,
                                     const mlir::Twine &name) {
  return SetNameLoc(value, name);
}

static inline mlir::Operation *operator<<(mlir::Operation *op,
                                          const mlir::Twine &name) {
  return SetNameLoc(op, name);
}

//===----------------------------------------------------------------------===//
// Declarations
//===----------------------------------------------------------------------===//

typedef struct {
  uint32_t l1_size;
  uint32_t l2_size;
  uint32_t l3_size;
  uint32_t vec_len;
  uint32_t num_vecs;
  uint32_t is_risc;
  uint32_t n_fmacs;
  uint32_t lat_fmacs;
  uint32_t has_op_ext;
  uint32_t n_rows;
  uint32_t row_bits;
  uint32_t is_scalable;
  uint32_t has_mask_regs;
} ArchInfo;

struct BLISParams {
  Value alpha_tensor;
  Value beta_tensor;
  Value mVal;
  Value nVal;
  Value kVal;
  Value rsc;
  Value csc;
};

typedef struct {
  uint8_t mr;
  uint8_t nr;
  uint8_t unroll;
  uint32_t kc;
} UKernelInfo;


#endif // UKERNEL_H

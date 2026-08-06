//===-- Ukernel.cpp - transform dialect Extension Ukernel ------------------===//
//----------------------------------------------------------------------------//

#include "Ukernel.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h" // Added MemRef Dialect
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/CommandLine.h"

#include "mlir/IR/DialectImplementation.h"
#include "mlir/Interfaces/CallInterfaces.h"

using namespace mlir;
using namespace mlir::affine;
using namespace mlir::linalg;
using namespace mlir::transform;

#define GET_OP_CLASSES
#include "Ukernel.cpp.inc"


static llvm::cl::opt<bool> clIsRVV(
    "is-rvv",
    llvm::cl::desc("Indicates whether the target architecture is RISC-V with vector extension (RVV). This flag can be used to adjust ukernel generation for RVV-specific widening operations."),
    llvm::cl::init(false) 
);


static llvm::cl::opt<uint32_t> clVecLen(
    "vec-len",
    llvm::cl::desc("Specifies the vector length in bits for the target architecture. This is used to calculate how many elements can fit in a vector register based on the data type's bit width."),
    llvm::cl::init(128) 
);



class Ukernel : public transform::TransformDialectExtension<Ukernel> {
  public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(Ukernel)
      using Base::Base;
    void init();
};

void Ukernel::init() {
  declareDependentDialect<linalg::LinalgDialect>();
  declareGeneratedDialect<affine::AffineDialect>();
  declareGeneratedDialect<arith::ArithDialect>();
  declareGeneratedDialect<index::IndexDialect>();
  declareGeneratedDialect<scf::SCFDialect>();
  declareGeneratedDialect<tensor::TensorDialect>();
  declareGeneratedDialect<memref::MemRefDialect>(); // Registered MemRef
  declareGeneratedDialect<vector::VectorDialect>();
  declareGeneratedDialect<LLVM::LLVMDialect>();

  registerTransformOps<
#define GET_OP_LIST
#include "Ukernel.cpp.inc"
    >();
}

UKernelInfo findUkernelDimensions(uint8_t aBitWidth, ArchInfo archInfo) {
  double elemPerVec = (double) archInfo.vec_len / (double) aBitWidth;
  uint32_t sRegs = (archInfo.is_risc) ? 3 : 2;
  double availRegs = (double) (archInfo.num_vecs - sRegs);
  double usedElems = elemPerVec * availRegs;

  double ncols_d = std::ceil((-1.0 + std::sqrt(1.0 + 4.0 * usedElems)) / (2.0 * elemPerVec)) * elemPerVec;
  uint8_t nr = static_cast<uint8_t>(ncols_d);

  double nrows_d = std::floor(usedElems / ncols_d) - 1;
  uint8_t mr = static_cast<uint8_t>(nrows_d);

  return {mr, nr, 4, 0};
}

void generateVectorUkernel(OpBuilder &rewriter, Location loc, Value A, Value B, Value C, Type type, Type outType, uint32_t vecLen, UKernelInfo ukInfo, ArchInfo archInfo, BLISParams blisParams, bool isRowMajor = true, uint32_t wideningFactor = 1, bool isEdge = false) {

  // if (isRowMajor && ukInfo.nr % vecLen != 0) {
  //   llvm::errs() << "Error: row-major layout requires nr to be a multiple of vector length.\n";
  //   return;
  // } else if (!isRowMajor && ukInfo.mr % vecLen != 0) {
  //   llvm::errs() << "Error: column-major layout requires mr to be a multiple of vector length.\n";
  //   return;
  // }

  Value allignedMatrix, otherMatrix, strideC, vecDim, scalarDim;
  uint32_t vecsPerLoop, outerDim;
  if (isRowMajor) {
    allignedMatrix = B;
    otherMatrix = A;
    strideC = blisParams.rsc;
    vecsPerLoop = ukInfo.nr / vecLen;
    outerDim = ukInfo.mr;
    scalarDim = blisParams.mVal;
    vecDim = blisParams.nVal;
  } else {
    allignedMatrix = A;
    otherMatrix = B;
    strideC = blisParams.csc;
    vecsPerLoop = ukInfo.mr / vecLen;
    outerDim = ukInfo.nr;
    scalarDim = blisParams.nVal;
    vecDim = blisParams.mVal;
  }

  Value c0 = arith::ConstantOp::create(rewriter, loc, rewriter.getIndexAttr(0));
  Value step = arith::ConstantOp::create(rewriter, loc, rewriter.getIndexAttr(ukInfo.unroll));

  Value zeroVal;
  if (isa<IntegerType>(type)) {
    zeroVal = arith::ConstantOp::create(rewriter, loc, type, rewriter.getIntegerAttr(type, 0));
  } else {
    zeroVal = arith::ConstantOp::create(rewriter, loc, type, rewriter.getFloatAttr(type, 0.0));
  }

  Value outZeroVal;
  if (isa<IntegerType>(outType)) {
    outZeroVal = arith::ConstantOp::create(rewriter, loc, outType, rewriter.getIntegerAttr(outType, 0));
  } else {
    outZeroVal = arith::ConstantOp::create(rewriter, loc, outType, rewriter.getFloatAttr(outType, 0.0));
  }

  Value alpha = memref::LoadOp::create(rewriter, loc, blisParams.alpha_tensor, ValueRange{});
  Value beta = memref::LoadOp::create(rewriter, loc, blisParams.beta_tensor, ValueRange{});

  SmallVector<Value> masks;
  if (isEdge) {
    Value vlConst = arith::ConstantOp::create(rewriter, loc, rewriter.getIndexAttr(vecLen));
    for (int i = 0; i < vecsPerLoop; i++) {
      Value vIdx = arith::ConstantOp::create(rewriter, loc, rewriter.getIndexAttr(i*vecLen));

      Value rem = arith::SubIOp::create(rewriter, loc, vecDim, vIdx);
      Value positive = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::sgt, rem, c0);
      Value remSafe = arith::SelectOp::create(rewriter, loc, positive, rem, c0);

      Value exceed = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::sgt, remSafe, vlConst);
      Value finalRem = arith::SelectOp::create(rewriter, loc, exceed, vlConst, remSafe);


      VectorType maskVecTy = VectorType::get({vecLen}, rewriter.getI1Type());
      masks.push_back(
        vector::CreateMaskOp::create(rewriter, loc, maskVecTy, ValueRange{finalRem})
      );
    }
  }

  // for (int i = 0; i < outerDim; ++i) {
  //   Value iIdx = arith::ConstantOp::create(rewriter, loc, rewriter.getIndexAttr(i));
  //   Value i_rsc = arith::MulIOp::create(rewriter, loc, iIdx, strideC);
  //   memref::PrefetchOp::create(rewriter, 
  //       loc, C, ValueRange{i_rsc}, 
  //       /*isWrite=*/true,       // We intend to write to this memory
  //       /*localityHint=*/3,     // Extreme locality (Keep in L1 cache)
  //       /*isDataCache=*/true);  // It is data, not instructions
  // }
  //


  Value kFull = blisParams.kVal;
  // if (wideningFactor > 1 && !clIsRVV) {
  //   Value wideVal = arith::ConstantOp::create(rewriter, loc, rewriter.getIndexAttr(wideningFactor));
  //   kFull = arith::DivSIOp::create(rewriter, loc, kFull, wideVal);
  // }
  Value unrollVal = arith::ConstantOp::create(rewriter, loc, rewriter.getIndexAttr(ukInfo.unroll));
  Value kDim = arith::DivSIOp::create(rewriter, loc, kFull, unrollVal);
  kDim = arith::MulIOp::create(rewriter, loc, kDim, unrollVal);

  uint32_t inVecLen = vecLen;

  // if (wideningFactor > 1 && !clIsRVV) {
  //   inVecLen = vecLen * wideningFactor;
  // }


  VectorType vecTy = VectorType::get({inVecLen}, type);
  VectorType outVecTy = VectorType::get({vecLen}, outType);

  SmallVector<Value> accVars;
  SmallVector<bool> inBoundsRow(1, true);
  ArrayAttr inBoundsAttr = rewriter.getBoolArrayAttr(inBoundsRow);

  // 1D Map for vector reads
  AffineMap map1D = AffineMap::getMinorIdentityMap(1, 1, rewriter.getContext());
  AffineMapAttr mapAttr1D = AffineMapAttr::get(map1D);
  SmallVector<bool> inBoundsFlat(1, true);
  ArrayAttr inBoundsAttrFlat = rewriter.getBoolArrayAttr(inBoundsFlat);
  AffineMap map2D = AffineMap::getMinorIdentityMap(2, 1, rewriter.getContext());
  AffineMapAttr mapAttr2D = AffineMapAttr::get(map2D);

  for (int i = 0; i < outerDim; ++i) {
    for (int j = 0; j < vecsPerLoop; ++j) {
      Value acc = vector::BroadcastOp::create(rewriter, loc, outVecTy, outZeroVal);      
      accVars.push_back(acc);
    }
  }

  auto createKLoopBody = [&](OpBuilder &builder, Location loc, Value k, SmallVectorImpl<Value> &currentAccs) {
    Value lookahead = arith::ConstantOp::create(rewriter, loc, rewriter.getIndexAttr(32));
    Value prefetchK = arith::AddIOp::create(rewriter, loc, k, lookahead);
    Value zeroIdx = arith::ConstantOp::create(rewriter, loc, rewriter.getIndexAttr(0));

    // memref::PrefetchOp::create(rewriter, 
    //     loc, otherMatrix, ValueRange{prefetchK, zeroIdx},
    //     /*isWrite=*/false,      // Read-only memory
    //     /*localityHint=*/3,     // Cache in L1
    //     /*isDataCache=*/true);

    SmallVector<Value> bRow;
    for (int v = 0; v < vecsPerLoop; ++v) {
      int bStep = v * vecLen;
      // if (wideningFactor > 1 && !clIsRVV) {
      //   bStep = v * inVecLen;
      // }
      Value vIdx = arith::ConstantOp::create(rewriter, loc, rewriter.getIndexAttr(bStep));

      // Read directly from flat B
      // Value bVec = (isEdge) ? 
      //   vector::TransferReadOp::create(rewriter, loc, vecTy, allignedMatrix, ValueRange{k, vIdx}, map2D, zeroVal, masks[v], inBoundsAttr) :
      //   vector::TransferReadOp::create(rewriter, loc, vecTy, allignedMatrix, ValueRange{k, vIdx}, zeroVal, inBoundsRow);

      Value bVec = vector::TransferReadOp::create(rewriter, loc, vecTy, allignedMatrix, ValueRange{k, vIdx}, zeroVal, inBoundsRow);
      bRow.push_back(bVec);
    }

    for (int m = 0; m < outerDim; ++m) {
      Value mIdx = arith::ConstantOp::create(rewriter, loc, rewriter.getIndexAttr(m));

      // Load directly from flat A
      Value aScalar = memref::LoadOp::create(rewriter, loc, otherMatrix, ValueRange{k, mIdx});
      Value aBcast = vector::BroadcastOp::create(rewriter, loc, vecTy, aScalar);

      for (int v = 0; v < vecsPerLoop; ++v) {
        int accIdx = m * vecsPerLoop + v;
        if (wideningFactor > 1) {
          Value aBcastWidened;
          Value bWidened;
          if (isa<IntegerType>(type)) {
            aBcastWidened = arith::ExtSIOp::create(rewriter, loc, outVecTy, aBcast);
            bWidened = arith::ExtSIOp::create(rewriter, loc, outVecTy, bRow[v]);
            Value tmp = arith::MulIOp::create(rewriter, loc, aBcastWidened, bWidened);
            currentAccs[accIdx] = arith::AddIOp::create(rewriter, loc, tmp, currentAccs[accIdx]);
          } else {
            aBcastWidened = arith::ExtFOp::create(rewriter, loc, outVecTy, aBcast);
            bWidened = arith::ExtFOp::create(rewriter, loc, outVecTy, bRow[v]);
            currentAccs[accIdx] = vector::FMAOp::create(rewriter, loc, aBcastWidened, bWidened, currentAccs[accIdx]);
          }
        } else {
          if (isa<IntegerType>(type)) {
            Value tmp = arith::MulIOp::create(rewriter, loc, aBcast, bRow[v]);
            currentAccs[accIdx] = arith::AddIOp::create(rewriter, loc, tmp, currentAccs[accIdx]);
          } else {
            currentAccs[accIdx] = vector::FMAOp::create(rewriter, loc, aBcast, bRow[v], currentAccs[accIdx]);
          }
        }
      }
    }
  };

  SmallVector<Value> loopOperands = accVars;
  auto loop = scf::ForOp::create(rewriter, loc, c0, kDim, step, loopOperands);
  {
    OpBuilder::InsertionGuard loopGuard(rewriter);
    rewriter.setInsertionPointToStart(loop.getBody());

    Value k_base = loop.getInductionVar();
    SmallVector<Value> currentAccs;
    for (int i = 0; i < accVars.size(); ++i) {
      currentAccs.push_back(loop.getRegionIterArgs()[i]);
    }

    for (int u = 0; u < ukInfo.unroll; ++u) {
      Value uIdx = arith::ConstantOp::create(rewriter, loc, rewriter.getIndexAttr(u));
      Value k = arith::AddIOp::create(rewriter, loc, k_base, uIdx);
      createKLoopBody(rewriter, loc, k, currentAccs);
    }

    SmallVector<Value> yieldOperands = currentAccs;
    scf::YieldOp::create(rewriter, loc, yieldOperands);
  }

  scf::ForOp finalLoop = loop;
  SmallVector<Value> finalAccs;
  for (int i = 0; i < accVars.size(); ++i) finalAccs.push_back(finalLoop.getResults()[i]);

  Value step1 = arith::ConstantOp::create(rewriter, loc, rewriter.getIndexAttr(1));
  SmallVector<Value> mainLoopResults;
  for (Value res : finalLoop.getResults()) mainLoopResults.push_back(res);

  auto remLoop = scf::ForOp::create(rewriter, loc, kDim, kFull, step1, mainLoopResults);
  {
    OpBuilder::InsertionGuard remGuard(rewriter);
    rewriter.setInsertionPointToStart(remLoop.getBody());
    Value k_rem = remLoop.getInductionVar();

    SmallVector<Value> remAccs;
    for (int i = 0; i < accVars.size(); ++i) remAccs.push_back(remLoop.getRegionIterArgs()[i]);

    createKLoopBody(rewriter, loc, k_rem, remAccs);
    SmallVector<Value> remYieldOperands = remAccs;
    scf::YieldOp::create(rewriter, loc, remYieldOperands);
  }

  finalAccs.clear();
  for (int i = 0; i < accVars.size(); ++i) finalAccs.push_back(remLoop.getResults()[i]);

  Value alphaBCast = vector::BroadcastOp::create(rewriter, loc, outVecTy, alpha);
  Value betaBCast = vector::BroadcastOp::create(rewriter, loc, outVecTy, beta);
  Value zeroBCast = vector::BroadcastOp::create(rewriter, loc, outVecTy, outZeroVal);

  if (!isEdge) {
    for (int i = 0; i < outerDim; ++i) {
      Value iIdx = arith::ConstantOp::create(rewriter, loc, rewriter.getIndexAttr(i));
      Value i_rsc = arith::MulIOp::create(rewriter, loc, iIdx, strideC);

      for (int j = 0; j < vecsPerLoop; ++j) {
        Value jIdx = arith::ConstantOp::create(rewriter, loc, rewriter.getIndexAttr(j*vecLen));
        Value cOffset = arith::AddIOp::create(rewriter, loc, i_rsc, jIdx);

        Value acc = vector::TransferReadOp::create(rewriter, loc, outVecTy, C, ValueRange{cOffset}, outZeroVal, inBoundsFlat);

        Value scaledAcc;
        Value scaledRes;
        if (isa<IntegerType>(outType)) {
          scaledAcc = arith::MulIOp::create(rewriter, loc, betaBCast, acc);
          Value scaledResTmp = arith::MulIOp::create(rewriter, loc, alphaBCast, finalAccs[i*vecsPerLoop + j]);
          scaledRes = arith::AddIOp::create(rewriter, loc, scaledAcc, scaledResTmp);
        } else {
          scaledAcc = vector::FMAOp::create(rewriter, loc, betaBCast, acc, zeroBCast);
          scaledRes = vector::FMAOp::create(rewriter, loc, alphaBCast, finalAccs[i*vecsPerLoop + j], scaledAcc);
        }

          vector::TransferWriteOp::create(rewriter, loc, scaledRes, C, ValueRange{cOffset}, inBoundsFlat);
      }
    }
  } else {
    for (int j = 0; j < vecsPerLoop; ++j) {
      Value jIdx = arith::ConstantOp::create(rewriter, loc, rewriter.getIndexAttr(j*vecLen));
      for (int i = 0; i < outerDim; ++i) {
        Value iIdx = arith::ConstantOp::create(rewriter, loc, rewriter.getIndexAttr(i));

        Value cmpI = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::sgt, scalarDim, iIdx);

        scf::IfOp calcDim = scf::IfOp::create(rewriter, loc, cmpI);
        {
          OpBuilder::InsertionGuard thenGuard(rewriter);
          rewriter.setInsertionPointToStart(calcDim.thenBlock());

          Value i_rsc = arith::MulIOp::create(rewriter, loc, iIdx, strideC);

          Value cOffset = arith::AddIOp::create(rewriter, loc, i_rsc, jIdx);

          Value acc = vector::TransferReadOp::create(rewriter, loc, outVecTy, C, ValueRange{cOffset}, mapAttr1D, outZeroVal, masks[j], inBoundsAttrFlat);

          Value scaledAcc;
          Value scaledRes;
          if (isa<IntegerType>(outType)) {
            scaledAcc = arith::MulIOp::create(rewriter, loc, betaBCast, acc);
            Value scaledResTmp = arith::MulIOp::create(rewriter, loc, alphaBCast, finalAccs[i*vecsPerLoop + j]);
            scaledRes = arith::AddIOp::create(rewriter, loc, scaledAcc, scaledResTmp);
          } else {
            scaledAcc = vector::FMAOp::create(rewriter, loc, betaBCast, acc, zeroBCast);
            scaledRes = vector::FMAOp::create(rewriter, loc, alphaBCast, finalAccs[i*vecsPerLoop + j], scaledAcc);
          }

          vector::TransferWriteOp::create(rewriter, loc, scaledRes, C, ValueRange{cOffset}, mapAttr1D, masks[j], inBoundsAttrFlat);
        }
      }
    }
  }

}

LogicalResult generateBLISUkernel(func::FuncOp blisFunc, OpBuilder &rewriter, ArchInfo archInfo) {
  if (blisFunc.getNumArguments() < 12) return failure();
  Location loc = blisFunc.getLoc();

  blisFunc.eraseBody();
  Block *body = blisFunc.addEntryBlock();
  rewriter.setInsertionPointToStart(body);

  Value mVal = blisFunc.getArgument(0);
  Value nVal = blisFunc.getArgument(1);
  Value kVal = blisFunc.getArgument(2);
  Value alpha = blisFunc.getArgument(3);
  Value A = blisFunc.getArgument(4);
  Value B = blisFunc.getArgument(5);
  Value beta = blisFunc.getArgument(6);
  Value C_init = blisFunc.getArgument(7);
  Value rsc = blisFunc.getArgument(8); 
  Value csc = blisFunc.getArgument(9); 

  auto shapedTypeA = dyn_cast<ShapedType>(A.getType());
  auto shapedTypeB = dyn_cast<ShapedType>(B.getType());
  Type aType = shapedTypeA.getElementType();
  uint8_t aBitWidth = aType.getIntOrFloatBitWidth();


  auto shapedTypeC = dyn_cast<ShapedType>(C_init.getType());
  Type cType = shapedTypeC.getElementType();
  uint8_t cBitWidth = cType.getIntOrFloatBitWidth();

  UKernelInfo ukBest = findUkernelDimensions(aBitWidth, archInfo);
  ukBest.unroll = 4;
  ukBest.mr = shapedTypeA.getDimSize(1);
  ukBest.nr = shapedTypeB.getDimSize(1);

  BLISParams blisParams = {alpha, beta, mVal, nVal, kVal, rsc, csc};

  bool isRowMajor = true;
  if (auto layoutAttr = blisFunc->getAttrOfType<StringAttr>("ukernel_layout")) {
    if (layoutAttr.getValue() == "col") isRowMajor = false;
  }


  uint32_t wideningFactor = cBitWidth / aBitWidth;
  uint32_t vecLen = (uint32_t) (archInfo.vec_len / aBitWidth);
  // if (wideningFactor > 1 && !clIsRVV) {
  //   vecLen = (uint32_t) (archInfo.vec_len / cBitWidth);
  // } else {

  if (isRowMajor && archInfo.vec_len / aBitWidth > ukBest.nr) {
    emitError(loc) << "Error: row-major layout requires nr to be larger than or equal to vector length in elements. Given aBitWidth=" << aBitWidth << " and vec_len=" << archInfo.vec_len << ", the minimum nr is " << (archInfo.vec_len / aBitWidth) << ".";
    return failure();
  } else if (!isRowMajor && archInfo.vec_len / aBitWidth > ukBest.mr) {
    emitError(loc) << "Error: column-major layout requires mr to be larger than or equal to vector length in elements. Given aBitWidth=" << aBitWidth << " and vec_len=" << archInfo.vec_len << ", the minimum mr is " << (archInfo.vec_len / aBitWidth) << ".";
    return failure();
  }
  // }


  Value mr = arith::ConstantOp::create(rewriter, loc, rewriter.getIndexAttr(ukBest.mr)); 
  Value nr = arith::ConstantOp::create(rewriter, loc, rewriter.getIndexAttr(ukBest.nr));
  Value compMr = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::eq, mr, mVal);
  Value compNr = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::eq, nr, nVal);
  Value comp = arith::AndIOp::create(rewriter, loc, compMr, compNr);

  auto fastPath = scf::IfOp::create(rewriter, loc, comp, /*withElseRegion=*/true);
  {
    OpBuilder::InsertionGuard fastGuard(rewriter);
    rewriter.setInsertionPointToStart(fastPath.thenBlock());
    generateVectorUkernel(rewriter, loc, A, B, C_init, aType, cType, vecLen, ukBest, archInfo, blisParams, isRowMajor, wideningFactor, false);
  }
  // Else Block
  {
    OpBuilder::InsertionGuard slowGuard(rewriter);
    rewriter.setInsertionPointToStart(fastPath.elseBlock());
    generateVectorUkernel(rewriter, loc, A, B, C_init, aType, cType, vecLen, ukBest, archInfo, blisParams, isRowMajor, wideningFactor, true);
  }

  func::ReturnOp::create(rewriter, loc);
  return success();
}

DiagnosedSilenceableFailure
transform::UkernelOp::apply(transform::TransformRewriter &rewriter,
    transform::TransformResults &results,
    transform::TransformState &state) {

  auto payloadOps = state.getPayloadOps(getTarget());
  SmallVector<Operation *> resultOps;

  ArchInfo archInfo = {
    .l1_size = 192 * 1024,
    .vec_len = clVecLen,
    .num_vecs = 16,
    .is_risc = 1,
    .n_fmacs = 2,
    .lat_fmacs = 5,
  };

  for (Operation *op : payloadOps) {
    auto parentFunc = op->getParentOfType<func::FuncOp>();
    if (failed(generateBLISUkernel(parentFunc, rewriter, archInfo)))
      return emitSilenceableError() << "Failed to generate BLIS ukernel.";
    resultOps.push_back(parentFunc);
  }

  results.set(cast<OpResult>(getMicrokernel()), resultOps);
  for (OpResult loopResult : getLoops()) results.set(loopResult, {});

  return DiagnosedSilenceableFailure::success();
}

void transform::UkernelOp::getEffects(SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  consumesHandle(getTargetMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
  modifiesPayload(effects);
}

LogicalResult
transform::UkernelOp::verify() {
  return success();
}

void registerUkernel(mlir::DialectRegistry &registry) {
  registry.addExtensions<Ukernel>();
}

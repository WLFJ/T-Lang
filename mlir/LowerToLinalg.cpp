//====- LowerToAffineLoops.cpp - Partial lowering from Toy to Affine+Std --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a partial lowering of Toy operations to a combination of
// affine loops, memref operations and standard operations. This lowering
// expects that all calls have been inlined, and all shapes have been resolved.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/BufferizationToMemRef/BufferizationToMemRef.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/PDL/IR/PDLOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "tc/Dialect.h"
#include "tc/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Error.h"
#include <iostream>

using namespace mlir;

//===----------------------------------------------------------------------===//
// ToyToLinalg RewritePatterns
//===----------------------------------------------------------------------===//

/// Convert the given TensorType into the corresponding MemRefType.
static MemRefType convertTensorToMemRef(TensorType type) {
  assert(type.hasRank() && "expected only ranked shapes");
  return MemRefType::get(type.getShape(), type.getElementType());
}

/// Insert an allocation and deallocation for the given MemRefType.
static Value insertAllocAndDealloc(MemRefType type, Location loc,
                                   PatternRewriter &rewriter) {
  auto alloc = rewriter.create<memref::AllocOp>(loc, type);

  // Make sure to allocate at the beginning of the block.
  auto *parentBlock = alloc->getBlock();
  alloc->moveBefore(&parentBlock->front());

  // Make sure to deallocate this alloc at the end of the block. This is fine
  // as toy functions have no control flow.
  auto dealloc = rewriter.create<memref::DeallocOp>(loc, alloc);
  dealloc->moveBefore(&parentBlock->back());
  return alloc;
}

/// This defines the function type used to process an iteration of a lowered
/// loop. It takes as input an OpBuilder, an range of memRefOperands
/// corresponding to the operands of the input operation, and the range of loop
/// induction variables for the iteration. It returns a value to store at the
/// current index of the iteration.
using LoopIterationFn = function_ref<Value(
    OpBuilder &rewriter, ValueRange memRefOperands, ValueRange loopIvs)>;

namespace {
//===----------------------------------------------------------------------===//
// ToyToLinalg RewritePatterns: Binary operations
//===----------------------------------------------------------------------===//

SmallVector<Value> condenseValues(const SmallVector<Value> &values) {
  SmallVector<Value> condensedValues;
  for (auto value : values)
    if (value)
      condensedValues.push_back(value);
  return condensedValues;
}

template <typename BinaryOp, typename LoweredBinaryOp>
struct BinaryOpLowering : public ConversionPattern {
  BinaryOpLowering(MLIRContext *ctx)
      : ConversionPattern(BinaryOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *operation, ArrayRef<Value>,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = operation->getLoc();

    assert(operation->getNumResults() == 1 &&
           "All TOSA elementwise ops should only return a single result.");

    auto results = operation->getResults();
    auto resultTy = operation->getResult(0).getType().dyn_cast<ShapedType>();

    if (!resultTy)
      return rewriter.notifyMatchFailure(operation,
                                         "All results must be a shaped type");

    unsigned rank = resultTy.getRank();

    // Construct the indexing maps needed for linalg.generic ops.
    SmallVector<Type> bodyArgTypes;

    for (Value in : operation->getOperands())
      bodyArgTypes.emplace_back(getElementTypeOrSelf(in.getType()));

    SmallVector<Type> opResultTypes;
    SmallVector<Value> emptyTensors;

    SmallVector<Value> dynDims;
    dynDims.resize(results.front().getType().cast<ShapedType>().getRank());

    for (auto arg : operation->getOperands()) {
      auto operandTy = arg.getType().cast<ShapedType>();
      for (int i = 0; i < operandTy.getRank(); i++) {
        if (operandTy.isDynamicDim(i) && !dynDims[i])
          dynDims[i] = rewriter.create<tensor::DimOp>(loc, arg, i);
      }
    }

    SmallVector<Value> filteredDims = condenseValues(dynDims);

    for (auto result : results) {
      auto resultTy = result.getType().template cast<ShapedType>();
      emptyTensors.push_back(rewriter.create<tensor::EmptyOp>(
          loc, resultTy.getShape(), resultTy.getElementType(), filteredDims));
      opResultTypes.push_back(result.getType());
    }

    auto bodyResultTypes = llvm::to_vector<4>(llvm::map_range(
        emptyTensors, [](Value v) { return getElementTypeOrSelf(v); }));

    SmallVector<Value, 2> operands;
    SmallVector<AffineMap, 2> indexingMaps;
    indexingMaps.reserve(operation->getNumOperands() + bodyResultTypes.size());

    // Input indexing maps may be broadcasted.
    for (Value operand : operation->getOperands()) {
      ShapedType type = operand.getType().cast<ShapedType>();

      if (type.getShape() == resultTy.getShape()) {
        operands.push_back(operand);
        indexingMaps.push_back(rewriter.getMultiDimIdentityMap(rank));
        continue;
      }

      SmallVector<int64_t, 5> newShape;
      SmallVector<AffineExpr, 4> affineExprs;
      newShape.reserve(type.getRank());
      for (const auto &it : llvm::enumerate(type.getShape())) {
        if (it.value() == resultTy.getDimSize(it.index())) {
          newShape.push_back(it.value());
          affineExprs.push_back(
              mlir::getAffineDimExpr(it.index(), rewriter.getContext()));
        }
      }

      if (newShape.size() != rank) {
        // operand = rewriter.create<tosa::ReshapeOp>(
        //     loc, RankedTensorType::get(newShape, type.getElementType()),
        //     operand, rewriter.getI64ArrayAttr(newShape));
        return rewriter.notifyMatchFailure(operation, "Op shape mismatch.");
      }

      operands.push_back(operand);
      indexingMaps.push_back(AffineMap::get(
          /*dimCount=*/rank, /*symbolCount=*/0, affineExprs,
          rewriter.getContext()));
    }

    indexingMaps.append(operation->getNumResults(),
                        rewriter.getMultiDimIdentityMap(rank));

    bool didEncounterError = false;
    auto linalgOp = rewriter.create<linalg::GenericOp>(
        loc, opResultTypes, operands, emptyTensors, indexingMaps,
        SmallVector<StringRef>(rank, getParallelIteratorTypeName()),
        [&](OpBuilder &nestedBuilder, Location nestedLoc,
            ValueRange blockArgs) {
          Value opResult = rewriter.create<LoweredBinaryOp>(
              loc, blockArgs.take_front(operation->getNumOperands()));
          if (!opResult) {
            didEncounterError = true;
            return;
          }
          nestedBuilder.create<linalg::YieldOp>(loc, opResult);
        });

    if (didEncounterError)
      return failure();

    rewriter.replaceOp(operation, linalgOp->getResults());
    return success();
  }
};
using AddOpLowering = BinaryOpLowering<tc::AddOp, arith::AddFOp>;
using MulOpLowering = BinaryOpLowering<tc::MulOp, arith::MulFOp>;

//===----------------------------------------------------------------------===//
// ToyToLinalg RewritePatterns: Constant operations
//===----------------------------------------------------------------------===//

/// Previously we convert it into memref manually, but it will generate
/// too much code so that it makes optimization slowly.
struct ConstantOpLowering : public OpRewritePattern<tc::ConstantOp> {
  using OpRewritePattern<tc::ConstantOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tc::ConstantOp op,
                                PatternRewriter &rewriter) const final {
    DenseElementsAttr constantValue = op.getValue();
    Location loc = op->getLoc();

    auto data = op.getValue();
    Type dataTy = op.getType();

    // Convert toy.constant -> arith.constant
    auto aConstOp = rewriter.create<arith::ConstantOp>(loc, data, dataTy);

    // but it not working.
    rewriter.replaceOp(op, {aConstOp});
    // rewriter.eraseOp(op);
    // aConstop->getBlock()->dump();
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ToyToLinalg RewritePatterns: Func operations
//===----------------------------------------------------------------------===//

struct FuncOpLowering : public OpConversionPattern<tc::FuncOp> {
  using OpConversionPattern<tc::FuncOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tc::FuncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    // We only lower the main function as we expect that all other functions
    // have been inlined.
    if (op.getName() != "main")
      return failure();

    // Verify that the given main has no inputs and results.
    if (op.getNumArguments() || op.getFunctionType().getNumResults()) {
      return rewriter.notifyMatchFailure(op, [](Diagnostic &diag) {
        diag << "expected 'main' to have 0 inputs and 0 results";
      });
    }

    // Create a new non-toy function, with the same region.
    auto func = rewriter.create<mlir::func::FuncOp>(op.getLoc(), op.getName(),
                                                    op.getFunctionType());
    rewriter.inlineRegionBefore(op.getRegion(), func.getBody(), func.end());
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ToyToLinalg RewritePatterns: Print operations
//===----------------------------------------------------------------------===//

struct PrintOpLowering : public OpConversionPattern<tc::PrintOp> {
  using OpConversionPattern<tc::PrintOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tc::PrintOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    // We don't lower "toy.print" in this pass, but we need to update its
    // operands.
    rewriter.updateRootInPlace(op,
                               [&] { op->setOperands(adaptor.getOperands()); });
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ToyToLinalg RewritePatterns: Return operations
//===----------------------------------------------------------------------===//

struct ReturnOpLowering : public OpRewritePattern<tc::ReturnOp> {
  using OpRewritePattern<tc::ReturnOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tc::ReturnOp op,
                                PatternRewriter &rewriter) const final {
    // During this lowering, we expect that all function calls have been
    // inlined.
    if (op.hasOperand())
      return failure();

    // We lower "toy.return" directly to "func.return".
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ToyToLinalg RewritePatterns: Transpose operations
//===----------------------------------------------------------------------===//

struct TransposeOpLowering : public ConversionPattern {
  TransposeOpLowering(MLIRContext *ctx)
      : ConversionPattern(tc::TransposeOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    auto transOp = dyn_cast<tc::TransposeOp>(op);

    // TODO: for now only support "linear algebra transpose", we should make it
    // more generic.
    SmallVector<int64_t> weightPerm;
    for (int64_t i = transOp.getType().getShape().size(); i != 0; --i) {
      weightPerm.push_back(i - 1);
    }
    auto weightPermAttr = DenseI64ArrayAttr::get(getContext(), weightPerm);

    auto initTensor = rewriter.create<tensor::EmptyOp>(
        loc, transOp.getType().getShape(), transOp.getType().getElementType());
    auto linTransOp = rewriter.create<linalg::TransposeOp>(
        loc, transOp.getInput(), initTensor, weightPermAttr);
    rewriter.replaceOp(op, linTransOp.getResult());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ToyToLinalg RewritePatterns: Matmul operations
//===----------------------------------------------------------------------===//

// TODO: here we convert to linalg.matmul, next we need convert into
// affine loops.
struct MatmulOpLowering : public ConversionPattern {
  MatmulOpLowering(MLIRContext *ctx)
      : ConversionPattern(tc::MatmulOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    tc::MatmulOp mmOp = llvm::dyn_cast<tc::MatmulOp>(op);
    assert(operands.size() == 2);

    auto ty = mmOp.getType();
    // due to matmul is a += matmul(b, c), so, we need a empty tensor a.
    auto resTensor = rewriter.create<tensor::EmptyOp>(loc, ty.getShape(),
                                                      ty.getElementType());

    auto linalgMatmulOp = rewriter.create<linalg::MatmulOp>(
        loc, ValueRange{operands[0], operands[1]}, ValueRange{resTensor});

    // convert linalg.matmul into loops.
    // (void)linalg::linalgOpToAffineLoops(rewriter, linalgMatmulOp);

    // linalgMatmulOp->erase();

    rewriter.replaceOp(op, {linalgMatmulOp.getResult(0)});
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// ToyToLinalgLoweringPass
//===----------------------------------------------------------------------===//

/// This is a partial lowering to affine loops of the toy operations that are
/// computationally intensive (like matmul for example...) while keeping the
/// rest of the code in the Toy dialect.

namespace {
struct ToyToLinalgLoweringPass
    : public PassWrapper<ToyToLinalgLoweringPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ToyToLinalgLoweringPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect, func::FuncDialect, memref::MemRefDialect,
                    linalg::LinalgDialect>();
  }
  void runOnOperation() final;
};
} // namespace

void ToyToLinalgLoweringPass::runOnOperation() {
  // The first thing to define is the conversion target. This will define the
  // final target for this lowering.
  ConversionTarget target(getContext());

  // We define the specific operations, or dialects, that are legal targets for
  // this lowering. In our case, we are lowering to a combination of the
  // `Affine`, `Arith`, `Func`, and `MemRef` dialects.
  target.addLegalDialect<AffineDialect, BuiltinDialect, arith::ArithDialect,
                         func::FuncDialect, memref::MemRefDialect,
                         linalg::LinalgDialect, tensor::TensorDialect>();

  // We also define the Toy dialect as Illegal so that the conversion will fail
  // if any of these operations are *not* converted. Given that we actually want
  // a partial lowering, we explicitly mark the Toy operations that don't want
  // to lower, `toy.print`, as `legal`. `toy.print` will still need its operands
  // to be updated though (as we convert from TensorType to MemRefType), so we
  // only treat it as `legal` if its operands are legal.
  target.addIllegalDialect<tc::ToyDialect>();
  target.addDynamicallyLegalOp<tc::PrintOp>([](tc::PrintOp op) {
    return llvm::all_of(op->getOperandTypes(),
                        [](Type type) { return type.isa<TensorType>(); });
  });

  // Now that the conversion target has been defined, we just need to provide
  // the set of patterns that will lower the Toy operations.
  RewritePatternSet patterns(&getContext());
  patterns.add<AddOpLowering, ConstantOpLowering, FuncOpLowering, MulOpLowering,
               PrintOpLowering, ReturnOpLowering, TransposeOpLowering,
               MatmulOpLowering>(&getContext());
  // tosa::populateTosaToLinalgConversionPatterns(&patterns);

  // With the target and rewrite patterns defined, we can now attempt the
  // conversion. The conversion will signal failure if any of our `illegal`
  // operations were not converted successfully.
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

/// Create a pass for lowering operations in the `Affine` and `Std` dialects,
/// for a subset of the Toy IR (e.g. matmul).
std::unique_ptr<Pass> mlir::tc::createLowerToLinalgPass() {
  return std::make_unique<ToyToLinalgLoweringPass>();
}

# How to add Op

## add op define

in `include/tc/Ops.td`:

```
//===----------------------------------------------------------------------===//                                                                              
// TransposeOp
//===----------------------------------------------------------------------===//
    
def TransposeOp : Toy_Op<"transpose",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "transpose operation";
    
  let arguments = (ins F64Tensor:$input);
  let results = (outs F64Tensor);
    
  let assemblyFormat = [{
    `(` $input `:` type($input) `)` attr-dict `to` type(results)
  }];
    
  // Enable registering canonicalization patterns with this operation.
  let hasCanonicalizer = 1;
    
  // Allow building a TransposeOp with from the input operand.
  let builders = [
    OpBuilder<(ins "Value":$input)>
  ];
    
  // Indicate that additional verification for this operation is necessary.
  let hasVerifier = 1;
}
```

in `mlir/Dialect.cpp`, mainly add shape inference support.

```
//===----------------------------------------------------------------------===//
// MatmulOp
//===----------------------------------------------------------------------===//

// TODO: add verify func to check shape.

void MatmulOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                  mlir::Value lhs, mlir::Value rhs) {
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands({lhs, rhs});
}

mlir::ParseResult MatmulOp::parse(mlir::OpAsmParser &parser,
                               mlir::OperationState &result) {
  return parseBinaryOp(parser, result);
}

void MatmulOp::print(mlir::OpAsmPrinter &p) { printBinaryOp(p, *this); }

/// Infer the output shape of the MatmulOp, this is required by the shape inference
/// interface.
/*
 * a x b . b x c -> a x c
 * {other dims A} x a x b . {other dims B} x b x c -> max{A, B} x a x c
 */
void MatmulOp::inferShapes() {
  auto lhsType = getOperand(0).getType().cast<RankedTensorType>();
  auto rhsType = getOperand(1).getType().cast<RankedTensorType>();

  SmallVector<int64_t, 4> lhsDims(lhsType.getShape());
  SmallVector<int64_t, 4> rhsDims(rhsType.getShape());

  assert(lhsDims.size() == rhsDims.size() && "MatmulOp: lhs and rhs shape not match");
  // TODO: add shape check.
  // TODO: add auto dim match.
  // assert(lhsDims.back() == (rhsDims.back() - 1) && "MatmulOp: lhs and rhs shape not match.");

  SmallVector<int64_t, 4> resDims;
  auto size = lhsDims.size();
  for(int i = 0; i < size; ++ i){
    if(i == size - 2){
      resDims.push_back(lhsDims[i]);
    } else if(i == size - 1){
      resDims.push_back(rhsDims[i]);
    } else{
      resDims.push_back(std::max(lhsDims[i], rhsDims[i]));
    }
  }

  getResult().setType(RankedTensorType::get(resDims, lhsType.getElementType()));
}
```

in `mlir/LowerToAffineLoops.cpp`, convert `toy.matmul` into related affine code.
here I simply use `linalg.matmul` and convert it into affine loops.

```
//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: Matmul operations
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

    auto memRefType = convertTensorToMemRef(mmOp.getType());

    // TODO: find better place to save result, and fuse with potential AddOp.
    auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

    auto linalgMatmulOp = rewriter.create<linalg::MatmulOp>(loc, ValueRange{operands[0], operands[1]}, ValueRange{alloc});

    // convert linalg.matmul into loops.
    (void)linalg::linalgOpToAffineLoops(rewriter, linalgMatmulOp);

    linalgMatmulOp->erase();

    rewriter.replaceOp(op, alloc);
    return success();
  }
};
```

that's all, don't forget to add related test case under `test` folder ;).

```
def main(){
  a = [[1, 2]];
  b = [[1], [3]];
  c = [[100]];
  print(a . b . c); # matmul
}
```

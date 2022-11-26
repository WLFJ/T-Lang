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

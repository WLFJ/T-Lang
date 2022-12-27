#include <iostream>
#include <fstream>
#include <memory>
#include "mlir/Conversion/TosaToLinalg/TosaToLinalg.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/BufferizationToMemRef/BufferizationToMemRef.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Passes.h"
#include "parser.hh"
#include "tc/AST.h"
#include "tc/driver.h"

#include "tc/Dialect.h"
#include "tc/MLIRGen.h"
#include "tc/Passes.h"

#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

using namespace tc;

int dumpLLVMIR(mlir::ModuleOp module) {
  // Register the translation to LLVM IR with the MLIR context.
  mlir::registerLLVMDialectTranslation(*module->getContext());

  // Convert the module to LLVM IR in a new LLVM IR context.
  llvm::LLVMContext llvmContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
  if (!llvmModule) {
    llvm::errs() << "Failed to emit LLVM IR\n";
    return -1;
  }

  // Initialize LLVM targets.
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  mlir::ExecutionEngine::setupTargetTriple(llvmModule.get());

  /// Optionally run an optimization pipeline over the llvm module.
  /// TODO: disabled ope: enableOpt into false
  auto optPipeline = mlir::makeOptimizingTransformer(
      /*optLevel=*/false ? 3 : 0, /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);
  if (auto err = optPipeline(llvmModule.get())) {
    llvm::errs() << "Failed to optimize LLVM IR " << err << "\n";
    return -1;
  }

  // dump to .bc file for static link.
  std::error_code EC;
  llvm::raw_fd_ostream OS("module.llvmir", EC, llvm::sys::fs::OF_None);
  llvm::WriteBitcodeToFile(*llvmModule, OS);
  OS.flush();
  return 0;
}

int main(void){
  Driver drv;
  yy::parser parser(drv);
  auto res = parser.parse();
  if(!res){
    mlir::MLIRContext context;
    // Load our Dialect in this MLIR Context.
    context.getOrLoadDialect<mlir::tc::ToyDialect>();
    auto module = mlirGen(context, *drv.tcProgram);

    {
	    mlir::PassManager pm(&context);
	    mlir::OpPassManager &optPM = pm.nest<mlir::tc::FuncOp>();

	    pm.addPass(mlir::createInlinerPass());
	    optPM.addPass(mlir::createCanonicalizerPass());
	    (void)pm.run(*module);

	    module->dump();
    }
    {
	    mlir::PassManager pm(&context);
	    mlir::OpPassManager &optPM = pm.nest<mlir::tc::FuncOp>();
	    optPM.addPass(mlir::tc::createShapeInferencePass());
	    optPM.addPass(mlir::createCanonicalizerPass());
	    optPM.addPass(mlir::createCSEPass());

	    (void)pm.run(*module);

	    module->dump();
    }

    {
	    mlir::PassManager pm(&context);
	    mlir::OpPassManager &optPM = pm.nest<mlir::tc::FuncOp>();

	    pm.addPass(mlir::tc::createLowerToLinalgPass());
	    // optPM.addPass(mlir::tosa::createTosaToLinalg());

	    (void)pm.run(*module);

	    module->dump();
    }

    {
	    mlir::PassManager pm(&context);
	    mlir::OpPassManager &optPM = pm.nest<mlir::tc::FuncOp>();

	    pm.addPass(mlir::arith::createArithBufferizePass());

	    mlir::OpPassManager &funcPM = pm.nest<mlir::func::FuncOp>();

	    funcPM.addPass(mlir::bufferization::createEmptyTensorToAllocTensorPass());
	 
	    funcPM.addPass(mlir::createLinalgBufferizePass());
	    funcPM.addPass(mlir::createConvertLinalgToLoopsPass());

	    pm.addPass(mlir::tc::createToyBufferizePass());


	    funcPM.addPass(mlir::createConvertLinalgToAffineLoopsPass());

	    funcPM.addPass(mlir::createLowerAffinePass());

	    auto res = pm.run(*module);

	    module->dump();
    }

    {
	    mlir::PassManager pm(&context);
	    mlir::OpPassManager &optPM = pm.nest<mlir::tc::FuncOp>();

	    pm.addPass(mlir::tc::createLowerToLLVMPass()); // lower toy.print, buffrize and lower to LLVM.


	    auto res = pm.run(*module);

	    module->dump();
    }

    dumpLLVMIR(*module);

    return 0;
  }
  return res;
}

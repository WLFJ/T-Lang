#include <iostream>
#include <fstream>
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Linalg/Passes.h"
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
    // auto dumper = tc::ASTDumper;
    tc::dump(*drv.tcProgram);

    std::cout << "---------AST-END-------------" << std::endl;

    mlir::MLIRContext context;
    // Load our Dialect in this MLIR Context.
    context.getOrLoadDialect<mlir::tc::ToyDialect>();
    auto module = mlirGen(context, *drv.tcProgram);
    module->dump();

    std::cout << "---------MLIR-END-------------" << std::endl;

    {
      mlir::PassManager pm(&context);
      // applyPassManagerCLOptions(pm);
      pm.addPass(mlir::createInlinerPass());

      std::cout << "---------Inliner--------------" << std::endl;

      (void)pm.run(*module);
      module->dump();

      std::cout << "---------Inliner-END----------" << std::endl;
    }
    

    {
      mlir::PassManager pm(&context);
      mlir::OpPassManager &optPM = pm.nest<mlir::tc::FuncOp>();
      optPM.addPass(mlir::createCanonicalizerPass());
      optPM.addPass(mlir::tc::createShapeInferencePass());
      optPM.addPass(mlir::createCanonicalizerPass());
      optPM.addPass(mlir::createCSEPass());

      (void)pm.run(*module);

      std::cout << "---------Canonical-ShapeInfer-Canonical-CSE---" << std::endl;
      module->dump();
      std::cout << "---------Canonical-ShapeInfer-Canonical-CSE---" << std::endl;
    }

    {
      mlir::PassManager pm(&context);
      pm.addPass(mlir::tc::createLowerToAffinePass());

      std::cout << "---------Affine--------------" << std::endl;

      (void)pm.run(*module);
      module->dump();

      std::cout << "---------Affine-END----------" << std::endl;
    }

    {
      mlir::PassManager pm(&context);
      mlir::OpPassManager &optPM = pm.nest<mlir::func::FuncOp>();
      optPM.addPass(mlir::createLoopFusionPass());
      optPM.addPass(mlir::createAffineParallelizePass());

      (void)pm.run(*module);

      std::cout << "---------Fusion-Parallel---" << std::endl;
      module->dump();
      std::cout << "---------Fusion-Parallel---" << std::endl;

    }
    
    {
      mlir::PassManager pm(&context);
      mlir::OpPassManager &optPM = pm.nest<mlir::func::FuncOp>();
      optPM.addPass(mlir::createLowerAffinePass());
      optPM.addPass(mlir::createConvertSCFToCFPass());

      (void)pm.run(*module);

      std::cout << "---------Fusion-Parallel---" << std::endl;
      module->dump();
      std::cout << "---------Fusion-Parallel---" << std::endl;

    }

    // for now only have cf.

    {
      mlir::PassManager pm(&context);
      pm.addPass(mlir::tc::createLowerToLLVMPass());

      (void)pm.run(*module
      dumpLLVMIR(*module);
    }

    return 0;
  }
  return res;
}

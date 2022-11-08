cmake_minimum_required(VERSION 3.17)

project(tc)

set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")
set(MLIR_DIR 3rdparty/llvm-project/build/lib/cmake/mlir)

find_package(BISON)
find_package(FLEX)
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

include_directories(${CMAKE_CURRENT_BINARY_DIR})

set(LLVM_ENABLE_RTTI ON)
set(LLVM_ENABLE_EH ON)

# TODO: for now we just keep these stuff global.

# add_subdirectory(tc)
BISON_TARGET(Parser tc/parser.yy parser.cc
  COMPILE_FLAGS "--debug -Wcounterexamples"
  DEFINES_FILE parser.hh)
FLEX_TARGET(Scanner tc/scanner.ll scanner.cc)

set(LLVM_LINK_COMPONENTS
  Core
  Support
  nativecodegen
  OrcJIT
  )

include_directories(include)
add_subdirectory(include)

set(LLVM_TARGET_DEFINITIONS mlir/ToyCombine.td)
mlir_tablegen(ToyCombine.inc -gen-rewriters)
add_public_tablegen_target(CombineIncGen)

add_llvm_executable(tc
  tc.cpp
  tc/AST.cpp
  tc/driver.cpp
  ${BISON_Parser_OUTPUTS}
  ${FLEX_Scanner_OUTPUTS}
  mlir/MLIRGen.cpp
  mlir/Dialect.cpp
  mlir/LowerToAffineLoops.cpp
  mlir/LowerToLLVM.cpp
  mlir/ShapeInferencePass.cpp
  mlir/ToyCombine.cpp

  DEPENDS
  ShapeInferenceInterfaceIncGen
  OpsIncGen
  CombineIncGen
  )

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
target_link_libraries(tc
  PRIVATE
    ${dialect_libs}
    ${conversion_libs}
    MLIRAnalysis
    MLIRCallInterfaces
    MLIRCastInterfaces
    MLIRExecutionEngine
    MLIRIR
    MLIRLLVMCommonConversion
    MLIRLLVMToLLVMIRTranslation
    MLIRMemRefDialect
    MLIRParser
    MLIRPass
    MLIRSideEffectInterfaces
    MLIRTargetLLVMIRExport
    MLIRTransforms
    )



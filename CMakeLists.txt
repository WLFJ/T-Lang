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
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
# include_directories(${PROJECT_SOURCE_DIR}/include)
# include_directories(${PROJECT_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

# For a better template to copy, see examples/standalone
include_directories(include)
# add_subdirectory(include)


BISON_TARGET(Parser parser.yy parser.cc
  DEFINES_FILE parser.hh)
FLEX_TARGET(Scanner scanner.ll scanner.cc)

include_directories(${CMAKE_CURRENT_BINARY_DIR})

# add_executable(tc tc.cpp ${BISON_Parser_OUTPUTS} ${FLEX_Scanner_OUTPUTS})

set(LLVM_ENABLE_RTTI ON)
set(LLVM_ENABLE_EH ON)

add_llvm_executable(tc
  tc.cpp
  AST.cpp
  driver.cpp
  ${BISON_Parser_OUTPUTS}
  ${FLEX_Scanner_OUTPUTS}
  )



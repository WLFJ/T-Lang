#!/bin/bash

git submodule update --init --depth 1

if [ ! -d 3rdparty/llvm-project/build ]
then
	mkdir 3rdparty/llvm-project/build
fi

cd 3rdparty/llvm-project/build
cmake -G Ninja ../llvm \
   -DLLVM_ENABLE_PROJECTS=mlir \
   -DLLVM_BUILD_EXAMPLES=OFF \
   -DLLVM_TARGETS_TO_BUILD="X86;NVPTX;AMDGPU" \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DLLVM_ENABLE_RTTI=true

ninja

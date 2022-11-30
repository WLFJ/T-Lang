#!/usr/bin/env bash
export PATH=$PWD/build/bin:$PWD/3rdparty/llvm-project/build/bin:$PATH
tc < $@
llc -filetype=obj module.llvmir -o module.o
clang module.o -o a.out

rm module.llvmir module.o

echo "Compile Finished."

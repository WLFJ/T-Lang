#!/usr/bin/env bash
export PATH=$PWD/build/bin:$PWD/3rdparty/llvm-project/build/bin:$PATH
tc < $@
llc --relocation-model=pic -filetype=obj module.llvmir -o module.o
clang -fuse-ld=lld module.o -o a.out

rm module.llvmir module.o

echo "Compile Finished."

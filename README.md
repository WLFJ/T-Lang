# T-Lang

A Tensor based Programing Language.

[![Build](https://github.com/WLFJ/T-Lang/actions/workflows/cmake.yml/badge.svg)](https://github.com/WLFJ/T-Lang/actions/workflows/cmake.yml)

## Usage

[How to add Op](doc/how-to-add-op.md)

### How to build

note: due to we need latest bison, so you may need compile
from source and add soft link under `/usr/bin/bison` to make
sure cmake find it.

```bash
sudo apt install cmake flex ninja-build clang
git clone git@github.com:WLFJ/T-Lang.git
cd T-Lang
git submodule update --init --depth 1
mkdir 3rdparty/llvm-project/build && cd 3rdparty/llvm-project/build
cmake -G Ninja ../llvm \
   -DLLVM_ENABLE_PROJECTS=mlir \
   -DLLVM_BUILD_EXAMPLES=ON \
   -DLLVM_TARGETS_TO_BUILD="X86;NVPTX;AMDGPU" \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DLLVM_ENABLE_RTTI=true
ninja
cd -
mkdir build && cd build
cmake ..
cmake --build . -j$(nproc)
```

## How to use

1. compile project.
2. write your source code saved in `test.tc`.
3. run following command:

```bash
chmod +x build/tc && ./tcc test.tc && ./a.out
```


## How to write test cases

Coming Soon ...

## TO-DO

- [x] A simple (hello-world example) bison example.
- [x] A flex.
- [x] A AST dumper.
- [x] CMake
- [x] Migrate build system from `GNU make` to `CMake`.
- [x] ~~Mem safer (avoid useing `new` directly, instead of `unique_ptr`?).~~ No, we'll use AST from ToyLang.
- [x] ~~Simplify `yy` and `ll` file.~~ Seems it's already clean yet.
- [x] Freeze MLIR into `3rdparty`, and include it into CMakeLists. (Should we build it automatically ?).
- [x] Add backend support.
- [x] Support `print` in grammar.
- [x] Add `TIR`, make them all runnable.
- [x] Add variable dec like `var a = 1;`. (partial, id bind in grammar is needed.)
- [x] Add RTTI support cmd.
- [x] ~~Dump `llvm::Module` into file.~~ using JIT instead.
- [x] Add math calc. (for `+` `*`)
- [x] Add function fully support. (arbitary argument, return value, caller, callee).
- [x] For now sub function must have return type.
- [x] Add advanced operator (`transpost` etc.)
- [x] grammar confict fix (binop precident).
- [x] support redundent tensor decl.
- [x] support none value variable declearation.
- [x] research how ir generated.
- [ ] add `linalg` opt layer.
- [ ] writing some ref ;P.
- [ ] Add `linalg.generic` like expr support.
like this:
```
# a -> [1, 2, 3, 4] then b -> [ [1, 2], [3, 4] ]
# usage: [[ tile(a) ]]
# reshape
lin tile(a<m>) {
  map {a<i> -> <m>, b<j, k> -> <i / k, i % k>}
  (a) {
    a;
  }
} -> b<n, k>
```
- [ ] support more binop (`-`, `.`)
- [ ] FileCheck (LLVM is needed).
- [ ] Automatic Test.(Maybe `cmake test`).
- [ ] Add remain supported expr.
- [ ] Integret FileCheck into CMake. e.g. `./tcc test/case-check-FileCheck && ./a.out | FileCheck test/case-check-FileCheck`

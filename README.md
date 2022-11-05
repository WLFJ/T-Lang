# T-Lang

A Tensor based Programing Language.

## Usage

### How to build

```bash
mkdir build && cd build
cmake ..
cmake --build .
```

## How to extend the compiler

1. Edit `parser.yy` to set what grammer you want.
2. According to `parser`, you may want to add token support in `scanner.ll`
3. The same thing in `AST.hpp` and `AST.cpp`.

## How to write test cases

Coming Soon ...

## TO-DO

- [x] A simple (hello-world example) bison example.
- [x] A flex.
- [x] A AST dumper.
- [x] CMake
- [x] Migrate build system from `GNU make` to `CMake`.
- [ ] ~~Mem safer (avoid useing `new` directly, instead of `unique_ptr`?).~~ No, we'll use AST from ToyLang.
- [ ] ~~Simplify `yy` and `ll` file.~~ Seems it's already clean yet.
- [ ] Freeze MLIR into `3rdparty`, and include it into CMakeLists. (Should we build it automatically ?)
- [ ] FileCheck (LLVM is needed)
- [ ] Automatic Test.(Maybe `cmake test`)

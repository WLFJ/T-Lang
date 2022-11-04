# T-Lang

A Tensor based Programing Language.

## Usage

### Build And Run

```bash
make clean && make run
```

## How to getting start

1. Edit `parser.yy` to set what grammer you want.
2. According to `parser`, you may want to add token support in `scanner.ll`
3. The same thing in `AST.hpp` and `AST.cpp`.

## How to write test cases

## TO-DO

- [x] A simple (hello-world example) bison example.
- [x] A flex.
- [x] A AST dumper.
- [ ] CMake
- [ ] Migrate build system from `GNU make` to `CMake`.
- [ ] FileCheck
- [ ] Automatic Test.(Maybe `cmake test`)
- [ ] Mem safer (avoid useing `new` directly, instead of `unique_ptr`?).
- [ ] Simplify `yy` and `ll` file.

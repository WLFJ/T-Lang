BISON_TARGET(Parser parser.yy parser.cc
  DEFINES_FILE parser.hh)
FLEX_TARGET(Scanner scanner.ll scanner.cc)

add_llvm_executable(tc
  tc.cpp
  AST.cpp
  driver.cpp
  ${BISON_Parser_OUTPUTS}
  ${FLEX_Scanner_OUTPUTS}
  )



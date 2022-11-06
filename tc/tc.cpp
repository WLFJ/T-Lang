#include <iostream>
#include "tc/parser.hh"
#include "tc/AST.h"
#include "tc/driver.h"

int main(void){
  Driver drv;
  yy::parser parser(drv);
  auto res = parser.parse();
  if(!res){
    // auto dumper = tc::ASTDumper;
    tc::dump(*drv.tcProgram);
  }
  return res;
}

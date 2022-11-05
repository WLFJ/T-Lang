#include <iostream>
#include "tc/parser.hh"
#include "tc/AST.h"
#include "tc/driver.h"

int main(void){
  Driver drv;
  yy::parser parser(drv);
  return 0;
}

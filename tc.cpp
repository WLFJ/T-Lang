#include <iostream>
#include "parser.hh"
#include "AST.hpp"
#include "driver.hpp"

int main(void){
  Driver drv;
  yy::parser parser(drv);

  int res = parser.parse();
  if(!res){
    std::cout << "TC: parse finished, no error accure." << std::endl;
    ASTNode *ast = drv.astTree;
    ast->dump();
  }
  return 0;
}

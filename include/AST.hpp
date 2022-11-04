#include <string>
#include <iostream>

#ifndef __AST_H__
#define __AST_H__

class ASTNode{
  public:
  virtual void dump(){
    std::cout << "??" << std::endl;
  };
};

class Hello: public ASTNode {
  public:
  Hello(std::string val): val(val){};

  Hello() = default; 

  virtual void dump(){
    std::cout << "ASTNode: HELLO, val: [" << val << "]" << std::endl;
  };

  private:
    std::string val;
};

#endif

/*
  最简单的parser应该长什么样？
  我们希望是个C++的，并且能够和flex交互
*/

%require "3.2"
%language "c++"
%header

%define api.token.constructor
%define api.value.type variant
%define parse.assert

%code requires{
  #include "AST.hpp" 
  class Driver;
}

%param { Driver& drv }

%code {
  #include "driver.hpp"
}

%token
  LPAREN "("
  RPAREN ")"
;
%token <std::string> HELLO;

%nterm <ASTNode*> start-here;

%%
start-here:
  LPAREN HELLO RPAREN  { $$ = drv.astTree = new Hello($2); }
;

%%

void
yy::parser::error (const std::string& m)
{
  std::cerr << "parser: " << m << '\n';
}

%require "3.2"
%language "c++"
%header

%define api.token.constructor
%define api.value.type variant
%define parse.assert

%code requires{
  #include "tc/AST.h" 
  class Driver;
}

%param { Driver& drv }

%code {
  #include "tc/driver.h"
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

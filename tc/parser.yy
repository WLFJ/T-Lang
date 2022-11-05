%require "3.2"
%language "c++"
%header

%define api.token.constructor
%define api.value.type variant
%define parse.assert
/* %define api.location.type {Location} */

%code requires{
  #include "tc/AST.h" 
  class Driver;
}

%param { Driver& drv }

%locations

%code {
  #include "tc/driver.h"
}


%%

module-list:
  module
| module-list module
;

module:
  define
;

define:
  prototype block
;

prototype:
  DEF ID LPAREN decl-list RPAREN
;

decl-list:
  ID
| ID COMMA decl-list
;

block:
  LCBRACE expression-list RCBRACE
;

expression-list:
  block-expr SEMI expression-list
;

block-expr:
  decl-or-call
| RETURN
| expr
;

decl-or-call:
  typed-declaration
| call
;

/* we'll do type checking in AST, maybe not now? */
typed-declaration:
  ID ID EQUAL expression
;

/* TODO: only one expr for now */
call:
  LPAREN expression RPAREN
;

/* TODO: support more operator, for now only + */
expression:
  primary-lhs primary-rhs
;

primary-lhs:
  %empty
| primary
;

/* TODO: for now only A + B, not A + B + C */
primary-rhs:
  PLUS primary
;

primary:
  identifier-expr
| number-expr
| paren-expr
| tensor-literal-expr
;

identifier-expr:
  identifier
| LPAREN identifier RPAREN
;

number-expr:
  NUMBER
;

paren-expr:
  LPAREN expression RPAREN
;

tensor-literal-expr:
  LSBRACE literal-list RSBRACE
| NUMBER
;

literal-list:
  tensor-literal-expr
| tensor-literal-expr COMMA literal-list
;

%%

void
yy::parser::error (const location_type& l, const std::string& m)
{
  std::cerr << "parser: " << m << '\n';
}

%require "3.2"
%language "c++"
%header

%define api.token.constructor
%define api.value.type variant
%define parse.assert
/* %define api.location.type {Location} */

%code requires{
  #include "tc/AST.h" 
  using namespace tc;
  class Driver;
}

%param { Driver& drv }

%locations

%code {
  #include "tc/driver.h"
}

%define api.token.prefix {TOK_}

%token
  DEF		"def"
  LPAREN	"("
  RPAREN	")"
  COMMA		","
  LCBRACE	"{"
  RCBRACE	"}"
  SEMI		";"
  RETURN	"return"
  EQUAL		"="
  PLUS		"+"
  LSBRACE	"["
  RSBRACE	"]"
;

%token <std::string> ID "identifier"
%token <int> NUMBER "number"

%nterm <std::unique_ptr<RecordAST>> module
%nterm <std::vector<std::unique_ptr<RecordAST>>> module-list
%nterm <std::unique_ptr<FunctionAST>> define
%nterm <std::unique_ptr<PrototypeAST>> prototype
%nterm <std::unique_ptr<ExprASTList>> block

%%
program:
  module-list {
    drv.tcProgram = std::make_unique<ModuleAST>(std::move($1));
  }
;

module-list:
  module {
    std::vector<std::unique_ptr<RecordAST>> m;
    m.push_back(std::move($1));
    $$ = std::move(m);
  }
| module-list module {
    $1.push_back(std::move($2));
    $$ = std::move($1);
  }
;

/* we'll extend to support more global var? */
module:
  define {
    $$ = std::move($1);
  }
;

define:
  prototype block {
    auto block = std::make_unique<FunctionAST>(std::move($1), std::move($2));
    $$ = std::move(block);
  }
;

/* removed decl-list */
prototype:
  DEF ID LPAREN  RPAREN {
    std::vector<std::unique_ptr<VarDeclExprAST>> args;
    $$ = std::make_unique<PrototypeAST>(std::move(@1), $2,
					  std::move(args));
  }
;

/* removed: expression-list */
block:
  LCBRACE RCBRACE {
    $$ = std::make_unique<ExprASTList>();
  }
;

/* current at here :) */

decl-list:
  ID
| ID COMMA decl-list
;


expression-list:
  block-expr SEMI expression-list
;

block-expr:
  decl-or-call
| RETURN
| expression
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
  ID
| LPAREN ID RPAREN
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

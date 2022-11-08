%require "3.2"
%language "c++"
%header

%define api.token.constructor
%define api.value.type variant
%define parse.assert
/* %define api.location.type {Location} */

%code requires{
  #include "tc/AST.h" 
  #define YYDEBUG 1
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
  LCBRACE	"{" RCBRACE	"}"
  SEMI		";"
  RETURN	"return"
  EQUAL		"="
  PLUS		"+"
  LSBRACE	"["
  RSBRACE	"]"
  PRINT         "print"
;

%token <std::string> ID "identifier"
%token <int> NUMBER "number"
%token <std::string> STRING "string"

%nterm <std::unique_ptr<RecordAST>> module
%nterm <std::vector<std::unique_ptr<RecordAST>>> module-list
%nterm <std::unique_ptr<FunctionAST>> define
%nterm <std::unique_ptr<PrototypeAST>> prototype
%nterm <std::unique_ptr<ExprASTList>> block
%nterm <std::unique_ptr<ExprAST>> expression
%nterm <std::unique_ptr<ExprASTList>> expression-list
%nterm <std::unique_ptr<ExprAST>> block-expr
%nterm <std::unique_ptr<ExprAST>> decl-or-call
%nterm <std::unique_ptr<ExprAST>> call
%nterm <std::unique_ptr<ExprAST>> primary
%nterm <std::unique_ptr<ExprAST>> number-expr

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
  LCBRACE expression-list RCBRACE {
    $$ = std::move($2);
  }
;

expression-list:
  block-expr
  {
    auto exprList = std::move(std::make_unique<ExprASTList>());
    exprList->push_back(std::move($1));
    $$ = std::move(exprList);
  }
| block-expr SEMI expression-list
  {
    auto exprList = std::move($3);
    exprList->push_back(std::move($1));
    $$ = std::move(exprList);
  }
;

block-expr:
  decl-or-call
  {
    $$ = std::move($1);
  }
;

/* removed typed-declaration */
decl-or-call:
  call
  {
    $$ = std::move($1);
  }
;

/* TODO: only one expr for now */
/* TODO: only print now */
/* ID LPAREN expression RPAREN */
call:
  PRINT LPAREN expression RPAREN
  {
    $$ = std::move(std::make_unique<PrintExprAST>(@1, std::move($3)));
  }
;

/* TODO: support more operator, for now only + */
expression:
  primary
  {
    $$ = std::move($1);
  }
;

/* here we define what data type we support. */
primary:
  number-expr
  {
    $$ = std::move($1);
  }
;

number-expr:
  NUMBER
  {
    $$ = std::move(std::make_unique<NumberExprAST>(@1, $1));
  }
;

%%

void
yy::parser::error (const location_type& l, const std::string& m)
{
  std::cerr << "parser: " << m << '\n';
}

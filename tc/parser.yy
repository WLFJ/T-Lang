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
  LCBRACE	"{" 
  RCBRACE	"}"
  SEMI		";"
  RETURN	"return"
  EQUAL		"="
  PLUS		"+"
  LSBRACE	"["
  RSBRACE	"]"
  PRINT         "print"
  LS		"<"
  GT		">"
  VAR		"var"
  PLUS		"+"
  MINUS		"-"
  STAR		"*"
  DOT		"."
;

%token <std::string> ID "identifier"
%token <int> NUMBER "number"
%token <std::string> STRING "string"

%nterm <std::unique_ptr<RecordAST>> module
%nterm <std::vector<std::unique_ptr<RecordAST>>> module-list
%nterm <std::vector<std::unique_ptr<ExprAST>>> literal-list
%nterm <std::unique_ptr<FunctionAST>> define
%nterm <std::unique_ptr<PrototypeAST>> prototype
%nterm <std::unique_ptr<ExprASTList>> block
%nterm <std::unique_ptr<ExprAST>> expression
%nterm <std::unique_ptr<ExprASTList>> expression-list
%nterm <std::vector<std::unique_ptr<ExprAST>>> expression-list-comma
%nterm <std::unique_ptr<ExprAST>> block-expr
%nterm <std::unique_ptr<ExprAST>> decl-or-call
%nterm <std::unique_ptr<ExprAST>> call
%nterm <std::unique_ptr<ExprAST>> primary
%nterm <std::unique_ptr<ExprAST>> number-expr
%nterm <std::unique_ptr<ExprAST>> tensor-literal
%nterm <std::unique_ptr<ExprAST>> identifier-expr
%nterm <std::unique_ptr<ExprAST>> paren-expr
%nterm <std::unique_ptr<ReturnExprAST>> return
%nterm <std::unique_ptr<VarDeclExprAST>> declaration
%nterm <std::unique_ptr<VarType>> type
%nterm <std::vector<int64_t>> shape-list


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

block:
  LCBRACE expression-list RCBRACE {
    $$ = std::move($2);
  }
;

/* TODO: here we may need to find a elegent way to handle expr; expr; .. */
expression-list:
  block-expr SEMI
  {
    std::vector<std::unique_ptr<ExprAST>> exprList;
    exprList.push_back(std::move($1));
    $$ = std::make_unique<ExprASTList>(std::move(exprList));
  }
| expression-list block-expr SEMI
  {
    auto exprList = std::move($1);
    exprList->push_back(std::move($2));
    $$ = std::move(exprList);
  }
;

block-expr:
  return
  {
    $$ = std::move($1);
  }
| declaration
  {
    $$ = std::move($1);
  }
| expression
  {
    $$ = std::move($1);
  }
;

return:
  RETURN
  {
    $$ = std::move(std::make_unique<ReturnExprAST>(@1, llvm::Optional<std::unique_ptr<ExprAST>>()));
  }
| RETURN expression
  {
    $$ = std::move(std::make_unique<ReturnExprAST>(@1, std::move($2)));
  }
;

/*
 For variable declaration we design this pattern:
 [var]? identifier [[ tensor size ]]? ( = expr )?;
*/
declaration:
  VAR ID type
  {
    $$ = std::move(std::make_unique<VarDeclExprAST>(@1, $2, *($3)));
  }
| VAR ID type EQUAL expression
  {
    $$ = std::move(std::make_unique<VarDeclExprAST>(@1, $2, *($3), std::move($5)));
  }
| VAR ID EQUAL expression
  {
    $$ = std::move(std::make_unique<VarDeclExprAST>(@1, $2, (VarType()), std::move($4)));
  }
| ID type EQUAL expression
  {
    $$ = std::move(std::make_unique<VarDeclExprAST>(@1, $1, *($2), std::move($4)));
  }
| ID EQUAL expression
  {
    $$ = std::move(std::make_unique<VarDeclExprAST>(@1, $1, (VarType()), std::move($3)));
  }
;

type:
  LS shape-list GT 
  {
    auto type = std::make_unique<VarType>();
    type->shape = std::move($2);
    $$ = std::move(type);
  }
;

shape-list:
  NUMBER
  {
    std::vector<int64_t> sizeList;
    sizeList.push_back($1);
    $$ = std::move(sizeList); 
  }
| shape-list COMMA NUMBER
  {
    auto sizeList = std::move($1);
    sizeList.push_back($3);
    $$ = std::move(sizeList); 
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

/* TODO: struct-literal-expr is not support for now. */
/* TODO: std::move redundent will remove via bison config. */
primary:
  identifier-expr
  {
    $$ = std::move($1);
  }
| paren-expr
  {
    $$ = std::move($1);
  }
| tensor-literal
  {
    $$ = std::move($1);
  }
| number-expr
  {
    $$ = std::move($1);
  }
;

/* -> up<ExprAST> parseIdentifierExpr() */
identifier-expr:
  ID
  {
    $$ = std::move(std::make_unique<VariableExprAST>(@1, $1));
  }
/* function call */
/*
| ID LPAREN expression-list-comma RPAREN
  {
    $$ = std::move(std::make_unique<CallExprAST>(@1, std::string($1), std::move($3)));
  }
*/
| call
  {
    $$ = std::move($1);
  }
;

/* further test is needed:
print(hi());
print(hi(a));
print(hi(a, b));
*/
expression-list-comma:
  %empty
  {
    std::vector<std::unique_ptr<ExprAST>> exprList;
    $$ = std::move(exprList);
  }
| expression
  {
    std::vector<std::unique_ptr<ExprAST>> exprList;
    exprList.push_back(std::move($1));
    $$ = std::move(exprList);
  }
| expression-list-comma COMMA expression
  {
    auto exprList = std::move($1);
    exprList.push_back(std::move($3));
    $$ = std::move(exprList);
  }
;

paren-expr:
  LPAREN expression RPAREN
  {
    $$ = std::move($2);
  }
;

/* -> up<LiteralExprAST> parseTensorLiteralExpr() */
tensor-literal:
  LSBRACE literal-list RSBRACE
  {
    std::vector<int64_t> dims;
    auto values = std::move($2);

    // First add all current layer dims.
    dims.push_back(values.size());
    // Then handle nested dim.
    // TODO: for now only support 1-d tensor.
    $$ = std::make_unique<LiteralExprAST>(@1, std::move(values), dims);
  }
| number-expr
  {
    $$ = std::move($1);
  }
;

literal-list:
  tensor-literal
  {
    std::vector<std::unique_ptr<ExprAST>> values;
    values.push_back(std::move($1));
    $$ = std::move(values);
  }
| literal-list COMMA tensor-literal
  {
    auto values = std::move($1);
    values.push_back(std::move($3));
    $$ = std::move(values);
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

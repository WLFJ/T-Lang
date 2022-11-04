/*
 * A simple driver interact with parser.
 */

#ifndef __DRIVER_H__
#define __DRIVER_H__

#include "AST.hpp"
#include "parser.hh"

#define YY_DECL \
  yy::parser::symbol_type yylex (Driver& drv)

YY_DECL;

class Driver {
  public:
    ASTNode *astTree;
};

#endif

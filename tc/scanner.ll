%{ /* -*- C++ -*- */
# include <cerrno>
# include <climits>
# include <cstdlib>
# include <cstring> // strerror
# include <string>
# include "../include/driver.hpp"
# include "parser.hh"
%}

/* no debug */
%option noyywrap nounput noinput batch

%{
  // A number symbol corresponding to the value in S.
  yy::parser::symbol_type
  make_HELLO (const char* str);
%}

blank [ \t\r]

%%
{blank}+      ;
\n+           ;

"("           return yy::parser::make_LPAREN ();
")"           return yy::parser::make_RPAREN ();

"hello"       return yy::parser::make_HELLO (yytext);

.          {
  throw yy::parser::syntax_error
  ("invalid character: " + std::string(yytext));
}

<<EOF>>    return yy::parser::make_YYEOF ();
%%

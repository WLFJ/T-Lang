%{ /* -*- C++ -*- */
# include <cerrno>
# include <climits>
# include <cstdlib>
# include <cstring> // strerror
# include <string>
# include "tc/driver.h"
# include "tc/parser.hh"
%}

/* no debug */
%option noyywrap nounput noinput batch

%{
  // A number symbol corresponding to the value in S.
%}

blank [ \t\r]

%{
  # define YY_USER_ACTION loc.columns (yyleng);
%}
%%
%{
  yy::location& loc = drv.location;
  loc.step ();
%}
{blank}+      loc.step();
\n+           loc.lines (yyleng); loc.step ();

.          {
  throw yy::parser::syntax_error
  (loc, "invalid character: " + std::string(yytext));
}

<<EOF>>    return yy::parser::make_YYEOF (loc);
%%

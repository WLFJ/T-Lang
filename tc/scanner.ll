%{ /* -*- C++ -*- */
# include <cerrno>
# include <climits>
# include <cstdlib>
# include <cstring> // strerror
# include <string>
# include "tc/driver.h"
# include "parser.hh"
%}

/* no debug */
%option noyywrap nounput noinput batch debug

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

{blank}+        loc.step();
\n+             loc.lines (yyleng); loc.step ();
"#".*           loc.lines (yyleng); loc.step ();

"("		return yy::parser::make_LPAREN(loc);
")"		return yy::parser::make_RPAREN(loc);
"["		return yy::parser::make_LSBRACE(loc);
"]"		return yy::parser::make_RSBRACE(loc);
"{"		return yy::parser::make_LCBRACE(loc);
"}"		return yy::parser::make_RCBRACE(loc);

"<"		return yy::parser::make_LS(loc);
">"		return yy::parser::make_GT(loc);

";"		return yy::parser::make_SEMI(loc);
","		return yy::parser::make_COMMA(loc);

"="		return yy::parser::make_EQUAL(loc);

"def"		return yy::parser::make_DEF(loc);
"var"		return yy::parser::make_VAR(loc);
"print"		return yy::parser::make_PRINT(loc);
"return"	return yy::parser::make_RETURN(loc);

[0-9]+		return yy::parser::make_NUMBER(std::stoi(yytext), loc);

[a-z]+		return yy::parser::make_ID(yytext, loc);

.          {
  throw yy::parser::syntax_error
  (loc, "invalid character: " + std::string(yytext));
}

<<EOF>>    return yy::parser::make_YYEOF (loc);
%%

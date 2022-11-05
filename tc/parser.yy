%require "3.2"
%language "c++"
%header

%define api.token.constructor
%define api.value.type variant
%define parse.assert
%define api.location.type {Location}

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
noting-here:

;

%%

void
yy::parser::error (const location_type& l, const std::string& m)
{
  std::cerr << "parser: " << m << '\n';
}

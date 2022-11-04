cmake_minimum_required(VERSION 3.17)

project(tc)

find_package(BISON)
find_package(FLEX)

BISON_TARGET(Parser parser.yy parser.cc
  DEFINES_FILE parser.hh)
FLEX_TARGET(Scanner scanner.ll scanner.cc)

include_directories(${CMAKE_CURRENT_BINARY_DIR})
add_executable(tc tc.cpp ${BISON_Parser_OUTPUTS} ${FLEX_Scanner_OUTPUTS})

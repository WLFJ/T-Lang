#!/bin/bash
echo "install bison..."
sudo scripts/install-gnu-bison.sh

# sudo rm -rf bison
echo "compile llvm ..."
scripts/llvm-compile.sh

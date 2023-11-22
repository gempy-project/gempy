#!/bin/bash

source ~/.virtualenvs/gempy_dependencies/bin/activate

cd ../docs || exit
#make clean
make html
cd - || exit
#!/bin/bash

source ~/.virtualenvs/gempy-geotop-pilot/bin/activate

cd ../docs || exit
#make clean
make html
cd - || exit
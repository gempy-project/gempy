#!/usr/bin/env bash

# sets the working directory to the current directory
cd '../notebooks/tutorials'

# converts all Jupyter Notebook files to basic html
# for f in *.ipynb; do jupyter nbconvert --to rst --output-dir ../../docs/source/_notebooks $f; done
for f in *.ipynb; do jupytext --to py:sphinx --output ./examples $f; done


# cd ../../docs
# make html
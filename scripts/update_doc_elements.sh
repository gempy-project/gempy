#!/bin/bash  

# #!/bin/bash

# sets the working directory to the current directory
cd './notebooks/tutorials'

# converts all Jupyter Notebook files to basic html
for f in *.ipynb; do jupyter nbconvert --to rst --output-dir ../../docs/source/_notebooks $f; done

cd ../../docs
make html

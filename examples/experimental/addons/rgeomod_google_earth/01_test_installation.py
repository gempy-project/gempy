# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: sphinx
#       format_version: '1.1'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

"""
# 1 - Installation test

This notebook is for testing the dependencies for rgeomod. Just run the following code cell.

<div class="alert alert-info">
**Your task**: Run the code in the following cells (either using the "Run"-button above, or by usign the corresponding shortcut (typically <code>"Shift"+"Enter"</code>, see Help function above): 
</div>



"""

###############################################################################
# #### Fundamental libraries

import numpy, sys, os, tqdm
print("Fundamental Python libraries imported correctly.")

###############################################################################
# #### Plotting libraries

import matplotlib.pyplot, mplstereonet
from mpl_toolkits.mplot3d import Axes3D
print("Plotting libraries imported correctly.")

###############################################################################
# #### 3D-Visualization library

###############################################################################
# <div class="alert alert-danger">
# **Important note**: 3D-Visualization using VTK is not available if you use Docker to run this exercise. This module is **optional**.
# </div>

try:
    import vtk
    print("3D-Visualization library VTK imported correctly.")
except: 
    print("3D-Visualization library VTK not found.")
    print("This module is optional for the 3D visualization of data and geological models.")

###############################################################################
# #### remote-geomod

sys.path.append(r"..")  # append local path to access rgeomod module
import rgeomod
print("Python package remote-geomod imported correctly.")

###############################################################################
# <div class="alert alert-danger">
# **Important note**: If importing `rgeomod` fails, then make sure that you pulled it correctly from github (especially when using the rgeomod-dependencies docker image).
# </div>

###############################################################################
# #### GemPy
#
# <div class="alert alert-danger">
# **Important note**: Any warnings that occur when loading the library can safely be ignored for the context of the exercise. If you want to get rid of the warning notes, simply execute the cell twice.
# </div>
#

sys.path.append("../../gempy/")
sys.path.append("../gempy/")
import gempy
print("3D-Modeling library GemPy imported correctly.")

###############################################################################
# <div class="alert alert-danger">
# If importing `gempy` fails, then check that the relative path is set correctly in `sys.path.append()` above.
# </div>

###############################################################################
# <div class="alert alert-success">
# If all packages were imported correctly, then you should see `... imported correctly` after executing each cell - and you are good to go with the following exercises! If you get a warning message from gempy, then it can be ignored (you can execute the cell again, then this warning should disappear).
# </div>



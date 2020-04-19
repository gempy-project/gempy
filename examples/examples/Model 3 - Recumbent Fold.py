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
# Model 3 - Recumbent Fold
"""

###############################################################################
# A recumbent (overturned) fold. We start by importing the necessary dependencies:

# These two lines are necessary only if GemPy is not installed
import sys, os
sys.path.append("../..")

# Importing GemPy
import gempy as gp

# Embedding matplotlib figures in the notebooks
# %matplotlib inline

# Importing auxiliary libraries
import numpy as np
import matplotlib.pyplot as plt

###############################################################################
# Creating the model by importing the input data and displaying it:

path_to_data = os.pardir+"/data/input_data/jan_models/"

geo_data = gp.create_data([0,1000,0,1000,0,1000],resolution=[50,50,50], 
                        path_o = path_to_data + "model3_orientations.csv",
                        path_i = path_to_data + "model3_surface_points.csv") 

""
geo_data.get_data()

###############################################################################
# Setting and ordering the units and series:

gp.map_series_to_surfaces(geo_data, {"Strat_Series": ('rock2','rock1'),"Basement_Series":('basement')})

""
# %matplotlib inline
gp.plot.plot_data(geo_data, direction='y')

###############################################################################
# Calculating the model:

interp_data = gp.set_interpolation_data(geo_data, theano_optimizer='fast_compile')

""
geo_data.additional_data

""
sol = gp.compute_model(geo_data)

###############################################################################
# Displaying the result in x and y direction:

# %matplotlib inline
gp.plot.plot_section(geo_data, cell_number=25,
                         direction='y', show_data=True)

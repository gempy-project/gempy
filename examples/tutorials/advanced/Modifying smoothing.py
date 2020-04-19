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
# Modifing smoothing
"""

###############################################################################
# A simple anticline structure. We start by importing the necessary dependencies:

# These two lines are necessary only if GemPy is not installed
import sys, os
sys.path.append("../../..")

# Importing GemPy
import gempy as gp

# Embedding matplotlib figures in the notebooks
# %matplotlib inline

# Importing auxiliary libraries
import numpy as np
import matplotlib.pyplot as plt

###############################################################################
# Creating the model by importing the input data and displaying it:

path_to_data = os.pardir+"/../data/input_data/jan_models/"

geo_data = gp.create_data([0,1000,0,1000,0,1000],resolution=[50,50,50], 
                          path_o = path_to_data + "model2_orientations.csv",
                          path_i = path_to_data + "model2_surface_points.csv") 

""
geo_data.get_data().head()

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
geo_data.update_to_interpolator()

""
geo_data.orientations

""
geo_data.interpolator.theano_graph.nugget_effect_grad_T.get_value()

""
sol = gp.compute_model(geo_data)

""


###############################################################################
# Displaying the result in y and x direction:

# %matplotlib inline
gp.plot.plot_section(geo_data, cell_number=15,
                         direction='y', show_data=True)

""
# %matplotlib inline
gp.plot.plot_section(geo_data, cell_number=25,
                         direction='x', show_data=True)

""
vtkplot = gp.plot.plot_3D(geo_data, )

""
vtkplot.set_real_time_on()

""
a = geo_data.surface_points.df['X'] > 510
sel = geo_data.surface_points.df.index[a]
sel

""
geo_data.modify_surface_points(sel, smooth=100, plot_object=vtkplot)

""
geo_data.interpolator.theano_graph.nugget_effect_scalar_T.get_value()

""


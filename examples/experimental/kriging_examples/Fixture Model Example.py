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

# These two lines are necessary only if GemPy is not installed
import sys, os
sys.path.append("../../..")
# just here as it is subfolder in experimental
os.pardir = '../..'

# Importing GemPy
import gempy as gp

# Embedding matplotlib figures in the notebooks
# %matplotlib inline

# Importing auxiliary libraries
#import numpy as np
#import matplotlib.pyplot as plt

""
path_to_data = os.pardir+"/data/input_data/jan_models/"

geo_data = gp.create_data([0,2500,0,1000,0,1000],resolution=[50,20,20], 
                        path_o = path_to_data + "fixture_model_orientations.csv",
                        path_i = path_to_data + "fixture_model_surfaces.csv") 

""
gp.map_series_to_surfaces(geo_data, {"Fault_Series": ('fault'), "Strat_Series1": ('rock3'),
                                     "Strat_Series2": ('rock2','rock1'),
                                     "Basement_Series":('basement')})

geo_data.set_is_fault(['Fault_Series'])

""
geo_data.surfaces.colors.change_colors({'fault': '#000000', 'rock1': '#CC081F', 'rock2': '#FFAA00',
                                        'rock3': '#006C8C', 'basement': '#097703'})

""
geo_data.surfaces

""
# %matplotlib inline
gp.plot.plot_data(geo_data, direction='y')

""
interp_data = gp.set_interpolation_data(geo_data, theano_optimizer='fast_compile')

""
sol = gp.compute_model(geo_data)

""
# %matplotlib inline
gp.plot.plot_section(geo_data, cell_number=1,
                         direction='y', show_data=True)

""
gp._plot.plot_3d(geo_data)

""


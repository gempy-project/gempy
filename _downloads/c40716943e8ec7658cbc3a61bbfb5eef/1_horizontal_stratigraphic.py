"""
Model 1 - Horizontal stratigraphic
==================================

"""

# %%
# This is the most simple model of horizontally stacked layers. We start
# by importing the necessary dependencies:
# 

# %%
# Importing GemPy
import gempy as gp

import pandas as pd
pd.set_option('precision', 2)

# %%
# Creating the model by importing the input data and displaying it:
# 

# %%
data_path = 'https://raw.githubusercontent.com/cgre-aachen/gempy_data/master/'
geo_data = gp.create_data('horizontal', extent=[0, 1000, 0, 1000, 0, 1000], resolution=[50, 50, 50],
                          path_o=data_path + "/data/input_data/jan_models/model1_orientations.csv",
                          path_i=data_path + "/data/input_data/jan_models/model1_surface_points.csv")

# %%
# Setting and ordering the units and series:
# 

# %% 
gp.map_stack_to_surfaces(geo_data, {"Strat_Series": ('rock2', 'rock1'), "Basement_Series": ('basement')})

# %% 
gp.plot_2d(geo_data, direction=['y'])

# %%
# Calculating the model:
# 

# %% 
interp_data = gp.set_interpolator(geo_data, compile_theano=True,
                                  theano_optimizer='fast_compile')

# %% 
sol = gp.compute_model(geo_data)

# %%
# Displaying the result in x and y direction:
# 

# %%
gp.plot_2d(geo_data, cell_number=[25],
           direction=['x'], show_data=True)

# %%
# sphinx_gallery_thumbnail_number = 2
gp.plot_2d(geo_data, cell_number=[25],
           direction=['y'], show_data=True)

gp.save_model(geo_data)
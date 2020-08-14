"""
Model 4 - Pinchout
==================

"""

# %%
# Forcing GemPy to create a layer of varying thickness. We start by
# importing the necessary dependencies:
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

path_to_data = data_path + "/data/input_data/jan_models/"
geo_data = gp.create_data('pinchout',
                          extent=[0, 1000, 0, 1000, 0, 1000], resolution=[50, 50, 50],
                          path_o=path_to_data + "model4_orientations.csv",
                          path_i=path_to_data + "model4_surface_points.csv")

# %% 
geo_data.get_data()

# %%
# Setting and ordering the units and series:
# 

# %% 
gp.map_stack_to_surfaces(geo_data,
                         {"Strat_Series": ('rock2', 'rock1'),
                          "Basement_Series": ('basement')})

# %%
gp.plot_2d(geo_data, direction=['y'])

# %%
# Calculating the model:
# 

# %% 
interp_data = gp.set_interpolator(geo_data, theano_optimizer='fast_compile')

# %% 
sol = gp.compute_model(geo_data)

# %%
# Displaying the result in x and y direction:
# 

# %%
gp.plot_2d(geo_data, cell_number=[25],
           direction=['x'], show_data=True)

# %%
# sphinx_gallery_thumbnail_number = 3
gp.plot_2d(geo_data, cell_number=[25],
           direction=['y'], show_data=True)

# %%
gp.save_model(geo_data)
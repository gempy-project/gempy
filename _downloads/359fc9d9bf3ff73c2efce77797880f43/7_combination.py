"""
Model 7 - Combination
======================

"""

# %%
# A folded domain featuring an unconformity and a fault. We start by importing
# the necessary dependencies:
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

geo_data = gp.create_data('combination',
                          extent=[0, 2500, 0, 1000, 0, 1000],
                          resolution=[125, 50, 50],
                          path_o=path_to_data + "model7_orientations.csv",
                          path_i=path_to_data + "model7_surface_points.csv")

# %%
geo_data.get_data()

# %%
# Setting and ordering the units and series:
#

# %%
gp.map_stack_to_surfaces(geo_data, {"Fault_Series": ('fault'), "Strat_Series1": ('rock3'),
                                     "Strat_Series2": ('rock2','rock1'),
                                     "Basement_Series":('basement')})

geo_data.set_is_fault(['Fault_Series'])

# %%
gp.plot_2d(geo_data, direction='y')

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
gp.plot_2d(geo_data, cell_number=5,
           direction='y', show_data=False, show_boundaries=True)

# %%
# sphinx_gallery_thumbnail_number = 2
gp.plot_2d(geo_data, cell_number=5,
           direction='x', show_data=True)

# %%
gp.plot_3d(geo_data)
gp.save_model(geo_data)
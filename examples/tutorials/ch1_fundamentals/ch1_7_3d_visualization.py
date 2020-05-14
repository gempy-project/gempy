"""
1.7: 3-D Visualization
======================

"""

# %% 
# Importing GemPy
import gempy as gp

# Importing auxiliary libraries
import numpy as np
import matplotlib.pyplot as plt

# %%
# Loading an example geomodel
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 

# %%

data_path = 'https://raw.githubusercontent.com/cgre-aachen/gempy_data/master/'

geo_model = gp.create_data('viz_3d',
                           [0, 2000, 0, 2000, 0, 1600],
                           [50, 50, 50],
                           path_o=data_path + "data/input_data/lisa_models/foliations" + str(7) + ".csv",
                           path_i=data_path + "data/input_data/lisa_models/interfaces" + str(7) + ".csv"
                           )

gp.map_stack_to_surfaces(
    geo_model,
    {"Fault_1": 'Fault_1', "Fault_2": 'Fault_2',
     "Strat_Series": ('Sandstone', 'Siltstone', 'Shale', 'Sandstone_2', 'Schist', 'Gneiss')}
)

geo_model.set_is_fault(['Fault_1', 'Fault_2'])
geo_model.set_topography()

gp.set_interpolator(geo_model)
gp.compute_model(geo_model, compute_mesh=True)

# %%
# Basic plotting API
# ------------------
# 


# %%
# Data plot
# ~~~~~~~~~
# 

# %% 
gp.plot_3d(geo_model, show_surfaces=False, show_data=True, show_lith=False, image=False)

# %%
# Geomodel plot
# ~~~~~~~~~~~~~
# 

# %% 
gp.plot_3d(geo_model, image=False)

# %%

# sphinx_gallery_thumbnail_number = 2
gpv = gp.plot.plot_3d(geo_model,
                      plotter_type='basic',off_screen=False,
                      show_topography=True,
                      show_scalar=False,
                      show_lith=True,
                      kwargs_plot_structured_grid={'opacity': .5})


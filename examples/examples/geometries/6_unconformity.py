"""
Model 6 - Unconformity
======================

"""

# %%
# An unconformity cutting an anticline structure. We start by importing
# the necessary dependencies:
# 

# %%
# Importing GemPy
import gempy as gp
import gempy_viewer as gpv

# %%
# Creating the model by importing the input data and displaying it:
# 

# %%
data_path = 'https://raw.githubusercontent.com/cgre-aachen/gempy_data/master/'
path_to_data = data_path + "/data/input_data/jan_models/"

geo_data = gp.create_data(
    project_name='unconformity',
    extent=[0, 1000, 0, 1000, 0, 1000],
    resolution=[50, 50, 50],
    path_o=path_to_data + "model6_orientations.csv",
    path_i=path_to_data + "model6_surface_points.csv"
)

# %%
# Setting and ordering the units and series:
# 

# %% 
gp.map_stack_to_surfaces(
    gempy_model=geo_data,
    mapping_object={
        "Strat_Series1": ('rock3'),
        "Strat_Series2": ('rock2', 'rock1')
    }
)

# %%
gpv.plot_2d(geo_data, direction='y')

# %%
# Calculating the model:
# 

# %% 
sol = gp.compute_model(geo_data)

# %%
# Displaying the result in x and y direction:
# 

# %%
gpv.plot_2d(geo_data, cell_number=25, direction='y', show_data=True)

# %%
# sphinx_gallery_thumbnail_number = 2
gpv.plot_2d(geo_data, cell_number=25, direction='x', show_data=True)

"""
Model 3 - Recumbent Fold
========================

"""

# %%
# A recumbent (overturned) fold. We start by importing the necessary
# dependencies:
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
    project_name='recumbent',
    extent=[0, 1000, 0, 1000, 0, 1000],
    resolution=[50, 50, 50],
    path_o=path_to_data + "model3_orientations.csv",
    path_i=path_to_data + "model3_surface_points.csv"
)


# %%
# Setting and ordering the units and series:
# 

# %% 
gp.map_stack_to_surfaces(
    gempy_model=geo_data,
    mapping_object= {"Strat_Series": ('rock2', 'rock1')}
)

# %%
gpv.plot_2d(geo_data, direction=['y'])

# %%
# Calculating the model:
# 

# %% 
geo_data.interpolation_options

# %% 
sol = gp.compute_model(geo_data)

# %%
# Displaying the result in x and y direction:
# 

# %%
# sphinx_gallery_thumbnail_number = 2
gpv.plot_2d(geo_data, cell_number=[25], direction=['y'], show_data=True)

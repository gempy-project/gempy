"""
Model 2 - Anticline
===================

"""

# %%
# A simple anticline structure. We start by importing the necessary
# dependencies:
# 

# Importing GemPy
import gempy as gp
import gempy_viewer as gpv
from gempy.optional_dependencies import require_gempy_legacy
from gempy_3.gp3_to_gp2_input import gempy3_to_gempy2

# %%
# Creating the model by importing the input data and displaying it:
# 

# %%
data_path = 'https://raw.githubusercontent.com/cgre-aachen/gempy_data/master/'
path_to_data = data_path + "/data/input_data/jan_models/"
geo_data = gp.create_data(
    project_name='fold',
    extent=[0, 1000, 0, 1000, 0, 1000],
    resolution=[50, 5, 50],
    path_o=path_to_data + "model2_orientations.csv",
    path_i=path_to_data + "model2_surface_points.csv"
)

# %% 
geo_data.structural_frame.surface_points.df.head()  # This view needs to have pandas installed

# %%
# Setting and ordering the units and series:
# 

# %% 
gp.map_stack_to_surfaces(
    gempy_model=geo_data,
    mapping_object={"Strat_Series": ('rock2', 'rock1')}
)

# %%
gpv.plot_2d(geo_data, direction=['y'])

# %%
# Calculating the model:
# 

# %% 
geo_data.orientations

# %% 

if COMPUTE_LEGACY := True:
    gpl = require_gempy_legacy()
    legacy_model: gpl.Project = gempy3_to_gempy2(geo_data)
    gpl.set_interpolator(legacy_model, verbose=['cov_gradients', 'cov_surface_points', 'cov_interface_gradients',
    'U_I', 'U_G']) 
    gpl.compute_model(legacy_model)
    gpl.plot_2d(legacy_model, direction=['y'])

    gpl.plot_2d(legacy_model, direction=['y'], show_data=True, show_scalar=True)
    
geo_data.interpolation_options.tensor_dtype = 'float64'
sol = gp.compute_model(geo_data)


# %%
# Displaying the result in y and x direction:
# 

# %%
gpv.plot_2d(geo_data, direction='y', show_data=True)
gpv.plot_2d(geo_data, direction='y', show_scalar=True)

# %%
# sphinx_gallery_thumbnail_number = 2
gpv.plot_2d(geo_data, direction='x', show_data=True)

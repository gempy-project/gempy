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
from gempy_engine.core.data.stack_relation_type import StackRelationType
from gempy_engine.modules.octrees_topology.octrees_topology_interface import ValueType
from gempy_engine.plugins.plotting.helper_functions import plot_block_and_input_2d

# %%
# Creating the model by importing the input data and displaying it:
# 

# %%
data_path = 'https://raw.githubusercontent.com/cgre-aachen/gempy_data/master/'
path_to_data = data_path + "/data/input_data/jan_models/"

geo_data = gp.create_data_legacy(
    project_name='unconformity',
    extent=[0, 1000, 0, 1000, 0, 1000],
    resolution=[50, 5, 50],
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
        "Strat_Series1": 'rock3',
        "Strat_Series2": ('rock2', 'rock1')
    }
)

# %%
geo_data.structural_frame.structural_groups[0].structural_relation = StackRelationType.ERODE


# %%
gpv.plot_2d(geo_data, direction='y')

# %%
# Calculating the model:
# 

# %% 
sol = gp.compute_model(geo_data)

# %%
plot_block_and_input_2d(
    stack_number=0,
    interpolation_input=geo_data.interpolation_input,
    outputs=geo_data.solutions.octrees_output,
    structure=geo_data.structural_frame.input_data_descriptor.stack_structure,
    value_type=ValueType.mask_component
)

plot_block_and_input_2d(
    stack_number=0,
    interpolation_input=geo_data.interpolation_input,
    outputs=geo_data.solutions.octrees_output,
    structure=geo_data.structural_frame.input_data_descriptor.stack_structure,
    value_type=ValueType.squeeze_mask
)

plot_block_and_input_2d(
    stack_number=1,
    interpolation_input=geo_data.interpolation_input,
    outputs=geo_data.solutions.octrees_output,
    structure=geo_data.structural_frame.input_data_descriptor.stack_structure,
    value_type=ValueType.squeeze_mask
)
# %%
# Displaying the result in x and y direction:
# 

# %%
gpv.plot_2d(geo_data, direction='y', show_data=True)

# %%
# sphinx_gallery_thumbnail_number = 2
gpv.plot_2d(geo_data, direction='x', show_data=True)

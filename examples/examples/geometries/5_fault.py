"""
Model 5 - Fault
===============

"""
import numpy as np

# %%
# A simple fault model with constant offset. We start by importing the
# necessary dependencies:
# 

# %%
# Importing GemPy
import gempy as gp
import gempy_viewer as gpv
from gempy_engine.core.data.stack_relation_type import StackRelationType

from gempy_engine.modules.octrees_topology.octrees_topology_interface import ValueType
from plugins.plotting.helper_functions import plot_block_and_input_2d

# %%
# Creating the model by importing the input data and displaying it:
# 

# %%
data_path = 'https://raw.githubusercontent.com/cgre-aachen/gempy_data/master/'
path_to_data = data_path + "/data/input_data/jan_models/"

geo_data = gp.create_data_legacy(
    project_name='fault',
    extent=[0, 1000, 0, 1000, 0, 1000],
    resolution=[50, 50, 50],
    path_o=path_to_data + "model5_orientations.csv",
    path_i=path_to_data + "model5_surface_points.csv"
)

# %%
# Setting and ordering the units and series:
# 
gp.map_stack_to_surfaces(
    geo_data,
    {
        "Fault_Series": 'fault',
        "Strat_Series": ('rock2', 'rock1')
    }
)

# %%
# Define fault groups
# TODO: Abstract this away with the old set_fault method
geo_data.structural_frame.structural_groups[0].structural_relation = StackRelationType.FAULT
geo_data.structural_frame.fault_relations = np.array([[0, 1], [0, 0]])

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

plot_block_and_input_2d(
    stack_number=1,
    interpolation_input=geo_data.interpolation_input,
    outputs=geo_data.solutions.octrees_output,
    structure=geo_data.structural_frame.input_data_descriptor.stack_structure,
    value_type=ValueType.values_block
)

plot_block_and_input_2d(
    stack_number=1,
    interpolation_input=geo_data.interpolation_input,
    outputs=geo_data.solutions.octrees_output,
    structure=geo_data.structural_frame.input_data_descriptor.stack_structure,
    value_type=ValueType.ids
)
# %%
# sphinx_gallery_thumbnail_number = 2
gpv.plot_2d(geo_data, cell_number=25, direction='y', show_data=True)

# %%
gpv.plot_2d(geo_data, cell_number=25, direction='x', show_data=True)

# %% 
gpv.plot_2d(geo_data, cell_number=25, direction='y', show_data=True, show_scalar=True, series_n=1)

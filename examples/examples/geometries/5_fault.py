"""
Model 5 - Fault
===============

"""

# %%
# A simple fault model with constant offset. We start by importing the
# necessary dependencies:
# 

# %%
# Importing GemPy
import gempy as gp
import gempy_viewer as gpv
from gempy_engine.core.data.stack_relation_type import StackRelationType
from gempy_engine.test.helper_functions import plot_block_and_input_2d

from gempy_engine.modules.octrees_topology.octrees_topology_interface import ValueType

# %%
# Creating the model by importing the input data and displaying it:
# 

# %%
data_path = 'https://raw.githubusercontent.com/cgre-aachen/gempy_data/master/'
path_to_data = data_path + "/data/input_data/jan_models/"

geo_data = gp.create_data(
    project_name='fault',
    extent=[0, 1000, 0, 1000, 0, 1000],
    resolution=[50, 50, 50],
    path_o=path_to_data + "model5_orientations.csv",
    path_i=path_to_data + "model5_surface_points.csv"
)

# %%
# Setting and ordering the units and series:
# 
plot_block_and_input_2d(0, geo_data.interpolation_input, geo_data.solutions.outputs,
geo_data.solutions.structure.stack_structure, ValueType.values_block)


# %% 
gp.map_stack_to_surfaces(
    geo_data,
    {
        "Fault_Series": 'fault',
        "Strat_Series": ('rock2', 'rock1')
    }
)

# geo_data.set_is_fault(['Fault_Series'])
# TODO: Get the fault running
geo_data.structural_frame.structural_groups[0].structural_relation = StackRelationType.FAULT


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

_plot_stack_values_block(interpolation_input, outputs, structure)

# %%
# sphinx_gallery_thumbnail_number = 2
gpv.plot_2d(geo_data, cell_number=25, direction='y', show_data=True)

# %%
gpv.plot_2d(geo_data, cell_number=25, direction='x', show_data=True)

# %% 
gpv.plot_2d(geo_data, cell_number=25, direction='y', show_data=True, show_scalar=True, series_n=1)

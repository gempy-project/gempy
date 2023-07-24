"""
Model 7 - Combination
======================

"""
import numpy as np

# %%
# A folded domain featuring an unconformity and a fault. We start by importing
# the necessary dependencies:
#

# %%
# Importing GemPy
import gempy as gp
import gempy_viewer as gpv
from gempy_engine.core.data.stack_relation_type import StackRelationType

# %%
# Creating the model by importing the input data and displaying it:
#

# %%
data_path = 'https://raw.githubusercontent.com/cgre-aachen/gempy_data/master/'
path_to_data = data_path + "/data/input_data/jan_models/"

geo_data = gp.create_data(
    project_name='combination',
    extent=[0, 2500, 0, 1000, 0, 1000],
    resolution=[125, 50, 50],
    path_o=path_to_data + "model7_orientations.csv",
    path_i=path_to_data + "model7_surface_points.csv"
)

# %%
# Setting and ordering the units and series:
#

# %%
gp.map_stack_to_surfaces(
    gempy_model=geo_data,
    mapping_object={
        "Fault_Series" : ('fault'),
        "Strat_Series1": ('rock3'),
        "Strat_Series2": ('rock2', 'rock1'),
    }
)

geo_data.structural_frame.structural_groups[0].structural_relation = StackRelationType.FAULT
geo_data.structural_frame.fault_relations = np.array(
    [[0, 1, 1],
     [0, 0, 0],
     [0, 0, 0]]
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
gpv.plot_2d(geo_data, cell_number=5, direction='y', show_data=True, show_boundaries=True)

# %%
# sphinx_gallery_thumbnail_number = 2
gpv.plot_2d(geo_data, cell_number=5, direction='x', show_data=True)

# %%
# BUG: Revive 3D plotting
# gpv.plot_3d(geo_data)

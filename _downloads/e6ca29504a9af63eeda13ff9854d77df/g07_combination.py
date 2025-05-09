"""
Model 7 - Combination
======================

Combining faults and unconformities

This script creates a folded domain featuring an unconformity and a fault using GemPy,
an open-source, Python-based library for building implicit geological models.
"""

# Importing necessary libraries
import numpy as np
import gempy as gp
import gempy_viewer as gpv
from gempy_engine.core.data.stack_relation_type import StackRelationType

# sphinx_gallery_thumbnail_number = 2

# Generate the model
# Define the path to data
data_path = 'https://raw.githubusercontent.com/cgre-aachen/gempy_data/master/'
path_to_data = data_path + "/data/input_data/jan_models/"
# Create a GeoModel instance
data = gp.create_geomodel(
    project_name='combination',
    extent=[0, 2500, 0, 1000, 0, 1000],
    refinement=6,
    resolution=[20, 20, 20],
    importer_helper=gp.data.ImporterHelper(
        path_to_orientations=path_to_data + "model7_orientations.csv",
        path_to_surface_points=path_to_data + "model7_surface_points.csv"
    )
)
# Map geological series to surfaces
gp.map_stack_to_surfaces(
    gempy_model=data,
    mapping_object={
        "Fault_Series" : ('fault'),
        "Strat_Series1": ('rock3'),
        "Strat_Series2": ('rock2', 'rock1'),
    }
)
# Define the structural relation
data.structural_frame.structural_groups[0].structural_relation = StackRelationType.FAULT
data.structural_frame.fault_relations = np.array(
    [[0, 1, 1],
     [0, 0, 0],
     [0, 0, 0]]
)
# Compute the geological model
data.interpolation_options.number_octree_levels_surface = 5
gp.compute_model(data)
data.structural_frame

# %%
# Plot the initial geological model in the y direction
gpv.plot_2d(data, direction=['y'], show_results=False)

# %%
# Plot the result of the model in the y and x directions with data and boundaries
gpv.plot_2d(data, direction='y', show_data=True, show_boundaries=True)
gpv.plot_2d(data, direction='x', show_data=True)

# Plot the blocks accounting for fault blocks
gpv.plot_2d(
    model=data,
    override_regular_grid=data.solutions.raw_arrays.litho_faults_block,
    show_data=True, kwargs_lithology={'cmap': 'Set1', 'norm': None}
)

# %%
# The 3D plot is commented out due to a bug.
gpv.plot_3d(data)

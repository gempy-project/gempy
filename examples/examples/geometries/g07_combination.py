"""
Model 7 - Combination
======================

This script creates a folded domain featuring an unconformity and a fault using GemPy,
an open-source, Python-based library for building implicit geological models.
"""

# Importing necessary libraries
import numpy as np
import gempy as gp
import gempy_viewer as gpv
from gempy_engine.core.data.stack_relation_type import StackRelationType
from gempy_engine.core.data.transforms import GlobalAnisotropy

# sphinx_gallery_thumbnail_number = 2

# Generate the model
# Define the path to data
data_path = 'https://raw.githubusercontent.com/cgre-aachen/gempy_data/master/'
path_to_data = data_path + "/data/input_data/jan_models/"
# Create a GeoModel instance
data = gp.create_geomodel(
    project_name='combination',
    extent=[0, 2500, 0, 1000, 0, 1000],
    number_octree_levels=2,
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
data.update_transform(auto_anisotropy=GlobalAnisotropy.NONE)
gp.remove_structural_group_by_index(data, 1)
# gp.remove_structural_group_by_index(data, -1)

print(data.structural_frame)
data.interpolation_options.dual_contouring=False
gp.compute_model(data)
geo_data = data

# %%
# Plot the initial geological model in the y direction
gpv.plot_2d(geo_data, direction=['y'], show_results=False)

# %%
# Plot the result of the model in the y and x directions with data and boundaries
gpv.plot_2d(geo_data,  direction='y', show_data=True, show_boundaries=True)
gpv.plot_2d(geo_data,  direction='x', show_data=True)

gpv.plot_2d(geo_data,  direction='y', show_data=True,
            show_boundaries=True, show_scalar=True, show_lith=False, series_n=0)
gpv.plot_2d(geo_data,  direction='y', show_data=True,
            show_boundaries=True, show_scalar=True, show_lith=False, series_n=1)
gpv.plot_2d(geo_data,  direction='y', show_data=True,
            show_boundaries=True, show_scalar=True, show_lith=False, series_n=2)

geo_model = data
gpv.plot_2d(
    model=geo_model,
    
    override_regular_grid=geo_model.solutions.raw_arrays.mask_matrix[0],
    show_data=True, kwargs_regular_grid={'cmap': 'gray', 'norm': None}
)

gpv.plot_2d(
    model=geo_model,
    
    override_regular_grid=geo_model.solutions.raw_arrays.mask_matrix[1],
    show_data=True, kwargs_regular_grid={'cmap': 'gray', 'norm': None}
)

gpv.plot_2d(
    model=geo_model,
    
    override_regular_grid=geo_model.solutions.raw_arrays.mask_matrix_squeezed[0],
    show_data=True, kwargs_regular_grid={'cmap': 'gray', 'norm': None}
)

gpv.plot_2d(
    model=geo_model,
    
    override_regular_grid=geo_model.solutions.raw_arrays.mask_matrix_squeezed[1],
    show_data=True, kwargs_regular_grid={'cmap': 'gray', 'norm': None}
)
# 
# gpv.plot_2d(geo_data, cell_number=5, direction='y', show_data=True,
#             show_boundaries=True, show_scalar=True, show_lith=False, series_n=1)
# 
# gpv.plot_2d(geo_data, cell_number=5, direction='y', show_data=True,
#             show_boundaries=True, show_scalar=True, show_lith=False, series_n=2)
# %%
# The 3D plot is commented out due to a bug.
gpv.plot_3d(geo_data)

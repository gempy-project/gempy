"""
Model 6 - Unconformity
======================

Modeling unconformities through the combination of scalar fields

This script creates an unconformity cutting an anticline structure using GemPy,
an open-source, Python-based library for building implicit geological models.
"""

# Importing necessary libraries
import gempy as gp
import gempy_viewer as gpv
from gempy_engine.core.data.stack_relation_type import StackRelationType

# sphinx_gallery_thumbnail_number = 2


# %%
# Generate the model
# Define the path to data
data_path = 'https://raw.githubusercontent.com/cgre-aachen/gempy_data/master/'
path_to_data = data_path + "/data/input_data/jan_models/"
# Create a GeoModel instance
data = gp.create_geomodel(
    project_name='unconformity',
    extent=[0, 1000, 0, 1000, 0, 1000],
    refinement=6,
    importer_helper=gp.data.ImporterHelper(
        path_to_orientations=path_to_data + "model6_orientations.csv",
        path_to_surface_points=path_to_data + "model6_surface_points.csv"
    )
)
# Map geological series to surfaces
gp.map_stack_to_surfaces(
    gempy_model=data,
    mapping_object={
        "Strat_Series1": 'rock3',
        "Strat_Series2": ('rock2', 'rock1')
    }
)
# Define the structural relation
data.structural_frame.structural_groups[0].structural_relation = StackRelationType.ERODE

# Compute the geological model
gp.compute_model(data)
geo_data = data

# %%
# Plot the initial geological model in the y direction
gpv.plot_2d(geo_data, direction=['y'], show_results=False)

# %%
# Plot the result of the model in the y and x directions with data
gpv.plot_2d(geo_data, direction='y', show_data=True)
gpv.plot_2d(geo_data, direction='x', show_data=True)


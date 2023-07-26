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


# sphinx_gallery_thumbnail_number = 2
def generate_combination_model() -> gp.GeoModel:
    """
    Function to create a model with a folded domain featuring an unconformity and a fault,
    map the geological series to surfaces, and compute the geological model.
    """
    # Define the path to data
    data_path = 'https://raw.githubusercontent.com/cgre-aachen/gempy_data/master/'
    path_to_data = data_path + "/data/input_data/jan_models/"

    # Create a GeoModel instance
    geo_data = gp.create_geomodel(
        project_name='combination',
        extent=[0, 2500, 0, 1000, 0, 1000],
        resolution=[125, 50, 50],
        importer_helper=gp.ImporterHelper(
            path_to_orientations=path_to_data + "model7_orientations.csv",
            path_to_surface_points=path_to_data + "model7_surface_points.csv"
        )
    )

    # Map geological series to surfaces
    gp.map_stack_to_surfaces(
        gempy_model=geo_data,
        mapping_object={
            "Fault_Series" : ('fault'),
            "Strat_Series1": ('rock3'),
            "Strat_Series2": ('rock2', 'rock1'),
        }
    )

    # Define the structural relation
    geo_data.structural_frame.structural_groups[0].structural_relation = StackRelationType.FAULT
    geo_data.structural_frame.fault_relations = np.array(
        [[0, 1, 1],
         [0, 0, 0],
         [0, 0, 0]]
    )

    # Compute the geological model
    gp.compute_model(geo_data)

    return geo_data


# %%
# Generate the model
geo_data = generate_combination_model()

# %%
# Plot the initial geological model in the y direction
gpv.plot_2d(geo_data, direction=['y'], show_results=False)

# %%
# Plot the result of the model in the y and x directions with data and boundaries
gpv.plot_2d(geo_data, cell_number=5, direction='y', show_data=True, show_boundaries=True)
gpv.plot_2d(geo_data, cell_number=5, direction='x', show_data=True)

# %%
# The 3D plot is commented out due to a bug.
# gpv.plot_3d(geo_data)

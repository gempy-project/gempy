"""
Model 5 - Fault
===============

This script demonstrates how to create a simple fault model with constant offset using GemPy,
a Python-based, open-source library for implicit geological modeling.
"""

# Import necessary libraries
import gempy as gp
import gempy_viewer as gpv
import numpy as np
from gempy_engine.core.data.stack_relation_type import StackRelationType
from gempy_engine.modules.octrees_topology.octrees_topology_interface import ValueType
from plugins.plotting.helper_functions import plot_block_and_input_2d


# sphinx_gallery_thumbnail_number = 2
def generate_fault_model() -> gp.GeoModel:
    """
    Function to create a simple fault model,
    map the geological series to surfaces, and compute the geological model.
    """
    # Define the path to data
    data_path = 'https://raw.githubusercontent.com/cgre-aachen/gempy_data/master/'
    path_to_data = data_path + "/data/input_data/jan_models/"

    # Create a GeoModel instance
    geo_data = gp.create_geomodel(
        project_name='fault',
        extent=[0, 1000, 0, 1000, 0, 1000],
        resolution=[20, 5, 20],
        importer_helper=gp.ImporterHelper(
            path_to_orientations=path_to_data + "model5_orientations.csv",
            path_to_surface_points=path_to_data + "model5_surface_points.csv"
        )
    )

    # Map geological series to surfaces
    gp.map_stack_to_surfaces(
        gempy_model=geo_data,
        mapping_object={
            "Fault_Series": 'fault',
            "Strat_Series": ('rock2', 'rock1')
        }
    )

    # Define fault groups
    geo_data.structural_frame.structural_groups[0].structural_relation = StackRelationType.FAULT
    geo_data.structural_frame.fault_relations = np.array([[0, 1], [0, 0]])

    # Compute the geological model
    gp.compute_model(geo_data)

    return geo_data


# %%
# Generate the model
geo_data = generate_fault_model()

# %%
# Plot the initial geological model in the y direction
gpv.plot_2d(geo_data, direction=['y'], show_results=False)

# %%
# Plot the computed model
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

# Plot the result of the model in the x and y direction with data
gpv.plot_2d(geo_data, direction='y', show_data=True)
gpv.plot_2d(geo_data, direction='x', show_data=True)

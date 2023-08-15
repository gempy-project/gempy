"""
Model 6 - Unconformity
======================

This script creates an unconformity cutting an anticline structure using GemPy,
an open-source, Python-based library for building implicit geological models.
"""

# Importing necessary libraries
import gempy as gp
import gempy_viewer as gpv
from gempy_engine.core.data.stack_relation_type import StackRelationType
from gempy_engine.modules.octrees_topology.octrees_topology_interface import ValueType
from gempy_engine.plugins.plotting.helper_functions import plot_block_and_input_2d


# sphinx_gallery_thumbnail_number = 2
def generate_unconformity_model() -> gp.data.GeoModel:
    """
    Function to create a model with an unconformity,
    map the geological series to surfaces, and compute the geological model.
    """
    # Define the path to data
    data_path = 'https://raw.githubusercontent.com/cgre-aachen/gempy_data/master/'
    path_to_data = data_path + "/data/input_data/jan_models/"

    # Create a GeoModel instance
    geo_data = gp.create_geomodel(
        project_name='unconformity',
        extent=[0, 1000, 0, 1000, 0, 1000],
        number_octree_levels=4,
        importer_helper=gp.data.ImporterHelper(
            path_to_orientations=path_to_data + "model6_orientations.csv",
            path_to_surface_points=path_to_data + "model6_surface_points.csv"
        )
    )

    # Map geological series to surfaces
    gp.map_stack_to_surfaces(
        gempy_model=geo_data,
        mapping_object={
            "Strat_Series1": 'rock3',
            "Strat_Series2": ('rock2', 'rock1')
        }
    )

    # Define the structural relation
    geo_data.structural_frame.structural_groups[0].structural_relation = StackRelationType.ERODE

    # Compute the geological model
    gp.compute_model(geo_data)

    return geo_data


# %%
# Generate the model
geo_data = generate_unconformity_model()

# %%
# Plot the initial geological model in the y direction
gpv.plot_2d(geo_data, direction=['y'], show_results=False)

# %%
# Plot the computed model
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
# Plot the result of the model in the y and x directions with data
gpv.plot_2d(geo_data, direction='y', show_data=True)
gpv.plot_2d(geo_data, direction='x', show_data=True)


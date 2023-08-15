"""
Model 2 - Anticline
===================

This script demonstrates how to create a geological model of an anticline structure using GemPy,
a Python-based, open-source library for implicit geological modeling.
"""

# Import necessary libraries
import gempy as gp
import gempy_viewer as gpv


# sphinx_gallery_thumbnail_number = 2
def generate_anticline_model() -> gp.data.GeoModel:
    """
    Function to create a geological model of an anticline structure,
    map the geological series to surfaces, and compute the geological model.
    """
    # Define the path to data
    data_path = 'https://raw.githubusercontent.com/cgre-aachen/gempy_data/master/'
    path_to_data = data_path + "/data/input_data/jan_models/"

    # Create a GeoModel instance
    geo_data: gp.data.GeoModel = gp.create_geomodel(
        project_name='fold',
        extent=[0, 1000, 0, 1000, 0, 1000],
        number_octree_levels=4,
        importer_helper=gp.data.ImporterHelper(
            path_to_orientations=path_to_data + "model2_orientations.csv",
            path_to_surface_points=path_to_data + "model2_surface_points.csv"
        )
    )

    # Map geological series to surfaces
    gp.map_stack_to_surfaces(
        gempy_model=geo_data,
        mapping_object={"Strat_Series": ('rock2', 'rock1')}
    )

    # Compute the geological model
    gp.compute_model(geo_data)

    return geo_data


# %%
# Generate the model
geo_data = generate_anticline_model()

# %%
# Plot the initial geological model in the y direction without results
gpv.plot_2d(geo_data, direction=['y'], show_results=False)

# Plot the result of the model in the y and x direction with data and scalar
gpv.plot_2d(geo_data, direction='y', show_data=True, show_scalar=True)
gpv.plot_2d(geo_data, direction='x', show_data=True, show_scalar=True)

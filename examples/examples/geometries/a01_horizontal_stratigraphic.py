"""
Model 1 - Horizontal stratigraphic
==================================

This script demonstrates how to create a basic model of horizontally stacked layers using GemPy,
a Python-based, open-source library for implicit geological modeling.
"""

# Import necessary libraries
import gempy as gp
import gempy_viewer as gpv


# sphinx_gallery_thumbnail_number = 2
def generate_horizontal_stratigraphic_model() -> gp.GeoModel:
    """
    Function to create a geological model of horizontally stacked layers,
    map the geological series to surfaces, and compute the geological model.
    """
    # Define the path to data
    data_path = 'https://raw.githubusercontent.com/cgre-aachen/gempy_data/master/'

    # Create a GeoModel instance
    geo_data = gp.create_geomodel(
        project_name='horizontal',
        extent=[0, 1000, 0, 1000, 0, 1000],
        resolution=[50, 5, 50],
        importer_helper=gp.ImporterHelper(
            path_to_orientations=data_path + "/data/input_data/jan_models/model1_orientations.csv",
            path_to_surface_points=data_path + "/data/input_data/jan_models/model1_surface_points.csv"
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
geo_data = generate_horizontal_stratigraphic_model()

# %%
# Plot the initial geological model in the y direction without results
gpv.plot_2d(geo_data, direction=['y'], show_results=False)

# Plot the result of the model in the x and y direction with data and without boundaries
gpv.plot_2d(geo_data, cell_number=[25], direction=['x'], show_data=True, show_boundaries=False)
gpv.plot_2d(geo_data, cell_number=[25], direction=['y'], show_data=True, show_boundaries=False)

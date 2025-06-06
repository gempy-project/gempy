"""
Video Tutorial "code-along": Onlap relations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
"""


# %%
# This tutorial demonstrates step-by-step how to incorporate onlap relations to our geological models created with gempy.
# It follows the Video tutorial series available on the [gempy YouTube channel](https://www.youtube.com/@GemPy3D).
# Please follow the first and second part of the tutorials to learn the basics of modeling with gempy before diving into this tutorial.

# %%
# Video tutorial 11: Basic onlap scenario
# """"""""""""""""""""""""""""""""""""


# %%
# .. raw:: html
#
#     <iframe width="560" height="315"
#             src="https://www.youtube.com/embed/80QjnrFxubQ"
#             title="YouTube video player"
#             frameborder="0"
#             allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
#             allowfullscreen>
#     </iframe>
#
#

# %%

# Required imports
import gempy as gp
import gempy_viewer as gpv
import numpy as np

# %%

# Path to input data
data_path = "https://raw.githubusercontent.com/cgre-aachen/gempy_data/master/"
path_to_data = data_path + "/data/input_data/video_tutorials_v3/"

# %%

# Create instance of geomodel
geo_model_onlap = gp.create_geomodel(
    project_name = 'tutorial_model_onlap_1',
    extent=[0,2000,0,1000,0,1000],
    resolution=[100,50,50],
    importer_helper=gp.data.ImporterHelper(
        path_to_orientations=path_to_data+"tutorial_model_onlap_1_orientations.csv",
        path_to_surface_points=path_to_data+"tutorial_model_onlap_1_surface_points.csv"
    )
)


# %%

# Map geological series to surfaces
gp.map_stack_to_surfaces(
    gempy_model=geo_model_onlap,
    mapping_object={
        "Young_Series": ("basin_fill_2", "basin_fill_1"),
        "Old_Series": ("basin_top", "basin_bottom")
    }
)

# Alternative way of mapping geological series to surfaces
# gp.map_stack_to_surfaces(
#     gempy_model=geo_model_onlap,
#     mapping_object={
#         "Young_Series": ("basin_fill_2", "basin_fill_1"),
#         "Onlap_Series": ("basin_top"),
#         "Old_Series": ("basin_bottom")
#     }
# )

# %%

# Display a basic cross section of input data
gpv.plot_2d(geo_model_onlap, show_data=True)

# %%

# Compute a solution for the model
gp.compute_model(geo_model_onlap)

# %%

# Display the result in 2d section
gpv.plot_2d(geo_model_onlap, show_boundaries=False)

# %%

# Set the relation of the youngest group to Onlap
from gempy_engine.core.data.stack_relation_type import StackRelationType
geo_model_onlap.structural_frame.structural_groups[0].structural_relation = StackRelationType.ONLAP

# %%

# Display updated strucutral frame
geo_model_onlap.structural_frame

# %%

# Compute a solution for the model
gp.compute_model(geo_model_onlap)

# %%

# Display the result in 2d section
gpv.plot_2d(geo_model_onlap, show_boundaries=False)

# %%
# Video tutorial 12: Advanced onlap - Subduction zone
# """"""""""""""""""""""""""""

# %%
# .. raw:: html
#
#     <iframe width="560" height="315"
#             src="https://www.youtube.com/embed/R-vUld4V-OQ"
#             title="YouTube video player"
#             frameborder="0"
#             allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
#             allowfullscreen>
#     </iframe>
#
#

# %%

# Create instance of geomodel
geo_model_subduction = gp.create_geomodel(
    project_name = 'tutorial_model_onlap_2',
    extent=[0,2000,0,1000,0,1000],
    resolution=[100,50,50],
    importer_helper=gp.data.ImporterHelper(
        path_to_orientations=path_to_data+"tutorial_model_onlap_2_orientations.csv?",
        path_to_surface_points=path_to_data+"tutorial_model_onlap_2_surface_points.csv?"
    )
)

# %%

# Display a basic cross section of input data
gpv.plot_2d(geo_model_subduction)

# %%

# Map geological series to surfaces
gp.map_stack_to_surfaces(
    gempy_model=geo_model_subduction,
    mapping_object={
        "Top": ("continental_top"),
        "Continental_Series": ("continental_shallow", "continental_deep"),
        "Oceanic_Series": ("oceanic_top", "oceanic_bottom")
    }
)

# %%

# Set the realtion of the youngest and second youngest group to Onlap
geo_model_subduction.structural_frame.structural_groups[0].structural_relation = StackRelationType.ONLAP
geo_model_subduction.structural_frame.structural_groups[1].structural_relation = StackRelationType.ONLAP

# %%

# Display updated structural frame
geo_model_subduction.structural_frame

# %%

# Create a simple topography using numpy

# Define grid spacing
spacing = 20

# Generate grid
x = np.arange(geo_model_subduction.grid.regular_grid.extent[0], geo_model_subduction.grid.regular_grid.extent[1] + spacing, spacing)
y = np.arange(geo_model_subduction.grid.regular_grid.extent[2], geo_model_subduction.grid.regular_grid.extent[3] + spacing, spacing)
X, Y = np.meshgrid(x, y)

# Define elevation (z) based on x, creating a simple mountain range
Z = np.ones_like(X) * 590  # Default elevation
Z[(X >= 570) & (X < 1000)] = 590 + (200 * (X[(X >= 570) & (X < 1000)] - 600) / 400)
Z[(X >= 1000) & (X < 1300)] = 810 - (250 * (X[(X >= 1000) & (X < 1300)] - 1000) / 300)
Z[X >= 1300] = 540

# Flatten the data into (N,3) shape
topography_points = np.vstack((X.ravel(), Y.ravel(), Z.ravel())).T

# %%

# Set topography from numpy array
gp.set_topography_from_arrays(grid=geo_model_subduction.grid, xyz_vertices=topography_points)

# %%

# Compute a solution for the model
gp.compute_model(geo_model_subduction)

# %%

# Display the result in 2d section
gpv.plot_2d(geo_model_subduction, show_topography=True, show_boundaries=False)

# %%

# Display 3d plot of final model
gpv.plot_3d(geo_model_subduction, show_topography=True, image=True)

# sphinx_gallery_thumbnail_number = -1


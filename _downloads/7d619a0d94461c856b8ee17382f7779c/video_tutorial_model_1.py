"""
Video Tutorial "code-along": Modeling step by step
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
"""


# %%
# This tutorial demonstrates step-by-step geological modeling using the `gempy` and `gempy_viewer` libraries.
# It follows the Video tutorial series available on the `gempy YouTube channel <https://www.youtube.com/@GemPy3D>`_.


# %%
# Video tutorial 1: Introduction
# """"""""""""""""""""""""""""""
#
# The first video is an introduction to GemPy - please view online before starting the tutorial.
#
#

# %%
#.. raw:: html
#
#    <iframe width="560" height="315" 
#            src="https://www.youtube.com/embed/3elCCHhARsU" 
#            title="YouTube video player" 
#            frameborder="0" 
#            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
#            allowfullscreen>
#    </iframe>


# %%
# Video tutorial 2: Input data
# """"""""""""""""""""""""""""

# %%
#.. raw:: html
#
#    <iframe width="560" height="315" 
#            src="https://www.youtube.com/embed/yYwJVYy8ilw" 
#            title="YouTube video player" 
#            frameborder="0" 
#            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
#            allowfullscreen>
#    </iframe>



# Required imports
import gempy as gp
import gempy_viewer as gpv

# %%

# Path to input data
data_path = "https://raw.githubusercontent.com/cgre-aachen/gempy_data/master/"
path_to_data = data_path + "/data/input_data/video_tutorials_v3/"

# %%

# Create instance of geomodel
geo_model = gp.create_geomodel(
    project_name = 'tutorial_model',
    extent=[0,2500,0,1000,0,1000],
    resolution=[100,40,40],
    importer_helper=gp.data.ImporterHelper(
        path_to_orientations=path_to_data+"tutorial_model_orientations.csv",
        path_to_surface_points=path_to_data+"tutorial_model_surface_points.csv"
    )
)
# %%

# Display a basic cross section of input data
gpv.plot_2d(geo_model)

# %%

# Manually add a surface point
gp.add_surface_points(
    geo_model=geo_model,
    x=[2250],
    y=[500],
    z=[750],
    elements_names=['rock1']
)

# %%

# Show added point in cross section
gpv.plot_2d(geo_model)

# %%
# Video tutorial 3: Structural frame
# """"""""""""""""""""""""""""""""""

# %%
#.. raw:: html
#
#    <iframe width="560" height="315" 
#            src="https://www.youtube.com/embed/4rav3lb2AR8" 
#            title="YouTube video player" 
#            frameborder="0" 
#            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
#            allowfullscreen>
#    </iframe>

# View structural frame
geo_model.structural_frame

# %%

# View structural elements
geo_model.structural_frame.structural_elements

# %%

# Define structural groups and age/stratigraphic relationship
gp.map_stack_to_surfaces(
    gempy_model=geo_model,
    mapping_object={
        "Strat_Series2": ("rock3"),
        "Strat_Series1": ("rock2", "rock1")
    }
)

# %%
# Video tutorial 4: Computation and results
# """""""""""""""""""""""""""""""""""""""""

# %%
#.. raw:: html
#
#    <iframe width="560" height="315" 
#            src="https://www.youtube.com/embed/lrz1DeHrrBk" 
#            title="GemPy Tutorial Video 4" 
#            frameborder="0" 
#            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
#            allowfullscreen>
#    </iframe>


# View interpolation options
geo_model.interpolation_options

# %%

# Compute a solution for the model
gp.compute_model(geo_model)

# %%

# Display the result in 2d section
gpv.plot_2d(geo_model, cell_number=20)

# %%

# Some examples of how to access results
print(geo_model.solutions.raw_arrays.lith_block)
print(geo_model.grid.dense_grid.values)

# %%
# Video tutorial 5: 2D visualization
# """"""""""""""""""""""""""""""""""

# %%
#.. raw:: html
#
#    <iframe width="560" height="315" 
#            src="https://www.youtube.com/embed/nMms0JYn98o" 
#            title="GemPy tutorial video 5" 
#            frameborder="0" 
#            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
#            allowfullscreen>
#    </iframe>


# 2d plotting options
gpv.plot_2d(geo_model, show_value=True, show_lith=False, show_scalar=True, series_n=1, cell_number=25)

# %%

# Create custom section lines
gp.set_section_grid(
    grid=geo_model.grid,
    section_dict={
        'section1': ([0, 0], [2500, 1000], [100, 50]),
        'section2': ([1000, 1000], [1500, 0], [100, 100]),
    }
)

# %%

# Show custom cross-section traces
gpv.plot_section_traces(geo_model)

# %%

# Recompute model as a new grid was added
gp.compute_model(geo_model)

# %%

# Display custom cross-sections
gpv.plot_2d(geo_model, section_names=['section1', 'section2'], show_data=False)

# %%
# Video tutorial 6: 3D visualization
# """"""""""""""""""""""""""""""""""

# %%
#.. raw:: html
#
#    <iframe width="560" height="315" 
#            src="https://www.youtube.com/embed/MHEUQpVeSoQ" 
#            title="YouTube video player" 
#            frameborder="0" 
#            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
#            allowfullscreen>
#    </iframe>


# Display the result in 3d
gpv.plot_3d(geo_model, show_lith=True, show_boundaries=True, ve=None)

# %%

# How to access DC meshes
geo_model.solutions.dc_meshes[0].dc_data

# transform mesh vertices to original coordinate system

back_transformed_vertices = geo_model.input_transform.apply_inverse(geo_model.solutions.dc_meshes[0].vertices)

# %%
# Video tutorial 7: Topography
# """""""""""""""""""""""""""""""


# %%
#.. raw:: html
#
#    <iframe width="560" height="315" 
#            src="https://www.youtube.com/embed/FtAbUhwhiXE" 
#            title="YouTube video player" 
#            frameborder="0" 
#            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
#            allowfullscreen>
#    </iframe>


# Setting a randomly generated topography
import numpy as np

gp.set_topography_from_random(
    grid=geo_model.grid,
    fractal_dimension=1.2,
    d_z=np.array([700, 900]),
    topography_resolution=np.array([250, 100])
)

# %%

# Recompute model as a new grid was added
gp.compute_model(geo_model)

# %%

# Display a cross-section with topography
gpv.plot_2d(geo_model, show_topography=True)

# %%

# Displaying a geological map
gpv.plot_2d(geo_model, show_topography=True, section_names=['topography'], show_boundaries=False, show_data=False)

# %%

# Display the 3d model with topography
gpv.plot_3d(geo_model, show_lith=True, show_topography=True)

# sphinx_gallery_thumbnail_number = -1

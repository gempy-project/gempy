"""
1.3c: Adding topography to geological models
============================================

"""

# %%
import gempy as gp
import gempy_viewer as gpv
import numpy as np
import os

# %%
# 1. The common procedure to set up a model:
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 

# %%
data_path = os.path.abspath('../../')

geo_model: gp.data.GeoModel = gp.create_geomodel(
    project_name='Single_layer_topo',
    extent=[450000, 460000, 70000, 80000, -1000, 500],
    resolution=[50, 50, 50],
    refinement=4,
    importer_helper=gp.data.ImporterHelper(
        path_to_orientations=data_path + "/data/input_data/tut-ch1-7/onelayer_orient.csv",
        path_to_surface_points=data_path + "/data/input_data/tut-ch1-7/onelayer_interfaces.csv",
    )
)

# %% 

# %% 
gp.set_section_grid(
    grid=geo_model.grid,
    section_dict={
        'section1': ([450000, 75000], [460000, 75500], [100, 100]),
    }
)

# %%
# 2. Adding topography
# ~~~~~~~~~~~~~~~~~~~~
# 


# %%
# 2 a. Load from raster file
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
# 
#
# .. admonition:: Coming soon: Importing raster data
#
#     This feature is not yet available in the current version of GemPy. Probably will be moved to `subsurface` since
#     coupling it with the geological model does not add much value.
#
# %%

# This is to make it work in sphinx gallery
# cwd = os.getcwd()
# if not 'examples' in cwd:
#     path_dir = os.getcwd() + '/examples/tutorials/ch5_probabilistic_modeling'
# else:
#     path_dir = cwd
# 
# fp = path_dir + "/../../data/input_data/tut-ch1-7/bogota.tif"
# 
# # %% 
# geo_model.set_topography(source='gdal', filepath=fp)
# gp.plot_2d(geo_model, show_topography=True, section_names=['topography'], show_lith=False,
#            show_boundaries=False,
#            kwargs_topography={'cmap': 'gray', 'norm': None}
#            )
# plt.show()

# %%
# 2 b. create fun topography
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
# 


# %%
# If there is no topography file, but you think that your model with
# topography would look significantly cooler, you can use gempys
# :meth:`set_topography <gempy.core.model.ImplicitCoKriging.set_topography>` function
# to generate a random topography based on a fractal grid:
# 

# %%
# sphinx_gallery_thumbnail_number = 2

gp.set_topography_from_random(grid=geo_model.grid)
gpv.plot_2d(geo_model, show_topography=True, section_names=['topography'])

# %%
# It has additional keywords to play around with:
#
# * fd: fractal dimension:
#       defaults to 2.0. The higher (try 2.9), the rougher the landscape will
#       be.
#
# * d\_z: height difference:
#       If none, last 20% of the model in z
#       direction.
#
# * extent:
#       extent in xy direction. If none,
#       ``geo_model.grid.extent`` is used.
#
# * resolution:
#       resolution of the topography array.
#       If none, ``geo_model.grid.resoution`` is used. Increasing the resolution leads to
#       much nicer geological maps!
#
# 

# %% 
gp.set_topography_from_random(
    grid=geo_model.grid,
    fractal_dimension=1.9,
    d_z=np.array([0, 250]),
    topography_resolution=np.array([200, 200])
)

# %%
# Compute model
# ~~~~~~~~~~~~~
# 

# %% 
gp.compute_model(geo_model)

# %%
# Visualize:
# ^^^^^^^^^^
# 
# Now, the solutions object does also contain the computed geological map.
# It can be visualized using the 2D and 3D plotting functionality:
#

# %% 
gpv.plot_2d(geo_model, show_topography=True, section_names=['topography'], show_boundaries=False, show_data=True)

# %% 
gpv.plot_2d(geo_model, show_topography=True, section_names=['section1'])

# %%
g3d = gpv.plot_3d(
    model=geo_model,
    show_topography=True,
    show_lith=False,
    show_surfaces=False,
    show_results=False,
    ve=5
)

# %%
# sphinx_gallery_thumbnail_number = 3
g3d = gpv.plot_3d(
    model=geo_model,
    show_topography=True,
    show_lith=True,
    show_surfaces=True,
    ve=5
)

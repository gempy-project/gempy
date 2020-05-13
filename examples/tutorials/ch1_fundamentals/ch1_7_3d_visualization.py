"""
1.7: 3-D Visualization
======================

"""

# %% 
# Importing GemPy
import gempy as gp

# Importing auxiliary libraries
import numpy as np
import matplotlib.pyplot as plt

# %%
# Loading an example geomodel
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 

# %%

data_path = 'https://raw.githubusercontent.com/cgre-aachen/gempy_data/master/'

geo_model = gp.create_data('viz_3d',
                           [0, 2000, 0, 2000, 0, 1600],
                           [50, 50, 50],
                           path_o=data_path + "data/input_data/lisa_models/foliations" + str(7) + ".csv",
                           path_i=data_path + "data/input_data/lisa_models/interfaces" + str(7) + ".csv"
                           )

gp.map_stack_to_surfaces(
    geo_model,
    {"Fault_1": 'Fault_1', "Fault_2": 'Fault_2',
     "Strat_Series": ('Sandstone', 'Siltstone', 'Shale', 'Sandstone_2', 'Schist', 'Gneiss')}
)

geo_model.set_is_fault(['Fault_1', 'Fault_2'])
geo_model.set_topography()

gp.set_interpolator(geo_model)
gp.compute_model(geo_model, compute_mesh=True)

# %%
# Basic plotting API
# ------------------
# 


# %%
# Data plot
# ~~~~~~~~~
# 

# %% 
gp.plot_3d(geo_model, show_surfaces=False, show_data=True, show_lith=False, image=False)

# %%
# Geomodel plot
# ~~~~~~~~~~~~~
# 

# %% 
gp.plot_3d(geo_model, image=False)

# %%

# sphinx_gallery_thumbnail_number = 2
gpv = gp.plot.plot_3d(geo_model,
                      plotter_type='basic',off_screen=False,
                      show_topography=True,
                      show_scalar=False,
                      show_lith=True,
                      kwargs_plot_structured_grid={'opacity': .5})


# %%

#gpv = gp.plot_3d(geo_model, show_results=False, show_data=False, plotter_type='basic',
#                 show_scalar=True, scalar_field='Strat_Series')


# %%
# Granular 3-D Visualization
# --------------------------
# See notebook:




# %%
# Plotting all surfaces
# ~~~~~~~~~~~~~~~~~~~~~
# 
#
# # %%
# geo_model.surfaces
#
# %%
#gpv = gp.plot_3d(geo_model, show_results=False, show_data=False, plotter_type='basic',
#                 notebook=False, off_screen=True)

# %%
# gpv.plot_surfaces_all()
#gpv.plot_surfaces(['Siltstone', 'Gneiss'], clear=False)
# gpv.p.show()

# # %%
# # Plotting individual surfaces
# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# #
#
# # %%
# gpv = gp.plot_3d(geo_model, show_results=False, show_data=False, plotter_type='basic',
#                  off_screen=True)
# gpv.plot_surfaces(["Fault_1"], clear=False)
# gpv.plot_surfaces(["Shale"], clear=False)
# # gpv.p.show()
#
# # %%
# # Plotting input data
# # ~~~~~~~~~~~~~~~~~~~
# #
#
# # %%
# gpv = gp.plot_3d(geo_model, show_results=False, show_data=False, plotter_type='basic',
#                  off_screen=True)
# gpv.plot_surface_points()
# gpv.plot_orientations()
# # gpv.p.show()
#
# # %%
# gpv.surface_points_mesh
#
# # %%
# a = gpv.surface_points_mesh
#
# # %%
# a.points[:, -1]
#
# # %%
# a.n_arrays
#
#
# # %%
# # Plot structured grids
# # ~~~~~~~~~~~~~~~~~~~~~
# #
#
# # %%
# gpv = gp.plot_3d(geo_model, show_results=False, show_data=False, plotter_type='basic',
#                  off_screen=True)
# gpv.plot_structured_grid("scalar", series='Strat_Series')
# # gpv.p.show()

# %%
# Interactive Block with cross sections
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 

# %% 
#gp._plot.plot_interactive_3d(geo_model, name="lith", render_topography=False)

# # %%
# # Interactive Plotting
# # --------------------
# #
# # GemPy supports interactive plotting, meaning that you can drag & drop
# # the input data and GemPy will update the geomodel live. This does not
# # work in the static notebook plotter, but instead you have to pass the
# # ``notebook=False`` argument to open an interactive plotting window. When
# # running the next cell you can freely move the surface points (spheres)
# # and orientations (arrows) of the Shale horizon and see how it updates
# # the model.
# #
# # **Note**: Everytime you move a data point, GemPy will recompute the
# # geomodel. This works best whe running GemPy on a dedicated graphics card
# # (GPU).
# #
#
# # %%
# from importlib import reload
#
# reload(vista)
#
# # %%
# gpv = vista.Vista(geo_model, notebook=False)
#
# gpv.plot_surface("Shale")
# gpv.plot_surface("Fault_1", opacity=0.5)
# gpv.plot_surface("Fault_2", opacity=0.5)
#
# gpv.plot_surface_points_interactive("Shale")
# gpv.plot_surface_points_interactive("Fault_1")
# gpv.plot_surface_points_interactive("Fault_2")
#
# gpv.plot_orientations_interactive("Fault_2")
#
# gpv.show()
#
# # %%
# # For the entire geomodel:
# #
#
# # %%
# gpv = vista.Vista(geo_model, notebook=False)
#
# gpv.plot_surfaces()
# gpv.plot_surface_points_interactive_all()
#
# gpv.show()
#
# # %%

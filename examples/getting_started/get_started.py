"""
Getting Started
===============

"""

# %%

# Importing GemPy
import gempy as gp
import gempy_viewer as gpv

# Importing aux libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# %%
# Initializing the model:
# ~~~~~~~~~~~~~~~~~~~~~~~
# 
# The first step to create a GemPy model is create a gempy.Model object
# that will contain all the other data structures and necessary
# functionality.
# 
# In addition for this example we will define a regular grid since the
# beginning. This is the grid where we will interpolate the 3D geological
# model. GemPy comes with an array of different grids for different
# pourposes as we will see below. For visualization usually a regular grid
# is the one that makes more sense.
# 

# %% 
geo_model: gp.data.GeoModel = gp.create_geomodel(
    project_name='Model1',
    extent=[0, 791, -200, 200, -582, 0],
    resolution=[50, 50, 50],
    number_octree_levels=1,
    structural_frame=gp.data.StructuralFrame.initialize_default_structure()
    
)

print(geo_model)
# %%
# Creating figure:
# ~~~~~~~~~~~~~~~~
# 
# GemPy uses matplotlib and pyvista-vtk libraries for 2d and 3d
# visualization of the model respectively. One of the design decisions of
# GemPy is to allow real time construction of the model. What this means
# is that you can start adding input data and see in real time how the 3D
# surfaces evolve. Lets initialize the visualization windows.
# 
# The first one is the 2d figure. Just place the window where you can see
# it (maybe move the jupyter notebook to half screen and use the other
# half for the renderers).
# 

# %% 
# %matplotlib qt5
p2d = gpv.plot_2d(geo_model)

# %%
# Add model section
# ^^^^^^^^^^^^^^^^^
# 
# In the 2d renderer we can add several cross section of the model. In
# this case, for simplicity sake we are just adding one perpendicular to
# y.
# 


# %%
# Loading cross-section image:
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# 
# Remember that gempy is simply using matplotlib and therofe the ax object
# created above is a standard matplotlib axes. This allow to manipulate it
# freely. Lets load an image with the information of couple of boreholes
# 

# %% 
# Reading image
img = mpimg.imread('wells.png')
# Plotting it inplace
p2d = gpv.plot_2d(geo_model, show=False)
p2d.axes[0].imshow(img, origin='upper', alpha=.8, extent=(0, 791, -582, 0))
plt.show()

# %%
# We can do the same in 3D through pyvista and vtk rendering:
# 

# %% 
p3d = gpv.plot_3d(geo_model, image=True)

# %%
# Building the model
# ------------------
# 
# Now that we have everything initialize we can start the construction of
# the geological model.
# 
# Surfaces
# ~~~~~~~~
# 
# GemPy is a surface based interpolator. This means that all the input
# data we add has to be refered to a surface. The surfaces always mark the
# bottom of a unit. By default GemPy surfaces are empty:
# 

# %% 
geo_model.structural_frame.structural_elements

# %%
# Now we can start adding data. GemPy input data consist on surface points
# and orientations (perpendicular to the layers). The 2D plot gives you
# the X and Z coordinates when hovering the mouse over. We can add a
# surface point as follows:
# 

# %% 
# Add a point
# geo_model.add_surface_points(X=223, Y=0.01, Z=-94, surface='surface1')
gp.add_surface_points(
    geo_model=geo_model,
    x=[223],
    y=[0.01],
    z=[-94],
    elements_names=['surface1']
)

# Plot in 2D
gpv.plot_2d(geo_model, cell_number=11)

# Plot in 3D
gpv.plot_3d(geo_model, image=True)

# %%
# Now we can add the other two points of the layer:
# 

# %% 
# Add points
# geo_model.add_surface_points(X=458, Y=0, Z=-107, surface='surface1')
# geo_model.add_surface_points(X=612, Y=0, Z=-14, surface='surface1')
gp.add_surface_points(
    geo_model=geo_model,
    x=[458, 612],
    y=[0, 0],
    z=[-107, -14],
    elements_names=['surface1', 'surface1']
)

# Plotting
gpv.plot_2d(geo_model, cell_number=11)
gpv.plot_3d(geo_model, image=True)

# %%
# The minimum amount of data to interpolate anything in gempy is: a) 2
# surface points per surface b) One orientation per series.
# 
# Lets add an orientation anywhere in space:
# 

# %% 
# Adding orientation
# geo_model.add_orientations(X=350, Y=0, Z=-300, surface='surface1', pole_vector=(0, 0, 1))
gp.add_orientations(
    geo_model=geo_model,
    x=[350],
    y=[1],
    z=[-300],
    elements_names=['surface1'],
    pole_vector=[[0, 0, 1]]
)

gpv.plot_2d(geo_model, cell_number=5)
gpv.plot_3d(geo_model, image=True)

# %%
# Recompute transform
geo_model.update_transform(gp.data.GlobalAnisotropy.NONE)  # * Remove the auto anisotropy for this 2.5D model

# %%
# Now we have enough data for finally interpolate!
# 

# %% 
geo_model.interpolation_options.dual_contouring = False
gp.compute_model(geo_model, engine_config=gp.data.GemPyEngineConfig())

# %% 
geo_model.interpolation_options.kernel_options


# %%
# That is, we have interpolated the 3D surface. We can visualize:
# 

# %% 
# In 2D
gpv.plot_2d(geo_model, cell_number=[5])

# In 3D
gpv.plot_3d(geo_model, show_surfaces=True, image=True)

# %%
# Adding more layers:
# ~~~~~~~~~~~~~~~~~~~
# 
# So far we only need 2 units defined. The cross-section image that we
# load have 4 however. Lets add two layers more:
# 

# %% 
geo_model.structural_frame


# %% 
# geo_model.add_surfaces(['surface3', 'basement'])


# %%
# Layer 2
# ~~~~~~~
# 
# Add the layer next layers:
# 

# %% 
# Your code here:
element2 = gp.data.StructuralElement(
    name='surface2',
    color=next(geo_model.structural_frame.color_generator),
    surface_points=gp.data.SurfacePointsTable.from_arrays(
        x=np.array([225, 459]),
        y=np.array([0, 0]),
        z=np.array([-269, -279]),
        names='surface2'
    ),
    orientations=gp.data.OrientationsTable.initialize_empty()
)

geo_model.structural_frame.structural_groups[0].append_element(element2)

# --------------------

# %% 
# Compute model
gp.compute_model(geo_model)

# %% 
gpv.plot_2d(geo_model, cell_number=5, legend='force')
gpv.plot_3d(geo_model, image=True)

# %%
# Layer 3
# ~~~~~~~
# 

# %% 
# Your code here:

# geo_model.add_surface_points(X=225, Y=0, Z=-439, surface='surface3')
# geo_model.add_surface_points(X=464, Y=0, Z=-456, surface='surface3')
# geo_model.add_surface_points(X=619, Y=0, Z=-433, surface='surface3')

element3 = gp.data.StructuralElement(
    name='surface3',
    color=next(geo_model.structural_frame.color_generator),
    surface_points=gp.data.SurfacePointsTable.from_arrays(
        x=np.array([225, 464, 619]),
        y=np.array([0, 0, 0]),
        z=np.array([-439, -456, -433]),
        names='surface3'
    ),
    orientations=gp.data.OrientationsTable.initialize_empty()
)   

geo_model.structural_frame.structural_groups[0].append_element(element3)

# ------------------

# %% 
# Computing and plotting 3D
gp.compute_model(geo_model)

gpv.plot_2d(geo_model, cell_number=5, legend='force')
gpv.plot_3d(geo_model, kwargs_plot_structured_grid={'opacity': .2})

# %%
# Adding a Fault
# ~~~~~~~~~~~~~~
# 
# So far the model is simply a depositional unit. GemPy allows for
# unconformities and faults to build complex models. This input is given
# by categorical data. In general:
# 
# input data (surface points/ orientations) <belong to< surface <belong
# to< series
# 
# And series can be a fault—i.e. offset the rest of surface— or not. We
# are going to show how to add a fault as an example.
# 
# First we need to add a series:
# 

# %% 
geo_model.add_features('Fault1')

# %% 
geo_model.reorder_features(['Fault1', 'Default series'])

# %%
# Then define that is a fault:
# 

# %% 
geo_model.set_is_fault('Fault1')

# %%
# But we also need to add a new surface:
# 

# %% 
geo_model.add_surfaces('fault1')

# %%
# And finally assign the new surface to the new series/fault
# 

# %% 
gp.map_stack_to_surfaces(geo_model, {'Fault1': 'fault1'})

# %%
# Now we can just add input data as before (remember the minimum amount of
# input data to compute a model):
# 

# %% 
# Add input data of the fault
geo_model.add_surface_points(X=550, Y=0, Z=-30, surface='fault1')
geo_model.add_surface_points(X=650, Y=0, Z=-200, surface='fault1')
geo_model.add_orientations(X=600, Y=0, Z=-100, surface='fault1', pole_vector=(.3, 0, .3))

# Plotting Inpute data
gp.plot_2d(geo_model, show_solutions=False)

# %%
# And now is computing as before:
# 

# %% 
# Compute
gp.compute_model(geo_model)

# Plot
gp.plot_2d(geo_model, cell_number=5, legend='force')
gp.plot_3d(geo_model, kwargs_plot_structured_grid={'opacity': .2})

# %%
# As you can see now instead of having folding layers we have a sharp
# jump. Building on this you can pretty much any model you can imagine.
# 


# %%
# Additional features:
# ====================
# 
# Over the years we have built a bunch of assets integrate with gempy.
# Here we will show some of them:
# 
# Topography
# ~~~~~~~~~~
# 
# GemPy has a built-in capabilities to read and manipulate topographic
# data (through gdal). To show an example we can just create a random
# topography:
# 

# %% 
# Adding random topography
geo_model.set_topography(source='random', fd=1.9, d_z=np.array([-150, 0]),
                         resolution=np.array([200, 200]))

# %%
# The topography can we visualize in both renderers:
# 

# %% 
gp.plot_2d(geo_model, cell_number=5, legend='force')
gp.plot_3d(geo_model, kwargs_plot_structured_grid={'opacity':.2})

# %%
# But also allows us to compute the geological map of an area:
# 

# %% 
gp.compute_model(geo_model)

# sphinx_gallery_thumbnail_number = 16
gp.plot_3d(geo_model, show_topography=True)

# %%
# Gravity inversion
# ~~~~~~~~~~~~~~~~~
# 
# GemPy also allows for inversions (in production only gravity so far). We
# can see a small demo how this works.
# 
# The first thing to do is to assign densities to each of the units:
# 

# %% 
geo_model.add_surface_values([0, 2.6, 2.4, 3.2, 3.6], ['density'])

# %%
# Also we can create a centered grid around a device for precision:
# 

# %% 
geo_model.set_centered_grid(centers=[[400, 0, 0]], resolution=[10, 10, 100], radius=800)

# %%
# We need to modify the compile code:
# 

# %% 
gp.set_interpolator(geo_model, output=['gravity'], aesara_optimizer='fast_run')

# %%
# But now additionally to the interpolation we also compute the forward
# gravity of the model (at the point XYZ = 400, 0, 0)
# 

# %% 
gp.compute_model(geo_model)
geo_model.solutions.fw_gravity

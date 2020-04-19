# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: sphinx
#       format_version: '1.1'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

"""
## Importing used libraries
"""

# These two lines are necessary only if GemPy is not installed
import sys, os
import numpy as np
sys.path = list(np.insert(sys.path, 0, "../.."))

# Importing GemPy
import gempy as gp

# Importing aux libraries
from ipywidgets import interact
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Embedding matplotlib figures in the notebooks
# %matplotlib qt5

""
from gempy.plot import visualization_2d_pro as vv
from gempy.plot import vista

###############################################################################
# ### Initializing the model:
#
# The first step to create a GemPy model is create a gempy.Model object that will contain all the other data structures and necessary functionality.
#
# In addition for this example we will define a regular grid since the beginning. This is the grid where we will interpolate the 3D geological model. GemPy comes with an array of different grids for different pourposes as we will see below. For visualization usually a regular grid is the one that makes more sense.

geo_model = gp.create_model('Model1')
geo_model = gp.init_data(geo_model, extent= [0, 791, 0, 200, -582, 0], resolution=[100, 10, 100])

###############################################################################
# GemPy core code is written in Python. However for efficiency (and other reasons) most of heavy computations happend in optimize compile code, either C or CUDA for GPU. To do so, GemPy rely on the library theano. To guarantee maximum optimization theano requires to compile the code for every Python kernel. The compilation is done by calling the following line at any point (before computing the model):

gp.set_interpolator(geo_model, theano_optimizer='fast_compile', verbose=[])

###############################################################################
# ### Creating figure:
#
# GemPy uses matplotlib and pyvista-vtk libraries for 2d and 3d visualization of the model respectively. One of the design decisions of GemPy is to allow real time construction of the model. What this means is that you can start adding input data and see in real time how the 3D surfaces evolve. Lets initialize the visualization windows.
#
# The first one is the 2d figure. Just place the window where you can see it (maybe move the jupyter notebook to half screen and use the other half for the renderers).

# %matplotlib qt5
import pyvista as pv
pv.set_plot_theme("document")

p2d = vv.Plot2D(geo_model)
p2d.create_figure((13, 6))

###############################################################################
# #### Add model section
#
# In the 2d renderer we can add several cross section of the model. In this case, for simplicity sake we are just adding one perpendicular to y.

# In this case perpendicular to the y axes
ax = p2d.add_section(cell_number=1, direction='y')

###############################################################################
# #### Loading cross-section image:
#
# Remember that gempy is simply using matplotlib and therofe the ax object created above is a standard matplotlib axes. This allow to manipulate it freely. Lets load an image with the information of couple of boreholes

# Reading image
img = mpimg.imread('Picture 1.png')
# Plotting it inplace
ax.imshow(img, origin='upper', alpha=.8, extent = (0, 791, -582,0))

###############################################################################
# We can do the same in 3D through pyvista and vtk rendering. Click the qt5 button Back (+Y) to have the same view as in the 2D viwer:

p3d = vista.Vista(geo_model, plotter_type='background', notebook=False, real_time=False)

###############################################################################
# ## Building the model
#
# Now that we have everything initialize we can start the construction of the geological model. 
#
# ### Surfaces
#
# GemPy is a surface based interpolator. This means that all the input data we add has to be refered to a surface. The surfaces always mark the bottom of a unit. By default GemPy surfaces are empty:

geo_model.surfaces

###############################################################################
# If we do not care about the names and we just want to interpolate a surface we can use:

# Default surfaces:
geo_model.set_default_surfaces()

###############################################################################
# Now we can start adding data. GemPy input data consist on surface points and orientations (perpendicular to the layers). The 2D plot gives you the X and Z coordinates when hovering the mouse over. We can add a surface point as follows:

# Add a point
geo_model.add_surface_points(X=223, Y=0.01, Z=-94, surface='surface1')

# Plot in 2D
p2d.plot_data(ax, cell_number=11)

# Plot in 3D
p3d.plot_surface_points()

###############################################################################
# Now we can add the other two points of the layer:

# Add points
geo_model.add_surface_points(X=458, Y=0, Z=-107, surface='surface1')
geo_model.add_surface_points(X=612, Y=0, Z=-14, surface='surface1')

# Plotting
p2d.plot_data(ax, cell_number=11)
p3d.plot_surface_points()

###############################################################################
# The minimum amount of data to interpolate anything in gempy is:
# a) 2 surface points per surface
# b) One orientation per series.
#
# Lets add an orientation anywhere in space:

# Adding orientation
geo_model.add_orientations(X=350, Y=0, Z=-300, surface='surface1', pole_vector= (0,0,1))
p2d.plot_data(ax, cell_number=5)
p3d.plot_data()

###############################################################################
# Now we have enough data for finally interpolate!

gp.compute_model(geo_model)

""
geo_model.interpolator.theano_graph.a_T.get_value()

""
geo_model.additional_data.kriging_data

""
geo_model.additional_data.rescaling_data

###############################################################################
# That is, we have interpolated the 3D surface. We can visualize with:

# In 2D
p2d.plot_contacts(ax, cell_number=5)

# In 3D
p3d.plot_surfaces()

###############################################################################
# ## Adding more layers:
#
# So far we only need 2 units defined. The cross-section image that we load have 4 however. Lets add two layers more:

geo_model.surfaces

""
geo_model.series

""
geo_model.add_surfaces(['surface3', 'basement'])

###############################################################################
# ### Layer 2
#
# Add the layer next layers:

# Your code here:
geo_model.add_surface_points(X=225, Y=0, Z=-269, surface='surface2')
geo_model.add_surface_points(X=459, Y=0, Z=-279, surface='surface2')


#--------------------
# Plot data
p2d.remove(ax)
p2d.plot_data(ax, cell_number=5, legend='force')
p3d.plot_data()

""
# Compute model
gp.compute_model(geo_model)

""
geo_model.interpolator.theano_graph.c_o_T.get_value()

""
geo_model.additional_data.kriging_data

""
# Plot 2D
p2d.plot_lith(ax, cell_number=5)
p2d.plot_contacts(ax, cell_number=5)

# Plot 3D
p3d.plot_surfaces()
p3d.plot_structured_grid(opacity=.2, annotations = {1: 'surface1', 2:'surface2', 3:'surface3'})


###############################################################################
# ### Layer 3

# Your code here:
geo_model.add_surface_points(X=225, Y=0, Z=-439, surface='surface3')
geo_model.add_surface_points(X=464, Y=0, Z=-456, surface='surface3')
geo_model.add_surface_points(X=619, Y=0, Z=-433, surface='surface3')

# ------------------
# Plotting data
p2d.remove(ax)
p2d.plot_data(ax, cell_number=5)
p3d.plot_data()

""
# Computing and plotting 3D
gp.compute_model(geo_model)
p2d.plot_lith(ax, cell_number=5)
p2d.plot_contacts(ax, cell_number=5)

p3d.plot_surfaces()
p3d.plot_structured_grid(opacity=.2, annotations = {1: 'surface1', 2:'surface2', 3:'surface3', 4:'basement'})


""
geo_model.additional_data

""
geo_model.interpolator.len_series_u

###############################################################################
# ## Faults:
#
# So far the model is simply a depositional unit. GemPy allows for unconformities and faults to build complex models. This input is given by categorical data. In general:
#
# input data (surface points/ orientations) <belong to< surface <belong to< series
#
# And series can be a fault---i.e. offset the rest of surface--- or not. We are going to show how to add a fault as an example.
#
# First we need to add a series:

geo_model.add_series('Fault1')

""
geo_model.reorder_series(['Fault1', 'Default series'])

###############################################################################
# Then define that is a fault:

geo_model.set_is_fault('Fault1')

###############################################################################
# But we also need to add a new surface:

geo_model.add_surfaces('fault1')

###############################################################################
# And finally assign the new surface to the new series/fault

gp.map_series_to_surfaces(geo_model, {'Fault1':'fault1'})

###############################################################################
# Now we can just add input data as before (remember the minimum amount of input data to compute a model):

# Add input data of the fault
geo_model.add_surface_points(X=550, Y=0, Z=-30, surface='fault1')
geo_model.add_surface_points(X=650, Y=0, Z=-200, surface='fault1')
geo_model.add_orientations(X=600, Y=0, Z= -100, surface='fault1', pole_vector=(.3,0,.3))

# Plotting Inpute data
p2d.remove(ax)
p2d.plot_data(ax, cell_number=5)
p3d.plot_data()

###############################################################################
# And now is computing as before:

# Compute
gp.compute_model(geo_model)

# Plot
p2d.plot_lith(ax, cell_number=5)
p2d.plot_contacts(ax, cell_number=5)
p3d.plot_surfaces()
p3d.plot_structured_grid(opacity=.2, annotations = {2: 'surface1', 3:'surface2', 4:'surface3', 5:'basement'})

###############################################################################
# As you can see now instead of having folding layers we have a sharp jump. Building on this you can pretty much any model you can imagine.

###############################################################################
# # Additional features:
#
# Over the years we have built a bunch of assets integrate with gempy. Here we will show some of them:
#
# ### Topogragpy
#
# GemPy has a built-in capanilities to read and manipulate topographic data (through gdal). To show an example we can just create a random topography:

## Adding random topography
geo_model.set_topography(source='random',fd=1.9, d_z=np.array([-150, 0]), resolution=np.array([200,200]))

###############################################################################
# The topography can we visualize in both renderers:

p2d.plot_topography(ax, cell_number=5)
p3d.plot_topography(scalars='topography')

###############################################################################
# But also allows us to compute the geological map of an area:

gp.compute_model(geo_model)
p3d.plot_topography(scalars='geo_map')

###############################################################################
# ### Gravity inversion
#
# GemPy also allows for inversions (in production only gravity so far). We can see a small demo how this works.
#
# The first thing to do is to assign densities to each of the units:

geo_model.surfaces.add_surfaces_values([0, 2.6, 2.4, 3.2, 3.6], ['density'])

###############################################################################
# Also we can create a centered grid around a device for precision:

geo_model.set_centered_grid(centers= [[400, 0, 0]],  resolution = [10, 10, 100], radio=800)

###############################################################################
# We need to modify the compile code:

gp.set_interpolator(geo_model, output=['gravity'], theano_optimizer='fast_run', )

###############################################################################
# But now additionally to the interpolation we also compute the forward gravity of the model (at the point XYZ = 400, 0, 0)

gp.compute_model(geo_model)
geo_model.solutions.fw_gravity

###############################################################################
# We can visualize it it better in the following figure. The aim of an inversion is to find the set of parameters that fit a measured point the better. In this example the red x symbolize the measured gravity while the blue dots are the current gravity (and previous) fw gravity values. The widget moves up and down the surface 3

# Creating new figure
p_grav = vv.Plot2D(geo_model)
p_grav.create_figure((6, 9))

# Adding section for model
ax2 = p_grav.add_section(cell_number=1, direction='y', ax_pos=211)

# Adding section for gravity 
ax3 = p_grav.add_section(ax_pos=414)

# Plotting model in section
p2d.plot_data(ax2, cell_number=5, legend='force')
p2d.plot_lith(ax2, cell_number=5)
p2d.plot_contacts(ax2, cell_number=5)
ax2.plot(400, 0, '^', markersize=40, c='red')

# Plotting initial values of the gravity axes
target_grav = -810
ax3.tick_params( bottom=False)
ax3.spines['top'].set_visible(False)
ax3.spines['bottom'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.plot(0, target_grav, 'X', label='Target Gravity', markersize=4, c='red')
ax3.plot(1, geo_model.solutions.fw_gravity, 'o', label='Current Gravity', markersize=4, c='blue')
ax3.set_ylabel('grav')

# We store the original values of z for the surface 3
Z_ori = geo_model.surface_points.df.loc[[5,6,7], 'Z'].copy()
# init a list to store grav
grav_ = []

# Function that modify the model, compute it and plot
def gravity_invert(dz):
    new_z = Z_ori + dz
    geo_model.modify_surface_points(indices=[5, 6, 7], Z= new_z)
    gp.compute_model(geo_model)
    grav_.append(geo_model.solutions.fw_gravity[0])
    
    p2d.remove(ax2)
    p2d.plot_data(ax2, cell_number=5)
    p2d.plot_lith(ax2, cell_number=5)
    p2d.plot_contacts(ax2, cell_number=5)
    ax3.plot(np.arange(len(grav_))+1, grav_, 'o', label='Current Gravity', markersize=4, c='blue')
    ax3.set_xlim(-1, len(grav_)+1)
    
interact(gravity_invert, dz=(-200, 200, 10))

###############################################################################
# ## More advanced model:

geo_model = gp.create_model('Moureze')
geo_model = gp.init_data(geo_model, extent= [0, 3630, -0.5, 0.5, -1084, 0], resolution=[321, 21, 91])

""
img = mpimg.imread('model2.png')

""
p2d = vv.Plot2D(geo_model)
p2d.create_figure((13, 6))
ax4 = p2d.add_section(cell_number=11)

""
ax4.imshow(img, origin='upper', alpha=.8, extent = (0, 3630, -1084,0))

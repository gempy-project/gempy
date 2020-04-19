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
## Chapter 2.1 Forward Gravity: Simple example
"""

# These two lines are necessary only if gempy is not installed
import sys, os
sys.path.append("../../..")

# Importing gempy
import gempy as gp

# Embedding matplotlib figures into the notebooks
# %matplotlib inline


# Aux imports
import numpy as np
import pandas as pn
import matplotlib.pyplot as plt
import theano
import qgrid

""
geo_model = gp.load_model('Greenstone', path= '../../data/gempy_models')

""
geo_model.series

""
geo_model.surfaces

""
gp.plot.plot_data(geo_model)

""
# Compute normal model:
# gp.set_interpolation_data(geo_model,
#                           compile_theano=True,
#                           theano_optimizer='fast_compile',
#                           verbose=[])
# gp.compute_model(geo_model)

###############################################################################
# ### Creating grid

###############################################################################
# First we need to define the location of the devices. For this example we can make a map:

grav_res = 20
X = np.linspace(7.050000e+05, 747000, grav_res)
Y = np.linspace(6863000, 6925000, grav_res)
Z= 300
xyz= np.meshgrid(X, Y, Z)
xy_ravel = np.vstack(list(map(np.ravel, xyz))).T
xy_ravel

###############################################################################
# We can see the location of the devices relative to the model data:

import matplotlib.pyplot as plt
gp.plot.plot_data(geo_model, direction='z')
plt.scatter(xy_ravel[:,0], xy_ravel[:, 1], s=1)

###############################################################################
# Now we need to create the grid centered on the devices (see: https://github.com/cgre-aachen/gempy/blob/master/notebooks/tutorials/ch1-3-Grids.ipynb)

geo_model.set_centered_grid(xy_ravel,  resolution = [10, 10, 15], radius=5000)

""
geo_model.grid.centered_grid.kernel_centers

###############################################################################
# Now we need to compute the component tz (see https://github.com/cgre-achen/gempy/blob/master/notebooks/tutorials/ch2-2-Cell_selection.ipynb)

from gempy.assets.geophysics import GravityPreprocessing

""
g = GravityPreprocessing(geo_model.grid.centered_grid)

""
tz = g.set_tz_kernel()

""
tz

###############################################################################
# ### Compiling the gravity graph
#
# If geo_model has already a centered grid, the calculation of tz happens automatically. Alternatively you pass tz to `set_gravity interpolator`. This theano graph will return gravity instead the lithologies. In addition we need either to pass the density block (see below). Or the position of density on the surface(in the future the name) to compute the density block at running time.

geo_model.surfaces

###############################################################################
# In this case the densities of each layer are at the loc 1 (0 is the id)

# Old way
#geo_model.set_gravity_interpolator(pos_density=1, verbose=['grid_shape', 'slices'])

# New way
gp.set_interpolator(geo_model, output=['gravity'], pos_density=1,  gradient=False,
                    theano_optimizer='fast_run')  

###############################################################################
# Once we have created a gravity interpolator we can call it from compute model as follows:

sol = gp.compute_model(geo_model, output=['geology'])
grav = sol.fw_gravity

""
gp.plot.plot_data(geo_model, direction='z', height=7)
plt.scatter(xy_ravel[:,0], xy_ravel[:, 1], s=1)
plt.imshow(sol.fw_gravity.reshape(grav_res, grav_res), extent = (xy_ravel[:,0].min() + (xy_ravel[0, 0] - xy_ravel[1, 0])/2,
                                                       xy_ravel[:,0].max() - (xy_ravel[0, 0] - xy_ravel[1, 0])/2,
                                                       xy_ravel[:,1].min() + (xy_ravel[0, 1] - xy_ravel[30, 1])/2,
                                                       xy_ravel[:,1].max() - (xy_ravel[0, 1] - xy_ravel[30, 1])/2),
           cmap='viridis_r', origin='bottom')

###############################################################################
# #### Plotting lithologies
#
# If we want to compute the lithologies we will need to create a normal interpolator object as seen in the Chapter 1 of the tutorials  

###############################################################################
# Now we can plot all together (change the alpha parameter to see the gravity overlying):

gp.plot.plot_section(geo_model, -1, direction='z')
plt.scatter(xy_ravel[:,0], xy_ravel[:, 1], s=1)
plt.imshow(grav.reshape(grav_res, grav_res), extent = (xy_ravel[:,0].min() + (xy_ravel[0, 0] - xy_ravel[1, 0])/2,
                                                       xy_ravel[:,0].max() - (xy_ravel[0, 0] - xy_ravel[1, 0])/2,
                                                       xy_ravel[:,1].min() + (xy_ravel[0, 1] - xy_ravel[30, 1])/2,
                                                       xy_ravel[:,1].max() - (xy_ravel[0, 1] - xy_ravel[30, 1])/2),
           cmap='viridis_r', origin='bottom', alpha=.0)

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
# GemPy Paper Code: Compute Forward Gravity

In this notebook you will be able to see and run the code utilized to create the figures of the paper *GemPy - an open-source library for implicit geological modeling and uncertainty quantification*
"""

# Importing dependencies

# These two lines are necessary only if gempy is not installed
import sys, os
sys.path.append("../..")

import gempy as gp
# %matplotlib inline

# Aux imports

import numpy as np
import pandas as pn
import matplotlib.pyplot as plt

###############################################################################
# ## Gravity
#
# For the gravity we need to increment the extent of the model to extrapolate enough voxels around the points where we simulate the gravity response to avoid (or at least reduce) boundaries error. Other than that the model and the code is the same as in the previous notebook

geo_model = gp.load_model('GemPy-Paper-1')

""
gp.plot.plot_data(geo_model)

###############################################################################
# We extend everything to km to get a more realistic example

new_extent = geo_model.grid.regular_grid.extent*1000

""
geo_model.surface_points.df[['X', 'Y', 'Z']] *= 1000
geo_model.orientations.df[['X', 'Y', 'Z']] *= 1000


""
geo_model.grid.set_regular_grid(resolution=[10,10,10],
                                extent= new_extent)
geo_model.rescaling.set_rescaled_grid()

""
gp.plot.plot_data(geo_model)

###############################################################################
# ## Creating centered grid:
#
# First we need to define the location of the devices. For this example we can make a map:

grav_res = 20
X = np.linspace(0.1e3, 19.9e3, grav_res)
Y = np.linspace(.1e3,.9e3, grav_res)
Z= 0
xyz= np.meshgrid(X, Y, Z)
xy_ravel = np.vstack(list(map(np.ravel, xyz))).T

""
gp.plot.plot_data(geo_model, direction='z')
plt.scatter(xy_ravel[:, 0], xy_ravel[:, 1])

""
# Theano compilation
interp_data_g = gp.InterpolatorData(geo_data_g, u_grade=[1, 1, 1], dtype='float64', verbose=[],  output='gravity', compile_theano=True)

""
# Set the specific parameters for the measurement grid of gravity:
gp.set_geophysics_obj(interp_data_g,  
                      [0.1e3,19.9e3,.1e3,.9e3, -10e3, 0], # Extent
                      [30,20])                            # Resolution 

""
# Setting desity and precomputations 
t = gp.precomputations_gravity(interp_data_g, 25,
                         [2.92e6, 3.1e6, 2.61e6, 2.92e6])

""
lith, fault, grav = gp.compute_model(interp_data_g, output='gravity')

""
gp.plot_section(geo_data_g, lith[0], 5, direction='z',plot_data=True)
#annotate_plot(gp.get_data(geo_data_g, verbosity=2), 'annotations', 'X', 'Z', size = 20)
# ax = plt.gca()
# ax.set_xticks(np.linspace(0, 20, 50))
# ax.set_yticks(np.linspace(0, -10, 50))
plt.grid()
fig = plt.gcf()
ax = plt.gca()
p = ax.imshow(grav.reshape(20,30), cmap='viridis', origin='lower', alpha=0.8, extent=[0,20e3,0,10e3])
# plt.xlim(-2e3,22e3)
# plt.ylim(-2e3,12e3)

plt.xlim(-10e3,30e3)
plt.ylim(-10e3,20e3)

plt.colorbar(p, orientation='horizontal')
#plt.show()
# fig.savefig("doc/figs/gravity.png")

""
plt.imshow(grav.reshape(20,30), cmap='viridis', origin='lower', extent=[5,15,3,7])

""
gp.plot_section(geo_data_g, lith[0], 25, direction='y',plot_data=True)


""


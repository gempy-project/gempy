"""
2.1 Forward Gravity: Simple example
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

"""

# %%
# Importing gempy
import gempy as gp
from gempy.assets.geophysics import GravityPreprocessing

# Aux imports
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

np.random.seed(1515)
pd.set_option('precision', 2)

# %%
cwd = os.getcwd()
if 'examples' not in cwd:
    data_path = os.getcwd() + '/examples/'
else:
    data_path = cwd + '/../../'

geo_model = gp.load_model('Greenstone', path=data_path + 'data/gempy_models/Greenstone')

# %% 
geo_model.stack

# %% 
geo_model.surfaces

# %% 
gp.plot_2d(geo_model)

# %%

# %%
# Creating grid
# ~~~~~~~~~~~~~
# 

# %%
# First we need to define the location of the devices. For this example we
# can make a map:
# 

# %% 
grav_res = 20
X = np.linspace(7.050000e+05, 747000, grav_res)
Y = np.linspace(6863000, 6925000, grav_res)
Z = 300
xyz = np.meshgrid(X, Y, Z)
xy_ravel = np.vstack(list(map(np.ravel, xyz))).T
xy_ravel

# %%
# We can see the location of the devices relative to the model data:
# 

# %%

gp.plot_2d(geo_model, direction='z', show=False)
plt.scatter(xy_ravel[:, 0], xy_ravel[:, 1], s=1)
plt.show()

# %%
# Now we need to create the grid centered on the devices (see:
# https://github.com/cgre-aachen/gempy/blob/master/notebooks/tutorials/ch1-3-Grids.ipynb)
# 

# %% 
geo_model.set_centered_grid(xy_ravel, resolution=[10, 10, 15], radius=5000)

# %% 
geo_model.grid.centered_grid.kernel_centers

# %%
# Now we need to compute the component tz (see
# https://github.com/cgre-achen/gempy/blob/master/notebooks/tutorials/ch2-2-Cell_selection.ipynb)
# 

# %% 
g = GravityPreprocessing(geo_model.grid.centered_grid)

# %% 
tz = g.set_tz_kernel()

# %% 
tz

# %%
# Compiling the gravity graph
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# If geo_model has already a centered grid, the calculation of tz happens
# automatically.  This theano graph will return gravity
# as well as the lithologies. In addition we need either to pass the density
# block (see below). Or the position of density on the surface(in the
# future the name) to compute the density block at running time.
# 

# %% 
geo_model.surfaces

# %%
# In this case the densities of each layer are at the loc 1 (0 is the id)
# 

# New way
gp.set_interpolator(geo_model, output=['gravity'], pos_density=1, gradient=False,
                    theano_optimizer='fast_run')

# %%
# Once we have created a gravity interpolator we can call it from compute
# model as follows:
# 

# %% 
sol = gp.compute_model(geo_model)
grav = sol.fw_gravity

# %% 
gp.plot_2d(geo_model, direction=['z'], height=7, show_results=False, show_data=True,
           show=False)
plt.scatter(xy_ravel[:, 0], xy_ravel[:, 1], s=1)
plt.imshow(sol.fw_gravity.reshape(grav_res, grav_res),
           extent=(xy_ravel[:, 0].min() + (xy_ravel[0, 0] - xy_ravel[1, 0]) / 2,
                   xy_ravel[:, 0].max() - (xy_ravel[0, 0] - xy_ravel[1, 0]) / 2,
                   xy_ravel[:, 1].min() + (xy_ravel[0, 1] - xy_ravel[30, 1]) / 2,
                   xy_ravel[:, 1].max() - (xy_ravel[0, 1] - xy_ravel[30, 1]) / 2),
           cmap='viridis_r', origin='lower')
plt.show()
# %%
# Plotting lithologies
# ^^^^^^^^^^^^^^^^^^^^
# 
# If we want to compute the lithologies we will need to create a normal
# interpolator object as seen in the Chapter 1 of the tutorials
# 


# %%
# Now we can plot all together (change the alpha parameter to see the
# gravity overlying):
# 

# %%
# sphinx_gallery_thumbnail_number = 4
gp.plot_2d(geo_model, cell_number=[-1], direction=['z'], show=False,
           kwargs_regular_grid={'alpha': .5})

plt.scatter(xy_ravel[:, 0], xy_ravel[:, 1], s=1)
plt.imshow(grav.reshape(grav_res, grav_res),
           extent=(xy_ravel[:, 0].min() + (xy_ravel[0, 0] - xy_ravel[1, 0]) / 2,
                   xy_ravel[:, 0].max() - (xy_ravel[0, 0] - xy_ravel[1, 0]) / 2,
                   xy_ravel[:, 1].min() + (xy_ravel[0, 1] - xy_ravel[30, 1]) / 2,
                   xy_ravel[:, 1].max() - (xy_ravel[0, 1] - xy_ravel[30, 1]) / 2),
           cmap='viridis_r', origin='lower', alpha=.8)
cbar = plt.colorbar()
cbar.set_label(r'$\mu$gal')
plt.show()

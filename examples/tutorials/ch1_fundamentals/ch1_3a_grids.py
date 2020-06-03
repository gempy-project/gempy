"""
1.3a: Grids.
============
"""

import numpy as np
import pandas as pd
from gempy.core.data import Grid
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt

pd.set_option('precision', 2)
np.random.seed(55500)

# %%
# The Grid Class
# --------------
# 
# The grid class will interact with the rest of data classes and grid
# subclasses. Its main purpose is to feed coordinates XYZ to the
# interpolator.
# 

# %% 
grid = Grid()

# %%
# The most important attribute of Grid is ``values`` (and ``values_r``
# which are the values rescaled) which are the 3D points in space that
# kriging will be evaluated on. This array will be feed by "grid types" on
# a **composition** relation with Grid:
# 


# %%
# .. image:: /../../_static/grids.jpg
# 

# %% 
grid.values, grid.values_r

# %%
# At the moment of writing this tutorial, there is 5 grid types. The
# number of grid types is scalable and down the road we aim to connect
# other grid packages (like `Discretize <https://pypi.org/project/discretize/>`_) as an extra Grid type
# 

# %% 
grid.grid_types

# %%
# Each grid contains its own ``values`` attribute as well as other
# methods to manipulate them depending on the type of grid.
# 

# %% 
grid.regular_grid.values

# %%
# We can see what grids are activated (i.e. they are going to be
# interpolated and therefore will live on ``Grid().values``) by:
# 

# %% 
grid.active_grids

# %%
# By default only the *regular grid* (``grid.regular_grid``\ ) is active. However, since the regular
# grid is still empty ``Grid().values`` is empty too.
# 

# %% 
grid.values

# %%
# The last important attribute of Grid is the length:
# 

# %% 
grid.length

# %%
# Length gives back the interface indices between grids on the
# ``Grid().values`` attribute. This can be used after interpolation to
# know which interpolated values and coordinates correspond to each grid
# type. You can use the method get\_grid\_args to return the indices by
# name:
# 

# %% 
grid.get_grid_args('topography')

# %%
# By now all is a bit confusing because we have no values. Lets start
# adding values to the different grids:
# 


# %%
# Regular grid
# ~~~~~~~~~~~~
# 
# The ``Grid`` class has a bunch of methods to set each grid type and
# activate them.
# 

# %% 
help(grid.create_regular_grid)

# %% 
grid.create_regular_grid(extent=[0, 100, 0, 100, -100, 0], resolution=[20, 20, 20])

# %%
# Now the regular grid object composed on ``Grid`` has been filled:
# 

# %% 
grid.regular_grid.values

# %%
# And the regular grid has been set active (it was already active in any
# case):
# 

# %% 
grid.active_grids

# %%
# Therefore the grid values will be equal to the regular grid:
# 

# %% 
grid.values

# %%
# And the indices to extract the different arrays:
# 

# %% 
grid.length

# %%
# Custom grid
# ~~~~~~~~~~~
# 
# Completely free XYZ values.
# 

# %% 
grid.create_custom_grid(np.array([[1, 2, 3],
                                  [4, 5, 6],
                                  [7, 8, 9]]))

# %%
# Again ``set_any_grid`` will create a grid and activate it. So now the
# compose object will contain values:
# 

# %% 
grid.custom_grid.values

# %%
# and since it is active, will be added to the grid.values stack:
# 

# %% 
grid.active_grids

# %% 
grid.values.shape

# %%
# We can still recover those values with ``get_grid`` or by getting the
# slicing args:
# 

# %% 
grid.get_grid('custom')

# %% 
l0, l1 = grid.get_grid_args('custom')
l0, l1

# %% 
grid.values[l0:l1]

# %%
# Topography
# ~~~~~~~~~~
# 
# Now we can set the topography. :class:`Topography <gempy.core.grid_modules.topography.Topography>`
# contains methods to create manual topographies as well as gdal for
# dealing with raster data. By default we will create a random topography:
# 

# %%
grid.create_topography()

# %% 
grid.active_grids

# %%
# Now the grid values will contain both the regular grid and topography:
# 

# %% 
grid.values, grid.length

# %%
# The topography args are got as follows:
# 

# %% 
l0, l1 = grid.get_grid_args('topography')
l0, l1

# %%
# And we can slice the values array as any other numpy array:
# 

# %% 
grid.values[l0: l1]

# %%
# We can compare it to the topography.values:
# 

# %% 
grid.topography.values

# %%
# Now that we have more than one grid we can activate and deactivate any
# of them in real time:
# 

# %% 
grid.set_inactive('topography')
grid.set_inactive('regular')

# %%
# Since now all grids are deactivated the values will be empty:
# 

# %% 
grid.values

# %% 
grid.set_active('topography')

# %% 
grid.values, grid.values.shape

# %% 
grid.set_active('regular')

# %% 
grid.values

# %%
# Centered Grid
# ~~~~~~~~~~~~~
# 
# This grid contains an irregular grid where the majority of voxels are
# centered around a value (or values). This type of grid is usually used
# to compute certain types of forward physics where the influence
# decreases with distance (e.g. gravity: Check `tutorial 2.2-Cell-selection <https://github.com/cgre-aachen/gempy/blob/master/examples/tutorials/ch2-Geophysics/ch2_2_cell_selection.py>`_
# )
# 

# %% 
grid.create_centered_grid(centers=np.array([[300, 0, 0], [0, 0, 0]]),
                          resolution=[10, 10, 20], radius=100)

# %%
# Resolution and radius create a geometric spaced kernel (blue dots) which
# will be use to create a grid around each of the center points (red
# dots):
# 

# %% 

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(grid.values[:, 0], grid.values[:, 1], grid.values[:, 2], '.', alpha=.2)
ax.scatter(np.array([[300, 0, 0], [0, 0, 0]])[:, 0],
           np.array([[300, 0, 0], [0, 0, 0]])[:, 1],
           np.array([[300, 0, 0], [0, 0, 0]])[:, 2], c='r', alpha=1, s=30)

ax.set_xlim(-100, 400)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()

# %%
# Section Grid
# ~~~~~~~~~~~~
# 
# This grid type has its own tutorial. See ch1-3b
#


"""
1.3a: Grids.
============
"""

import numpy as np
import matplotlib.pyplot as plt

import gempy as gp
from gempy.core.data import Grid
from gempy.core.data.grid_modules import RegularGrid

np.random.seed(55500)

# %%
# The Grid Class
# --------------
# 
# The Grid class interacts with the rest of the data classes and grid
# subclasses. Its main purpose is to feed coordinates XYZ to the
# interpolator.
# 

# %% 
grid = Grid()

# %%
# The most important attribute of Grid is ``values`` (and ``values_r``
# which are the values rescaled) which are the 3D points in space that
# kriging will be evaluated on. This array will be fed by "grid types" on
# a **composition** relation with Grid:
# 

# %%
# .. image:: /_static/grids.jpg
# 

# %% 
print(grid.values)

# %%
# At the moment of writing this tutorial, there are 5 grid types. The
# number of grid types is scalable, and down the road we aim to connect
# other grid packages (like `Discretize <https://pypi.org/project/discretize/>`_) as an extra Grid type.
# 

# %% 
# This is an enum now
print(grid.GridTypes)

# %%
# Each grid contains its own ``values`` attribute as well as other
# methods to manipulate them depending on the type of grid.
# 

# %% 
print(grid.values)

# %%
# We can see which grids are activated (i.e. they are going to be
# interpolated and therefore will live on ``Grid().values``) by:
# 

# %% 
print(grid.active_grids)

# %%
# By default, only the *regular grid* (``grid.regular_grid``) is active. However, since the regular
# grid is still empty, ``Grid().values`` is empty too.
# 

# %% 
print(grid.values)

# %%
# The last important attribute of Grid is the length:
# 

# %% 
print(grid.length)

# %%
# Length gives back the interface indices between grids on the
# ``Grid().values`` attribute. This can be used after interpolation to
# know which interpolated values and coordinates correspond to each grid
# type. You can use the method ``get_grid_args`` to return the indices by
# name:
# 

# %% 
print(grid.topography)

# %%
# By now all is a bit confusing because we have no values. Let's start
# adding values to the different grids:
# 

# %%
# Regular Grid
# ~~~~~~~~~~~~
# 
# The ``Grid`` class has a bunch of methods to set each grid type and
# activate them.
# 

# %% 
help(RegularGrid)

# %% 
extent = np.array([0, 100, 0, 100, -100, 0])
resolution = np.array([20, 20, 20])
grid.dense_grid = RegularGrid(extent, resolution)
print(grid.regular_grid)  # RegularGrid will return either dense grid or octree grid depending on what is set

# %%
# Now the regular grid object composed in ``Grid`` has been filled:
# 

# %% 
print(grid.regular_grid.values)

# %%
# And the regular grid has been set active (it was already active in any
# case):
# 

# %% 
print(grid.active_grids)

# %%
# Therefore, the grid values will be equal to the regular grid:
# 

# %% 
print(grid.values)

# %%
# And the indices to extract the different arrays:
# 

# %% 
print(grid.length)

# %%
# Custom Grid
# ~~~~~~~~~~~
# 
# Completely free XYZ values.
# 

# %% 
gp.set_custom_grid(grid, np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))

# %%
# Again, ``set_any_grid`` will create a grid and activate it. So now the
# composed object will contain values:
# 

# %% 
print(grid.custom_grid.values)

# %%
# And since it is active, it will be added to the grid.values stack:
# 

# %% 
print(grid.active_grids)

# %% 
print(grid.values.shape)

# %%
# We can still recover those values with ``get_grid`` or by getting the
# slicing args:
# 

# %% 
print(grid.custom_grid)

# %% 
print(grid.custom_grid.values)

# %%
# Topography
# ~~~~~~~~~~
# 
# Now we can set the topography. :class:`Topography <gempy.core.grid_modules.topography.Topography>`
# contains methods to create manual topographies as well as using gdal for
# dealing with raster data. By default, we will create a random topography:
# 

# %%
gp.set_topography_from_random(grid)

# %% 
print(grid.active_grids)

# %%
# Now the grid values will contain both the regular grid and topography:
# 

# %% 
print(grid.values, grid.length)

# %% 
print(grid.topography.values)

# %%
# We can compare it to the topography.values:
# 

# %% 
print(grid.topography.values)

# %%
# Now that we have more than one grid, we can activate and deactivate any
# of them in real time:
# 

# %% 
grid.active_grids ^= grid.GridTypes.TOPOGRAPHY
grid.active_grids ^= grid.GridTypes.DENSE

# %%
# Since now all grids are deactivated, the values will be empty:
# 

# %% 
print(grid.values)

# %% 
grid.active_grids |= grid.GridTypes.TOPOGRAPHY

# %% 
print(grid.values, grid.values.shape)

# %% 
grid.active_grids |= grid.GridTypes.DENSE

# %% 
print(grid.values)

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
gp.set_centered_grid(
    grid,
    centers=np.array([[300, 0, 0], [0, 0, 0]]),
    resolution=[10, 10, 20],
    radius=np.array([100, 100, 100])
)

# %%
# Resolution and radius create a geometrically spaced kernel (blue dots) which
# will be used to create a grid around each of the center points (red
# dots):
# 

# %% 

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(
    grid.centered_grid.values[:, 0],
    grid.centered_grid.values[:, 1],
    grid.centered_grid.values[:, 2],
    '.',
    alpha=.2
)

ax.scatter(
    np.array([[300, 0, 0], [0, 0, 0]])[:, 0],
    np.array([[300, 0, 0], [0, 0, 0]])[:, 1],
    np.array([[300, 0, 0], [0, 0, 0]])[:, 2],
    c='r',
    alpha=1,
    s=30
)

ax.set_xlim(-100, 400)
ax.set_ylim(-100, 100)
ax.set_zlim(-120, 0)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()

# %%
# Section Grid
# ~~~~~~~~~~~~
# 
# This grid type has its own tutorial. See :doc:`ch1_3b_cross_sections`
#

"""
Centered Grids for Geophysics
================================

Precomputing the gravity kernel of a measurement-centered grid

This tutorial explains the concept of a centered grid and uses it to precompute the tz
kernel needed for GemPy's forward gravity computation.
"""

# %%
# Concept of a measurement-centered grid
# """"""""""""""""""""""""""""""""""""""
#
# Geophysics preprocessing builds on the centered grid (see the Grids tutorial in the
# Advanced section) to precompute the constant part of forward physical computations,
# such as gravity:

# .. math::

#     F_z = G_{\rho} ||| x \ln(y+r) + y \ln (x+r) - z \arctan (\frac{x y}{z r}) |^{x_2}_{x_1}|^{y_2}_{y_1}|^{
#     z_2}_{z_1}


# where we can compress the grid-dependent terms as

# .. math::

#     t_z = ||| x \ln (y+r) + y \ln (x+r)-z \arctan ( \frac{x y}{z r} ) |^{x_2}_{x_1}|^{y_2}_{y_1}|^{z_2}_{z_1}

# By doing this decomposition and keeping the grid constant, we can compute the forward
# gravity with a single operation:

# .. math::

#     F_z = G_{\rho} \cdot t_z


# %%

# Importing gempy
import gempy as gp

# Aux imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(1515)
pd.set_option('display.precision', 2)

# %% 
from gempy_engine.core.data.centered_grid import CenteredGrid
centered_grid = CenteredGrid(
    centers=np.array([0,0,0]),
    resolution=[10, 10, 20],
    radius=100
)
# %%
# Instantiating ``CenteredGrid`` creates a constant kernel around the point (0, 0, 0).
# This kernel will be what we use for each device.
#

# %% 
centered_grid.kernel_grid_centers

# %%
# :math:`t_z` is only dependent on distance and therefore we can use the
# kernel created in the previous cell
# 

# %% 
gravity_gradient = gp.calculate_gravity_gradient(centered_grid)
gravity_gradient

# %%
# To compute tz we also need the edges of each voxel. The distances to the
# edges are stored in ``left_voxel_edges`` and ``right_voxel_edges``. We
# can plot all the data as follows:
#

# %% 
a, b, c = centered_grid.kernel_grid_centers, centered_grid.left_voxel_edges, centered_grid.right_voxel_edges
tz = gravity_gradient

# %% 
fig = plt.figure(figsize=(13, 7))
plt.quiver(a[:, 0].reshape(11, 11, 21)[5, :, :].ravel(),
           a[:, 2].reshape(11, 11, 21)[:, 5, :].ravel(),
           np.zeros(231),
           tz.reshape(11, 11, 21)[5, :, :].ravel(), label='$t_z$', alpha=.3
           )

plt.plot(a[:, 0].reshape(11, 11, 21)[5, :, :].ravel(),
         a[:, 2].reshape(11, 11, 21)[:, 5, :].ravel(), 'o', alpha=.3, label='Centers')

plt.plot(a[:, 0].reshape(11, 11, 21)[5, :, :].ravel() - b[:, 0].reshape(11, 11, 21)[5, :, :].ravel(),
         a[:, 2].reshape(11, 11, 21)[:, 5, :].ravel(), '.', alpha=.3, label='Lefts')

plt.plot(a[:, 0].reshape(11, 11, 21)[5, :, :].ravel(),
         a[:, 2].reshape(11, 11, 21)[:, 5, :].ravel() - b[:, 2].reshape(11, 11, 21)[:, 5, :].ravel(), '.', alpha=.6,
         label='Ups')

plt.plot(a[:, 0].reshape(11, 11, 21)[5, :, :].ravel() + c[:, 0].reshape(11, 11, 21)[5, :, :].ravel(),
         a[:, 2].reshape(11, 11, 21)[:, 5, :].ravel(), '.', alpha=.3, label='Rights')

plt.plot(a[:, 0].reshape(11, 11, 21)[5, :, :].ravel(),
         a[:, 2].reshape(11, 11, 21)[:, 5, :].ravel() + c[:, 2].reshape(11, 11, 21)[5, :, :].ravel(), '.', alpha=.3,
         label='Downs')

plt.xlim(-200, 200)
plt.ylim(-200, 0)
plt.legend()
plt.show()

# %%
# Just the quiver:
# 

# %%
fig = plt.figure(figsize=(13, 7))
plt.quiver(a[:, 0].reshape(11, 11, 21)[5, :, :].ravel(),
           a[:, 2].reshape(11, 11, 21)[:, 5, :].ravel(),
           np.zeros(231),
           tz.reshape(11, 11, 21)[5, :, :].ravel()
           )
plt.show()

# %%
# Remember this is happening always in 3D:
# 

# %%
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(a[:, 0], a[:, 1], a[:, 2], c=tz)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()
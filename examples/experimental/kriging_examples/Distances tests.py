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

# More distance stuff
# These two lines are necessary only if GemPy is not installed
import sys, os
sys.path.append("../..")
# Importing GemPy
import gempy as gp
# Embedding matplotlib figures in the notebooks
# %matplotlib inline
# Importing auxiliary libraries
import numpy as np
import matplotlib.pyplot as plt

""
# set resolution, extent and input data
geo_data = gp.create_data([0,1000,0,1000,0,1000],resolution=[50,1,50], 
                        path_o = os.pardir+"/data/input_data/jan_models/model2_orientations.csv",
                        path_i = os.pardir+"/data/input_data/jan_models/model2_surface_points.csv") 

""
#F4B400
#DB4437
#4285F4
geo_data.surfaces.colors.change_colors({"rock1": '#DB4437', "rock2": "#4285F4", "basement": "#F4B400"})

""
#geo_data.surfaces.colors.change_colors()

""
# define series and assign surfaces
gp.map_series_to_surfaces(geo_data, {"Strat_Series": ('rock2','rock1'),"Basement_Series":('basement')})

""
# define the interpolator
interp_data = gp.set_interpolation_data(geo_data, compile_theano=True,
                                        theano_optimizer='fast_compile')

""
#calcualte the solution
sol = gp.compute_model(geo_data, compute_mesh=False)

""
sol.lith_block.shape
sol.grid.regular_grid.values[:,1].shape

""
sol.scalar_field_matrix.shape
sol.gradient

""
gp.plot.plot_section(geo_data, cell_number=0)

""


""
# gradient not implemented in gempy 2.0
np.unique(sol.lith_block)

""
plt.scatter(sol.grid.regular_grid.values[:,0], sol.grid.regular_grid.values[:,2], c=sol.lith_block, cmap='viridis')

""
#sol.scalar_field_matrix[0]

""
x =  np.unique(sol.grid.regular_grid.values[:,0])
y =  np.unique(sol.grid.regular_grid.values[:,2])

""
X, Y = np.meshgrid(x, y)

zs = sol.scalar_field_matrix
Z = zs.reshape(X.shape)



""
sol.grid.regular_grid.resolution

""
gx, gy = np.gradient(Z,20,20)

""
fig, ax = plt.subplots()
q = ax.quiver(Y[::3, ::3], X[::3, ::3], gx[::3, ::3], gy[::3, ::3])

ax.quiverkey(q, X=0.3, Y=1.1, U=1,
             label='Quiver key, length = 1')

plt.scatter(sol.grid.regular_grid.values[:,0], sol.grid.regular_grid.values[:,2], s=10, c=sol.lith_block, cmap='viridis')
plt.show()

""

fig, ax = plt.subplots()
#q = ax.quiver(Y, X, gx, gy)
q = ax.quiver(Y[::3, ::3], X[::3, ::3], gx[::3, ::3], gy[::3, ::3])

#ax.quiverkey(q, X=0.3, Y=1.1, U=1, label='Quiver key, length = 1')

#plt.scatter(sol.grid.regular_grid.values[:,0], sol.grid.regular_grid.values[:,2], s=10, c=sol.lith_block, cmap='viridis')

gp.plot.plot_section(geo_data, cell_number=0)

plt.show()

""


""
# %matplotlib inline

gp.plot.plot_section(geo_data, cell_number=0)
gp.plot.plot_scalar_field(geo_data, cell_number=0, N=20, show_data=False, alpha=0, colors='k', cmap=None, linewidths=1)

""
gp.plot.plot_section(geo_data, cell_number=0)
#plt.scatter(sol.grid.regular_grid.values[:,0], sol.grid.regular_grid.values[:,2], s=1, c='black')
plt.scatter(sol.grid.regular_grid.values[165,0], sol.grid.regular_grid.values[165,2], s=10, c='black', marker='o')
plt.scatter(sol.grid.regular_grid.values[2076,0], sol.grid.regular_grid.values[2076,2], s=10, c='black', marker='o')
#plt.plot()
gp.plot.plot_scalar_field(geo_data, cell_number=0, N=1, show_data=False, alpha=0, colors='k', cmap=None, linewidths=1)

""


""


""
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, gx)

""


""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


""
def fun(x, y):
    return x**2 + y


""
x = np.arange(-10,10,2)
y = np.arange(-10,10,2)

X, Y = np.meshgrid(x, y)

zs = np.array(fun(np.ravel(X), np.ravel(Y)))
Z = zs.reshape(X.shape)

""
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z)

""
gx,gy = np.gradient(Z,2,2)

""
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, gy)

""


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
sys.path = list(np.insert(sys.path, 0, "../../../pyvista"))


#sys.path("../../../pyvista")

import pyvista

""
pyvista.__path__

""
path_to_data = os.pardir+"/data/input_data/jan_models/"

geo_data = gp.create_data([0,1000,0,1000,0,1000],resolution=[50,50,50], 
                        path_o = path_to_data + "model5_orientations.csv",
                        path_i = path_to_data + "model5_surface_points.csv") 

""
gp.map_series_to_surfaces(geo_data, {"Fault_Series":'fault', 
                         "Strat_Series": ('rock2','rock1')})
geo_data.set_is_fault(['Fault_Series'])

""
# %matplotlib inline
gp.plot.plot_data(geo_data, direction='y')

""
geo_data.orientations

""
geo_data.surfaces

""
geo_data.orientations.sort_table()

""
interp_data = gp.set_interpolation_data(geo_data, theano_optimizer='fast_compile')

""
sol = gp.compute_model(geo_data)

""
# %matplotlib inline
gp.plot.plot_section(geo_data, cell_number=25,
                         direction='y', show_data=False, show_all_data=True)

"""
## Pyvista
"""

import pyvista as pv
import numpy as np

""
g = geo_data.grid.values
g

""
g_3D = g.reshape(*geo_data.grid.regular_grid.resolution, 3).T
g_3D.shape

""
vista_grid = pv.StructuredGrid(g_3D[0], g_3D[1], g_3D[2])
vista_grid = pv.StructuredGrid(*g_3D)

""
vista_grid

""
vista_grid.plot(use_panel=True)

""
d = {'lith': geo_data.solutions.lith_block}

for i in d:
    print(i)

""
vista_grid.point_arrays['lith'] = geo_data.solutions.lith_block

###############################################################################
# Colors:

import matplotlib.colors as mcolors

cmap=mcolors.ListedColormap(list(geo_data.surfaces.df.iloc[:, 4]))
norm = mcolors.Normalize(vmin=0.5, vmax=len(cmap.colors) + 0.5)

###############################################################################
# Plot:

vista_grid.plot(sclars='lith', show_edges=True, cmap = cmap, norm=norm, use_panel=False, clim=[1,4])

###############################################################################
# ### Plot data:

from importlib import reload
reload(pv)

""
p = pv.Plotter(notebook=False,)
if True:
    def foo(a, b):
        print(a,b, b.WIDGET_INDEX)
        return None
    
    k = p.add_sphere_widget(foo, center=(2,2,2))
    k.WIDGET_INDEX=55

""
p.bound= geo_data.grid.regular_grid.extent


""
surf1 = pv.PolyData(geo_data.solutions.vertices[0])
ss = surf1.delaunay_2d()
p.add_mesh(ss)

""

_e = geo_data.grid.regular_grid.extent
_e_dx = _e[1] - _e[0]
_e_dy = _e[3] - _e[2]
_e_dz = _e[5] - _e[4]
_e_d_avrg = (_e_dx + _e_dy + _e_dz) / 3

r_ = _e_d_avrg * .03


""
def foo(a):
    return None

def foo2(a, b):
    return None

if True:
    for e, val in geo_data.surface_points.df.iterrows():

        #print(val)
        #c = mcolors.hex2color(geo_data.surfaces.df.set_index('id')['color'][1])
        c = geo_data.surfaces.df.set_index('id')['color'][val['id']]
        a = p.add_sphere_widget(foo, center=val[['X', 'Y', 'Z']], color=c, radius=r_)

if True:
    for e, val in geo_data.orientations.df.iterrows():

        #print(val)
        #c = mcolors.hex2color(geo_data.surfaces.df.set_index('id')['color'][1])
        c = geo_data.surfaces.df.set_index('id')['color'][val['id']]
        p.add_plane_widget_simple(foo2, normal=val[['G_x', 'G_y', 'G_z']], 
                                  origin=val[['X', 'Y', 'Z']], color=c,
                                  bounds =  geo_data.grid.regular_grid.extent, factor=.1)

""
p.show_grid()


""
p.show()

###############################################################################
# ### Plot surface:

geo_data.solutions.vertices[0],  geo_data.solutions.edges[0]

###############################################################################
# #### This crash the kernel:

tri = geo_data.solutions.edges[0]



""
surf = pv.PolyData(geo_data.solutions.vertices[0], np.insert(tri, 0, 3, axis=1).ravel())
#surf.plot()

""
p.add_mesh(surf)

""
mesh = Out[49]



""
surf.color

""
surf.points /=10

""
surf.faces

""
surf.points = geo_data.solutions.vertices[1]
surf.faces =  np.insert(geo_data.solutions.edges[1], 0, 3, axis=1).ravel()

""
surf.plot()

###############################################################################
# ### Background Plotter

#p = pv.BackgroundPlotter()

""
#p.add_mesh(vista_grid, sclars='lith', show_edges=True, cmap = cmap)

""
#p.add_mesh(surf)

###############################################################################
# ### Testing stuff

import pyvista as pv
import numpy as np

""
n = 20
x = np.linspace(-200, 200, num=n) + np.random.uniform(-5, 5, size=n)
y = np.linspace(-200, 200, num=n) + np.random.uniform(-5, 5, size=n)
xx, yy = np.meshgrid(x, y)
A, b = 100, 100
zz = A * np.exp(-0.5 * ((xx / b) ** 2.0 + (yy / b) ** 2.0))

# Get the points as a 2D NumPy array (N by 3)
points = np.c_[xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)]
print(points[0:5, :])

""
cloud = pv.PolyData(points)
cloud.plot(point_size=15)

""
surf = cloud.delaunay_2d()
surf.plot(show_edges=True)


""
surf.points

""
surf.faces

""
import pyvista as pv
import numpy as np
# Create a triangle surface
surf = pv.PolyData()
surf.points = np.array([[-10,-10,-10],
                    [10,10,-10],
                    [-10,10,0],])
surf.faces = np.array([3, 0, 1, 2])
p = pv.Plotter(notebook=False)
def callback(point):
    surf.points[0] = point
#p.enable_sphere_widget(callback)

p.add_sphere_widget(callback)
p.add_mesh(surf, color=True)
p.show_grid()
p.show()


""
from scipy.interpolate import griddata
import numpy as np
import pyvista as pv
def get_colors(n):
    """A haleper function to get n colors"""
    from itertools import cycle
    import matplotlib
    cycler = matplotlib.rcParams['axes.prop_cycle']
    colors = cycle(cycler)
    colors = [next(colors)['color'] for i in range(n)]
    return colors
# Create a grid to interpolate to
xmin, xmax, ymin, ymax = 0, 100, 0, 100
x = np.linspace(xmin, xmax, num=25)
y = np.linspace(ymin, ymax, num=25)
xx, yy, zz = np.meshgrid(x, y, [0])
# Make sure boundary conditions exist
boundaries = np.array([[xmin,ymin,0],
                   [xmin,ymax,0],
                   [xmax,ymin,0],
                   [xmax,ymax,0]])
# Create the PyVista mesh to hold this grid
surf = pv.StructuredGrid(xx, yy, zz)
# Create some intial perturbations
# - this array will be updated inplace
points = np.array([[33,25,45],
               [70,80,13],
               [51,57,10],
               [25,69,20]])
# Create an interpolation function to update that surface mesh
def update_surface(point, i):
    points[i] = point
    tp = np.vstack((points, boundaries))
    zz = griddata(tp[:,0:2], tp[:,2], (xx[:,:,0], yy[:,:,0]), method='cubic')
    surf.points[:,-1] = zz.ravel(order='F')
    return
# Get a list of unique colors for each widget
colors = get_colors(len(points))
# Begin the plotting routine
p = pv.Plotter(notebook=False)
# Add the surface to the scene
p.add_mesh(surf, color=True)
# Add the widgets which will update the surface
p.enable_sphere_widget(update_surface, center=points,
                       color=colors, radius=3)
# Add axes grid
p.show_grid()
# Show it!
p.show()

""


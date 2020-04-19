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
# Importing Kinematic modelling into GemPy using pynoddy
"""

import sys, os
# Path to development gempy
sys.path.append('../../..')

# Path to development pynoddy
sys.path.append('../../../../pynoddy')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pn
# adjust some settings for matplotlib
from matplotlib import rcParams
# print rcParams
rcParams['font.size'] = 15
# determine path of repository to set paths corretly below
repo_path = os.path.realpath('../..')
import gempy as gp
import pynoddy
import pynoddy.history
import pynoddy.output
# %matplotlib inline

###############################################################################
# ## Making your favorite model in pynoddy
# ### Loading noddy model

#reload(pynoddy.history)
# Downloading a model
# his = pynoddy.history.NoddyHistory(url = \
#             "http://tectonique.net/asg/ch2/ch2_2/ch2_2_1/his/normal.his")
his = pynoddy.history.NoddyHistory('simple_model.his')
his.determine_model_stratigraphy()

""
# Choosing resolution
his.change_cube_size(200)

""
# Writing history file
history_name = "fold_thrust.his"
his.write_history(history_name)

""
# Computing history file
output = "fold_thrust_out"
pynoddy.compute_model(history_name, output, sim_type='BLOCK')

""
# load and visualise model
h_out = pynoddy.output.NoddyOutput(output)

""
# his.determine_model_stratigraphy()
h_out.plot_section('y', position=,
                   layer_labels = his.model_stratigraphy, 
                   colorbar_orientation = 'horizontal', 
                   colorbar=False,
                   title = '',
#                   savefig=True, fig_filename = 'fold_thrust_NS_section.eps',
                   cmap = 'YlOrRd')

""
a = gp.utils.find_interfaces_from_block(block, 2)

""


""
b = (block>2)[10, :, 1:] ^ (block>2)[10, :, :-1]

""
b = (block>2)[10, 1:, :] ^ (block>2)[10, -1:, :]

""
plt.imshow(b.T, origin='bottom')

""
np.array([0,1,1,0], dtype=bool) + np.array([0,1,1,0], dtype=bool)

""
plt.imshow(a[10, :, :].T, origin='bottom')

""
resolution, noddy_grid.reshape(resolution)

""
# Checkpoint. Saving noddy litholody block
np.save('noddy_block', h_out.block)

###############################################################################
# ## GemPy finding interface points

# Creating geo_data with the same data extent and resolution as the noddy model

# initialize geo_data object
geo_data = gp.create_data([-14000, 44000, 
                           -14000, 44000, 
                           -10000, 3000],
                          resolution=[80, 80, 80])
block = np.load('noddy_block.npy')

""
# extent = [0, 30000.0,
#           0, 30000.0,
#           0, 3000.0  ]# h_out.extent_x, h_out.extent_y, h_out.extent_z
extent = [0, h_out.extent_x,
          0, h_out.extent_y,
          0, h_out.extent_z]
cs = 200 #h_out.delx

resolution = [int(extent[1]/cs), 
              int(extent[3]/cs), 
              int(extent[5]/cs)]

noddy_grid = gp.GridClass.create_regular_grid_3d(extent, resolution)

""
# Importing some points at the interface
gp.utils.set_interfaces_from_block(geo_data, block, noddy_grid, reset_index=True)

""
a = gp.utils.find_interfaces_from_block(block, 0)

""
np.unique(block)

""
plt.imshow(a[:,20,:])

""
gp.plotting.plot_data_3D(geo_data)

""
# Visualiziing
# %matplotlib inline
gp.plotting.plot_data(geo_data, direction='y')

""
geo_data.interfaces.head()

""
geo_data.extent

""
# Setting orientation from interfaces
gp.set_orientation_from_interfaces(geo_data, [5,6,7,8,9,])

""
gp.plotting.plot_data_3D(geo_data)

""
gp.plotting.plot_data(geo_data, direction='x')

###############################################################################
# ### Computing and visualizing gempy model

interp_data = gp.InterpolatorData(geo_data, compile_theano=True)

""
lith, fault = gp.compute_model(interp_data)

""
# %matplotlib inline
gp.plotting.plot_section(geo_data, lith[0], 25, direction='y', plot_data= True)

""
ver, sim = gp.get_surfaces(interp_data, lith[1], None)
#gp.plotting.plot_surfaces_3D_real_time(geo_data, interp_data, ver, sim)

###############################################################################
# ## Comparing physics
# ### pynoddy gravity
#
# Density is 2.5 and 3.5

pynoddy.compute_model(history_name, output, sim_type = 'GEOPHYSICS')

""
geophys1 = pynoddy.output.NoddyGeophysics(output)

""
import matplotlib.pyplot as plt

plt.imshow(geophys1.grv_data, cmap='viridis', origin='lower',
           extent=extent[:-2] )
plt.colorbar()

###############################################################################
# ### GemPy Gravity

gp.get_data(geo_data, 'formations')

""
gp.set_formations(geo_data, formations_values=[2.5, 3.5])

""
interp_data_grav = gp.InterpolatorData(geo_data, output='gravity',
                                       compile_theano=True)

""
gp.set_geophysics_obj(interp_data_grav,
                      extent,
                      [50, 50])


""
gp.precomputations_gravity(interp_data_grav, 20);

""
geo_data.resolution

""


x_g = geo_data.grid.values[:,0].reshape(geo_data.resolution)[40:,:,:]
y_g = geo_data.grid.values[:,1].reshape(geo_data.resolution)[40:,:,:]
z_g = geo_data.grid.values[:,2].reshape(geo_data.resolution)[40:,:,:]

""
import vtkInterface
import numpy as np
#[7.050000e+05,747000,6863000,6925000,-20000, 200

# Lith block grid
grid = vtkInterface.StructuredGrid(
x_g, y_g, z_g
                               )
# Fixing lith block direction
e = lith[0].reshape(geo_data.resolution)[40:,:,:]
g = e.swapaxes(0,2)

# Gravity  mesh

#.reshape(20,30), cmap='viridis', origin='lower', alpha=0.8, extent=[0,20e3,0,10e3]
# x_v = np.linspace(0, 20e3, 30)
# y_v = np.linspace(0, 10e3, 20)
# z_v = 1500
# x, y, z = np.meshgrid(x_v, y_v, z_v)

# a = vtkInterface.StructuredGrid(x,y, z)

""
import copy

col = copy.deepcopy(gp.plotting.colors.cmap)
plobj = vtkInterface.PlotClass()
plobj.AddMesh(grid, scalars= g,
              showedges=True,
              interpolatebeforemap=False, colormap=col,
              lighting=False)

""
plobj.Plot()

""


""


""


""
interp_data_grav.update_interpolator(geo_data)

""
interp_data_grav.geophy.range_max

""
interp_data_grav.geo_data_res.extent[5]-interp_data_grav.geo_data_res.extent[4]

""
interp_data_grav.geo_data_res.extent

""
interp_data_grav.interpolator.tg.tz.get_value()[0].sum()

""
lith2, fault, grav = gp.compute_model(interp_data_grav, 'gravity')

""
import matplotlib.pyplot as plt

plt.imshow(grav.reshape(50, 50), cmap='viridis', origin='lower',
           extent=extent[:-2] )
plt.colorbar()

###############################################################################
# ### Comparing gravities
#
# We set gempy values to to noddys

G = grav.reshape(50, 50)
N = geophys1.grv_data

# rs_min, rs_max = np.min(grav_real['G']), np.max(grav_real['G'])
# rs_range = rs_max - rs_min
# rs_mid = 0.5*(rs_max+rs_min)

# Calibration parameters
G_min, G_max =  np.min(G), np.max(G)   #36.630742, 36.651496    #30.159309, 30.174104#
N_min, N_max = np.min(N), np.max(N)

# Average
G_mid = 0.5 * (G_max + G_min)
N_mid = 0.5 * (N_max + N_min)

# Shifting
G_range = G_max - G_min
N_range = N_max - N_min
 

# Rescaling
Reescaled_G = N_mid + (G - G_mid) / G_range * N_range

# e_sq = T.sqrt(T.sum(T.square(Reescaled_G - (grav_real_th))))

""
import matplotlib.pyplot as plt

plt.imshow(Reescaled_G.reshape(50, 50), cmap='viridis', origin='lower',
           extent=extent[:-2] )
plt.colorbar()
Reescaled_G.max(), N.max(), Reescaled_G.min(), N.min()

""
(N[::-1, :] - Reescaled_G ). max()

###############################################################################
# ### Dif plot

# %matplotlib inline
plt.imshow(Reescaled_G - N[::-1, :], cmap='bwr', origin='R',
           extent=extent[:-2], vmin=-50, vmax=50)
plt.colorbar()

""
# # his.determine_model_stratigraphy()
# h_out.plot_section('x', 
#                    layer_labels = his.model_stratigraphy, 
#                    colorbar_orientation = 'horizontal', 
#                    colorbar=False,
#                    title = '',
# #                   savefig=True, fig_filename = 'fold_thrust_NS_section.eps',
#                    cmap = 'YlOrRd')

gp.plotting.plot_section(geo_data, lith2[0], 30, direction='x', plot_data= True)
plt.plot(np.linspace(0,30000, 50),
         (geophys1.grv_data[:,25] - geophys1.grv_data[:,0].min()) * 30, linewidth = 5, label = 'noddy', c='black')
plt.plot(np.linspace(0,30000, 50),
          (Reescaled_G[::-1, 0] - Reescaled_G[:, 25].min()) * 30,  linewidth = 5, label = 'gempy', c = 'white')
plt.legend()

""
gp.plotting.plot_section(geo_data, lith2[0], 30, direction='x', plot_data= True)
plt.plot(np.linspace(0,30000, 50),
         (geophys1.grv_data[:,25] - geophys1.grv_data[:,0].min()) * 30, linewidth = 5, label = 'noddy', c='black')
plt.plot(np.linspace(0,30000, 50),
          (Reescaled_G[::-1, 0] - Reescaled_G[:, 25].min()) * 30,  linewidth = 5, label = 'gempy', c = 'white')
plt.legend()

""
interp_data_grav.interpolator.tg.tz.get_value().max()

""
interp_data_grav.interpolator.tg.tz.get_value().min()

""
interp_data_grav.interpolator.tg.tz.get_value().mean()

""
np.arange(-4,-2)

""
plt.hist(interp_data_grav.interpolator.tg.tz.get_value()[1000], log=False,
         bins= 10**(np.arange(-4,-2)))
plt.xscale('log')
plt.yscale('log', nonposy='clip')


""
dev = 10
a = interp_data_grav.interpolator.tg.tz.get_value()[dev]
#a.sort()

""
b = interp_data_grav.interpolator.tg.select.get_value()

""
c=b.reshape(2500, -1)[dev]

""
c = c.astype(bool)

""
d = interp_data_grav.geo_data_res.grid.values[c]

""
d

""
a.shape, d.shape

""
d[11187]

""
np.unique(d[:,1])

""
slicing = np.array(d[:,1]> 0) * np.array( d[:,1]<340)

""
import matplotlib
plt.figure(figsize=(20,10))
val = 100000
plt.scatter(d[slicing,0], d[slicing,2], s=40, c=a[slicing],
            cmap='viridis', norm=matplotlib.colors.LogNorm(), alpha= 1)

# plt.scatter(d[high,1], d[high,2], s=50, c=a[high],
#             cmap='viridis', norm=matplotlib.colors.LogNorm())

plt.colorbar()

""
high = a>1e-4

""
plt.plot(a[:10])

""
a.argmax()

""
plt.plot(a, '.')
plt.yscale('log', nonposy='clip')


""
gp.plotting.plot_section(geo_data, lith[0], 30, direction='x', plot_data= True)

plt.plot(np.linspace(0,30000, 50),
          (grav.reshape(50, 50)[:, 25] - grav.reshape(50, 50)[:, 25].min()) * 3000,  linewidth = 5, label = 'gempy', c = 'white')
plt.legend()


""
# %matplotlib notebook
gp.plotting.plot_section(geo_data, lith2[0], 30, direction='x', plot_data= True)


""
gp.plotting.plot_section(geo_data, lith2[0], 0, direction='z', plot_data= True)


""
plt.plot()

""
plt.plot(geophys.grv_data[:,0])
plt.plot(geophys2.grv_data[:,0])
plt.plot(geophys3.grv_data[:,0])
plt.plot(grav.reshape(50,50)[:,0])

""
plt.plot(grav.reshape(50,50)[:,0])

""
interp_data_grav.geo_data_res.interfaces

""


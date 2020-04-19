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
# Chapter 1: GemPy Basic

In this first example, we will show how to construct a first basic model and the main objects and functions. First we import gempy:
"""

# These two lines are necessary only if gempy is not installed
import sys, os
sys.path.append("../..")

# Importing gempy
import gempy as gp

# Embedding matplotlib figures into the notebooks
# %matplotlib inline

# Aux imports
import numpy as np
np.set_printoptions(linewidth=200)

# Importing the data from csv files and settign extent and resolution
geo_data = gp.create_data([0,2000,0,2000,-2000,0],[ 50,50,50],
                         path_f = os.pardir+"/input_data/FabLessPoints_Foliations.csv",
                         path_i = os.pardir+"/input_data/FabLessPoints_Points.csv")

# Assigning series to formations as well as their order (timewise)
gp.set_series(geo_data, {"fault":'MainFault', 
                      "Rest": ('SecondaryReservoir','Seal', 'Reservoir', 'Overlying'), 
                               },
                       order_series = ["fault", 'Rest'],
                       order_formations=['MainFault', 
                                         'SecondaryReservoir', 'Seal','Reservoir', 'Overlying',
                                         ]) 
#geo_data = gp.select_series(geo_data, ['Rest'])

""
gp.plotting.plot_data_3D(geo_data)

""
gp.plotting.plot_data(geo_data)

""
gp.set_orientation_from_interfaces(geo_data, [0,1,2], verbose=1)

""
geo_data.calculate_gradient()
geo_data.calculate_orientations()

""
np.rad2deg(np.nan_to_num(
                                                  np.arcsin(ge.orientations["G_x"]) /
                                                  (np.sin(np.arccos(self.orientations["G_z"] /
                                                  self.orientations["polarity"])) *

""
np.arsin()

""
gp.get_data(geo_data, 'orientations')

""
gp.get_data(geo_data, 'orientations')

""
gp.get_data(geo_data, 'interfaces')

""
# #gp.set_orientation_from_interfaces(geo_data, [0,1,2])
# geo_data.interfaces.loc[[0]]
# new_point = geo_data.interfaces.loc[[0]]
# # 90 deg
# #new_point['Z'] = -1200
# # 45 deg
# new_point['X'] = 750
# new_point['Z'] = -1500
# geo_data.interfaces = geo_data.interfaces.loc[[0]]
# geo_data.interfaces["X"] = 1500
# geo_data.interfaces["Z"] = -1300
# geo_data.update_df()
# geo_data.set_interfaces(new_point, append=True)
# geo_data.update_df()

# geo_data.modify_orientation(0, dip=130, Y=1000)

#new_ori = geo_data.orientations.loc[[0]]
#new_ori['dip'] = 270
#new_ori['X'] = 150
#geo_data.set_orientations(new_ori, append=True)

""
import theano.tensor as T
import theano
interp_data = gp.InterpolatorData(geo_data,u_grade=[1],
                                  output='gradients', dtype='float64', compile_theano=True,
                                 verbose=['solve_kriging'])

""
lith_block, fault_block =gp.compute_model(interp_data)

""
lith_block

""
geo_data.calculate_gradient()

""
# %matplotlib inline
gp.plot_section(geo_data, lith_block[0], cell_number=25,  direction='y', plot_data=True)

""
r = 1
V, s = np.gradient(lith_block[1].reshape(50,50,50)[::r,25,::r].T )

""
import importlib


""
importlib.reload(gp)

""
importlib.reload(gp.gempy_front)

""
plt.axes().set_aspect('auto')
gp.plot_gradient(geo_data=geo_data, scalar_field=lith_block[1], gx=lith_block[2],
                 gy=lith_block[3], gz=lith_block[4], cell_number=25,direction='y', plot_scalar=True)

""
lith_block[3]

""
import matplotlib.pyplot as plt

""
# %matplotlib notebook
import matplotlib.pyplot as plt
gp.plot_scalar_field(geo_data, lith_block[1], cell_number=25, N=60, 
                        direction='y', plot_data=True)

r = 5
s2= lith_block[2].reshape(50,50,50)[::r,25,::r].T 
V2 = lith_block[4].reshape(50,50,50)[::r,25,::r].T 
#plt.quiver(geo_data.grid.values[:,0].reshape(50,50,50)[::r,25,::r].T,
#           geo_data.grid.values[:,2].reshape(50,50,50)[::r,25,::r].T, 
#           s[::r,::r], 
#           V[::r,::r], pivot="tail")

plt.quiver(geo_data.grid.values[:,0].reshape(50,50,50)[::r,25,::r].T,
           geo_data.grid.values[:,2].reshape(50,50,50)[::r,25,::r].T, s2, V2, pivot="tail",
          color= 'blue', alpha=.6)
# plt.quiver(geo_data.grid.values[:,0].reshape(50,50,50)[::r,25,::r].T,
#            geo_data.grid.values[:,2].reshape(50,50,50)[::r,25,::r].T, s3, V2, pivot="tail",
#           color= 'red')

#plt.contour(lith[1][:1000].reshape(10,10,10)[:,5,:].T, 50, cmap='inferno',  )


""
import theano.tensor as T
import theano
interp_data2 = gp.InterpolatorData(geo_data,u_grade=[1],
                                  output='gradients', dtype='float64', compile_theano=False,
                                 verbose=[])

""
th1 = theano.function(interp_data2.interpolator.tg.input_parameters_list(),
                      interp_data2.interpolator.tg.contribution_gradient())

th2 = theano.function(interp_data2.interpolator.tg.input_parameters_list(),
                      interp_data2.interpolator.tg.contribution_interface_gradient())

th3 = theano.function(interp_data2.interpolator.tg.input_parameters_list(),
                      interp_data2.interpolator.tg.contribution_universal_drift_d())

""
i = interp_data.get_input_data()
a = th1(*i)
b = th2(*i)
c = th3(*i)

""
a[:125000].reshape(50,50,50)[::r,25,::r].T - b[:125000].reshape(50,50,50)[::r,25,::r].T + c[:125000].reshape(50,50,50)[::r,25,::r].T 

""
s3 = a[:125000].reshape(50,50,50)[::r,25,::r].T + b[:125000].reshape(50,50,50)[::r,25,::r].T + c[:125000].reshape(50,50,50)[::r,25,::r].T 
s3

""
from sklearn.preprocessing import scale
scale( s3, axis=0, with_mean=True, with_std=True, copy=True )

""
scale( s[::r,::r] , axis=0, with_mean=True, with_std=True, copy=True )

""
s[::r,::r] 

""
a[:125000].reshape(50,50,50)[::r,25,::r].T 

""
b[:125000].reshape(50,50,50)[::r,25,::r].T 

""
c[:125000].reshape(50,50,50)[::r,25,::r].T 

""


""


""


""
# %matplotlib notebook
import matplotlib.pyplot as plt
gp.plot_scalar_field(geo_data, lith_block[1], cell_number=25, N=60, 
                        direction='x', plot_data=True)

r = 1
V, s = np.gradient(lith_block[1].reshape(50,50,50)[25,::r,::r].T )

r = 5
s2= lith_block[3].reshape(50,50,50)[25,::r,::r].T 
V2 = lith_block[4].reshape(50,50,50)[25,::r,::r].T 
plt.quiver(geo_data.grid.values[:,1].reshape(50,50,50)[25,::r,::r].T,
           geo_data.grid.values[:,2].reshape(50,50,50)[25,::r,::r].T, 
           s[::r,::r], 
           V[::r,::r], pivot="tail", color='green')


plt.quiver(geo_data.grid.values[:,1].reshape(50,50,50)[25,::r,::r].T,
           geo_data.grid.values[:,2].reshape(50,50,50)[25,::r,::r].T, s2, V2, pivot="tail",
          color= 'blue')


""


""


""


""


""
# %matplotlib inline
gp.plot_section(geo_data, fault_block[0], cell_number=25, plot_data=True, direction='y')

""
gp.plot_scalar_field(geo_data, fault_block[1], cell_number=25, N=15, 
                        direction='y', plot_data=False)

r = 5
s= fault_block[2].reshape(50,50,50)[::r,25,::r].T 
V = fault_block[4].reshape(50,50,50)[::r,25,::r].T 
plt.quiver(geo_data.grid.values[:,0].reshape(50,50,50)[::r,25,::r].T,
           geo_data.grid.values[:,2].reshape(50,50,50)[::r,25,::r].T, s, V, pivot="tail")


""
s

""
V

""
geo_data.interfaces.head()

""
#gp.set_orientation_from_interfaces(geo_data, [0,1,2])
geo_data.interfaces.loc[[0]]
new_point = geo_data.interfaces.loc[[0]]
# 90 deg
#new_point['Z'] = -1200
# 45 deg
new_point['X'] = 750
new_point['Z'] = -1500
geo_data.interfaces = geo_data.interfaces.loc[[0]]
geo_data.interfaces["X"] = 850
geo_data.interfaces["Z"] = -1500
geo_data.update_df()
geo_data.set_interfaces(new_point, append=True)
geo_data.update_df()

geo_data.modify_orientation(0, dip=90, Y=1000)

new_ori = geo_data.orientations.loc[[0]]
new_ori['dip'] = 270
new_ori['X'] = 150
geo_data.set_orientations(new_ori, append=True)

""
geo_data.interfaces

""
#gp.set_orientation_from_interfaces(geo_data, [0,1,2])

""



""


""
geo_data.orientations

""
import theano.tensor as T
import theano
interp_data = gp.InterpolatorData(geo_data, u_grade=[1],
                                  output='gradients', dtype='float64', compile_theano=True,
                                 verbose=['solve_kriging', 'scalar_field_at_all'])
print(interp_data)
the = interp_data.interpolator.tg

""
gp.compute_model(interp_data)

""


""


""
import theano.tensor as T
import theano
interp_data = gp.InterpolatorData(geo_data, u_grade=[1],
                                  output='geology', dtype='float64', compile_theano=False,
                                 verbose=['solve_kriging', 'scalar_field_at_all'])
print(interp_data)
the = interp_data.interpolator.tg

""
lith, _ = gp.compute_model(interp_data)


""


""


""
import theano
th_fn2 = theano.function(the.input_parameters_list(),
                         the.compute_geological_model_gradient(),
#                         mode=theano.compile.MonitorMode(
#                         pre_func=inspect_inputs,
#                         post_func=inspect_outputs),
                                    on_unused_input='ignore',
                                    allow_input_downcast=False,
                                    profile=False)
i = interp_data.get_input_data()
a = th_fn2(*i)


""
a

""
# %matplotlib notebook

import matplotlib.pyplot as plt
lith, _ = gp.compute_model(interp_data)

geo_data.calculate_gradient()

#vec_field2 = a[:1000].reshape(10,10,10)
V = np.zeros((10,10))
import matplotlib.pyplot as plt
#plt.quiver(vec_field2[:,5,:].T, V)
s= a[0][2].reshape(10,10,10)[:,5,:].T 
V = a[0][4].reshape(10,10,10)[:,5,:].T 
gp.plot_scalar_field(geo_data, lith[1], 5, direction='y')

plt.quiver(geo_data.grid.values[:,0].reshape(10,10,10)[:,5,:].T,
           geo_data.grid.values[:,2].reshape(10,10,10)[:,5,:].T, s, V, pivot="tail")
#plt.contour(lith[1][:1000].reshape(10,10,10)[:,5,:].T, 50, cmap='inferno',  )
#ax = plt.gca()
#ax.set_ylim(ax.get_ylim()[1], ax.get_ylim()[0])

""
# %matplotlib notebook

import matplotlib.pyplot as plt
lith, _ = gp.compute_model(interp_data)

geo_data.calculate_gradient()

#vec_field2 = a[:1000].reshape(10,10,10)
V = np.zeros((10,10))
import matplotlib.pyplot as plt
#plt.quiver(vec_field2[:,5,:].T, V)
s= a[0][3].reshape(10,10,10)[5,:,:].T 
V = a[0][4].reshape(10,10,10)[5,:,:].T 
gp.plot_scalar_field(geo_data, lith[1], 5, direction='x')

plt.quiver(geo_data.grid.values[:,1].reshape(10,10,10)[5,:,:].T, geo_data.grid.values[:,2].reshape(10,10,10)[5,:,:].T, s, V, pivot="tail")

""
# %matplotlib notebook

import matplotlib.pyplot as plt
lith, _ = gp.compute_model(interp_data)

geo_data.calculate_gradient()

#vec_field2 = a[:1000].reshape(10,10,10)
V = np.zeros((10,10))
import matplotlib.pyplot as plt
#plt.quiver(vec_field2[:,5,:].T, V)
s= a[0][2].reshape(10,10,10)[:,:,5].T 
V = a[0][3].reshape(10,10,10)[:,:,5].T 
gp.plot_scalar_field(geo_data, lith[1], 5, direction='z')

plt.quiver(geo_data.grid.values[:,0].reshape(10,10,10)[:,:,5].T, geo_data.grid.values[:,1].reshape(10,10,10)[:,:,5].T, s, V, pivot="tail")

""
# %matplotlib inline
'''
==============
3D quiver plot
==============

Demonstrates plotting directional arrows at points on a 3d meshgrid.
'''

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.gca(projection='3d')


ax.quiver(geo_data.grid.values[:,0],
              geo_data.grid.values[:,1],
              geo_data.grid.values[:,2],
              b[0][:1000],
              np.zeros_like(b[0][:1000]),
              np.zeros_like(b[0][:1000]), length=80, normalize=False,)

plt.show()

""
import ipyvolume as p4
p4.quiver(geo_data.grid.values[:,0],
              geo_data.grid.values[:,1],
              geo_data.grid.values[:,2],
              b[0][:1000],
              np.zeros_like(b[0][:1000]),
              np.zeros_like(b[0][:1000]), size=10)
p4.show()

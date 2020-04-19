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
sys.path.append("../../..")

# Importing GemPy
import gempy as gp

# Embedding matplotlib figures in the notebooks
# %matplotlib inline

# Importing auxiliary libraries
import numpy as np
import matplotlib.pyplot as plt

import theano
import theano.tensor as T
import pymc3 as pm
theano.config.optimizer_including

from io import StringIO
import sys

import warnings
warnings.filterwarnings("ignore")

"""
### Model creation:
"""

# Data Preparation
path_to_data = os.pardir+"/../data/input_data/jan_models/"

geo_data = gp.create_data([0,1000,0,1000,0,1000],resolution=[10,1,10], 
                        path_o = path_to_data + "model1_orientations.csv",
                        path_i = path_to_data + "model1_surface_points.csv") 
geo_data.delete_surfaces('rock1',remove_data=True)
geo_data.delete_surface_points([1,3,4,5])
geo_data.modify_surface_points([0, 2], Y=500, Z=[500, 600])

""
gp.plot.plot_data(geo_data)

""
geo_data.interpolator.get_python_input_block()[6].shape

""
gp.set_interpolation_data(geo_data, theano_optimizer='fast_compile',
                         verbose=['compare', 'relu'])

""
gp.compute_model(geo_data);

""
geo_data.interpolator.theano_graph.sig_slope.set_value(np.array(50, dtype='float32'))

""
# %matplotlib notebook
gp.plot.plot_section(geo_data, 0, show_data=True)
#gp.plot.plot_scalar_field(geo_data, 0, plot_data=False)

""
gp.plot.plot_scalar_field(geo_data, 0, plot_data=False)
plt.colorbar()

""
geo_data.solutions.scalar_field_matrix.reshape(10,10).max(), geo_data.solutions.scalar_field_matrix.reshape(10,10).min()

""
plt.imshow(geo_data.solutions.scalar_field_matrix.reshape(10,10).T - a[1], cmap='magma', 
          origin='lower')

###############################################################################
# ## Plotting the sigmoid function:

# Cleaning buffer
old_stdout = sys.stdout
mystdout = sys.stdout = StringIO()

# Computing model
gp.compute_model(geo_data)

# Black magic update
sys.stdout = old_stdout

# Split print strings
output = mystdout.getvalue().split('\n')

# Init parameters
n_surface_op_float_sigmoid = []
n_surface_0 = []
n_surface_1 = []
a = []
b = []
drift = []
relu_up = []
relu_down = []

activ = False
activ_d = False
aux_str = ''
aux_str_d = ''

for s in output:
    if 'n_surface_op_float_sigmoid __str__' in s:
        n_surface_op_float_sigmoid.append(np.fromstring(s[s.find('[[')+2:-2], dtype='float', sep=' '))
    if 'n_surface_0 __str__' in s:
        n_surface_0.append(np.fromstring(s[s.find('[[')+2:-2], dtype='float', sep=' '))
    if 'n_surface_1 __str__' in s:
        n_surface_1.append(np.fromstring(s[s.find('[[')+2:-2], dtype='float', sep=' '))
    if 'a __str__' in s:
        a.append(float(s[s.find('= ')+2:]))
    if 'b __str__' in s:
        b.append(float(s[s.find('= ')+2:]))
    if 'drift[slice_init:slice_init+1][0] __str__' in s:
        drift.append(np.fromstring(s[s.find('[[')+2:-2], dtype='float', sep=' '))
    if 'ReLU_up __str__' in s or activ:
        activ = True
        find_c = s.find(']')
        if find_c != -1:
            activ = False
            l_1 = -1
        else:
            l_1 = 10000
        
        l_0 = int(s.find('[')+1)
        aux_str += (s[l_0:l_1])        
  
    
    if 'ReLU_down __str__' in s or activ_d:
        activ_d = True
        find_c = s.find(']')
        if find_c != -1:
            activ_d = False
            l_1 = -1
        else:
            l_1 = 10000
        
        l_0 = int(s.find('[')+1)
        aux_str_d += (s[l_0:l_1])   

relu_down.append(np.fromstring(aux_str_d, dtype='float', sep=' '))
relu_up.append(np.fromstring(aux_str, dtype='float', sep=' '))
  
a, b, n_surface_0, n_surface_1, drift#, relu_up, relu_down

""
relu = relu_up[0] + relu_down[0]
relu;


""
from gempy.utils.gradient import plot_sig
relu = relu_down+ relu_up
plot_sig(n_surface_0[:], n_surface_1, a, b, drift, Z_x = np.linspace(-1,2,2000),
         sf_max=geo_data.solutions.scalar_field_matrix.max(),
         sf_min=geo_data.solutions.scalar_field_matrix.min(),
         sf_at_scalar=geo_data.solutions.scalar_field_at_surface_points[0],
        relu =relu);

""
break

""
a = np.arange(10)
b= np.copy(a)
a, b

""
# %debug

""


###############################################################################
# ### Calculating the jacobian
#
# We are going to recompile so we do not print all the sigmoid values when we use `gp.compute model`

# gp.set_interpolation_data(geo_data, theano_optimizer='fast_run',
#                          verbose=[])

###############################################################################
# In any case, since the gradient is not yet implemented in gempy we need to recompile it appart. For this demo we will compute the jacobian of each of our lith_block respect the surface points: `geo_data.interpolator.theano_graph.input_parameters_loop[4]`

respect = geo_data.interpolator.theano_graph.input_parameters_loop[4]
th_f_j = theano.function(geo_data.interpolator.theano_graph.input_parameters_loop,
                         T.jacobian((geo_data.interpolator.theano_graph.compute_series()[0][-1][:100]), 
                                respect),
                          # mode=NanGuardMode(nan_is_error=True),
                         on_unused_input='ignore')
print("Respect: " + str(respect))

###############################################################################
# We compute it:

geo_data.interpolator.theano_graph.sig_slope.set_value(np.array(50, dtype='float32'))

""
geo_data.modify_orientations(0, dip=330, azimuth=130)

""
jac = th_f_j(*geo_data.interpolator.get_python_input_block())

###############################################################################
# The shape of the result will be:

jac[:, 0,:]

""
np.abs(jac).min(), np.abs(jac).max()

###############################################################################
# - With ReLU 0.1 and slope 50 jac.min-max = 5.75e-05, 12.13 dip 0, azi = 90
# - With ReLU 0.01 and slope 50 jac.min-max = 2.83e-06 11.93 dip 0, azi = 90
#
# - With ReLU 0.01 and slope 50 jac.min-max = 2.83e-06 11.93 dip 30, azi = 130

###############################################################################
# where axis 0 (100) is the number of cells (2 is the number of surface points), axis 1 (2) is respect the 2 surface points and axis 2 (3) XYZ.

# %matplotlib notebook
point = 0

gp.plot.plot_section(geo_data, 0,
                     block=jac[:100, point, 2].reshape(geo_data.grid.regular_grid.resolution), show_data=True,
                     cmap='viridis', show_grid=True, norm=None)

###############################################################################
# If we plot the gradient respect the point 0 (left), we can observe that the high rate changes are concentrated around the interface line of the two surfaces. Also since the model is simetric we can expect the same gradient for the other point:

point = 1

gp.plot.plot_section(geo_data, 0,
                     block=jac[:100, point, 2].reshape(geo_data.grid.regular_grid.resolution), show_data=True,
                     cmap='viridis', show_grid=True, norm=None)

###############################################################################
# ### Gradient?

respect = geo_data.interpolator.theano_graph.input_parameters_loop[4]
th_f_g = theano.function(geo_data.interpolator.theano_graph.input_parameters_loop,
                         T.grad((geo_data.interpolator.theano_graph.compute_series()[0][-1]).sum(), 
                                respect),
                          # mode=NanGuardMode(nan_is_error=True),
                         on_unused_input='ignore')
print("Respect: " + str(respect))

""
grad = th_f_g(*geo_data.interpolator.get_python_input_block())

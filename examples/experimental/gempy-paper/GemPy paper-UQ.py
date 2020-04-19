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
# GemPy Paper Code: UQ

In this notebook you will be able to see and run the code utilized to create the figures of the paper *GemPy - an open-source library for implicit geological modeling and uncertainty quantification*
"""

# Importing dependencies

# These two lines are necessary only if gempy is not installed
import sys, os
sys.path.append("../..")

import gempy as gp
# %matplotlib inline
from copy import copy, deepcopy
# Aux imports

import numpy as np
import pandas as pn
import matplotlib.pyplot as plt

###############################################################################
# ## Uncertainty Quatification.
#
# In this model we will explore how to create a model of GemPy with PyMC which some of the parameters are stochastic  in order to quatify the uncertainty that those parameters propagate to the final results.
#
# We will use the same model as until now

geo_data = gp.create_data([0,20,0,10,-10,0],[100,10,100],
                         path_o = "input_data/paper_Orientations.csv",
                         path_i = "input_data/paper_Points.csv")

""
geo_data.interfaces.head()

###############################################################################
# Defining all different series that form the most complex model. In the paper you can find figures with different combination of these series to examplify the possible types of topolodies supported in GemPy

# %matplotlib inline
gp.set_series(geo_data, {'Fault_serie': 'fault',
                         'Younger_serie' : 'Unconformity',
                         'Older_serie': ('Layer1', 'Layer2')},
              order_formations= ['fault', 'Unconformity',
                                 'Layer2', 'Layer1'],
              verbose=2)


fig=plt.gcf()
# fig.savefig('doc/figs/fault_pile.pdf')

""
gp.plotting.plot_data(geo_data, direction="y")

""
interp_data = gp.InterpolatorData(geo_data,
                                  dtype='float64',
                                  compile_theano=True)

###############################################################################
# Computing the model

lith, fault = gp.compute_model(interp_data)

###############################################################################
# ## Topology
#
# compute the initial topology for use in the topology likelihood function

topo = gp.topology_compute(geo_data, lith[0], fault)
gp.plotting.plot_section(geo_data, lith[0], 0, plot_data=True)
gp.plotting.plot_topology(geo_data, topo[0], topo[1])
# save topology state for likelihood use
topo_G = copy(topo[0])

###############################################################################
# # PYMC2 Code

import pymc

###############################################################################
# ## Priors

geo_data.interfaces.head()

""
#from gempy.UncertaintyAnalysisPYMC2 import modify_plane_dip
from copy import deepcopy
import pymc

###############################################################################
# First we store the original object with the data (rescaled)

geo_data_stoch_init = deepcopy(interp_data.geo_data_res)

""
interp_data.geo_data_res.interfaces.tail()

###############################################################################
# Rescaling the std we want to give to the input data:

0.3/interp_data.rescaling_factor

###############################################################################
# The first step is to define the pdf which describe our priors. In this case we will add noise to the interfaces of 0.3 (0.01 rescaled).

# Positions (rows) of the data we want to make stochastic
ids = range(2,12)

# List with the stochastic parameters 
interface_Z_modifier = [pymc.Normal("interface_Z_mod_"+str(i), 0., 1./0.01**2) for i in ids]

# Plotting the first element of the list
samples = [interface_Z_modifier[0].rand() for i in range(10000)]
plt.hist(samples, bins=24, normed=True);

###############################################################################
# To store the value of the input data---i.e. the value x at every each iteration---we need to wrap the process inside a detereministic decorator of pymc2:

geo_data.interfaces.loc[2, 'Z'] + interface_Z_modifier[2].rand()

""
ids

""
gp.get_data(geo_data_stoch_init, numeric=True)


###############################################################################
# ## Deterministic functions

@pymc.deterministic(trace=True)
def input_data(value = 0, 
               interface_Z_modifier = interface_Z_modifier,
               geo_data_stoch_init = geo_data_stoch_init,
               ids = ids,
               verbose=1):
    # First we extract from our original intep_data object the numerical data
    #that is necessary for the interpolation.
    
    # geo_data_stoch is a pandas Dataframe
    geo_data_stoch = gp.get_data(geo_data_stoch_init, numeric=True)
    
    # Now we loop each id which share the same uncertainty variable. In this case, each layer.
    for e, i in enumerate(ids):
        # We add the stochastic part to the initial value
        #new_val = (geo_data_stoch_init.interfaces.loc[i, "Z"] + interface_Z_modifier[num])
        # print(new_val)
        #interp_data.geo_data_res.interfaces.iat[i, 2] = new_val
        geo_data_stoch.xs('interfaces').loc[i, 'Z'] += np.array(interface_Z_modifier[e])
        
        
    if verbose > 0:
        print(geo_data_stoch)
        print(interp_data.geo_data_res.interfaces[["X", "Y", "Z"]].values)
        
    # then return the input data to be input into the modeling function. Due to the way pymc2 stores the traces
    # We need to save the data as numpy arrays
    return  [geo_data_stoch.xs('interfaces')[["X", "Y", "Z"]].values,
             geo_data_stoch.xs('orientations').values]


""
# # %matplotlib notebook
@pymc.deterministic(trace=True)
def gempy_model(value=0,
                input_data=input_data, verbose=False):
    
    # modify input data values accordingly
    interp_data.geo_data_res_no_basement.interfaces[["X", "Y", "Z"]] = input_data[0]
    
    # Gx, Gy, Gz are just used for visualization. The theano function gets azimuth dip and polarity!!!
    interp_data.geo_data_res_no_basement.orientations[["G_x", "G_y", "G_z", "X", "Y", "Z", 'dip', 'azimuth', 'polarity']] = input_data[1]
    print(interp_data.geo_data_res_no_basement.interfaces[["X", "Y", "Z"]])
    try:
        # try to compute model
        lb, fb = gp.compute_model(interp_data)
        if verbose:
            gp.plot_section(interp_data.geo_data_res, lb[0], 5, plot_data=True)
        #    gp.plot_data(interp_data.geo_data_res, direction='y')

        return lb, fb
    
    except np.linalg.linalg.LinAlgError as err:
        # if it fails (e.g. some input data combinations could lead to 
        # a singular matrix and thus break the chain) return an empty model
        # with same dimensions (just zeros)
        if verbose:
            print("Exception occured.")
        return np.zeros_like(lith_block), np.zeros_like(fault_block)


""
@pymc.deterministic(trace=True)
def gempy_surfaces(value=0, gempy_model=gempy_model):
    vert, simp = gp.get_surfaces(interp_data, gempy_model[0][1], gempy_model[1][1], original_scale=True)
    return vert


###############################################################################
# ## Likelihood functions

###############################################################################
# ## pymc Model

# We add all the pymc objects to a list
params = [input_data, gempy_model, gempy_surfaces, *interface_Z_modifier] 

# We create the pymc model i.e. the probabilistic graph
model = pymc.Model(params)

###############################################################################
# Create pymc probabilistic graph plot:

graph = pymc.graph.dag(model)
graph.write_png('figures/paper_graph.png')

###############################################################################
# Running inference:

runner = pymc.MCMC(model, db="hdf5", dbname="pymc-db/UQ.hdf5")

""
iterations = 5

""
runner.use_step_method(pymc.AdaptiveMetropolis, params, delay=10)
np.random.seed(34)
runner.sample(iterations, tune_interval=1000, tune_throughout=True, )

###############################################################################
# Computing the initial model:

lb, fb = gp.compute_model(interp_data)

""
v_l, s_l = gp.get_surfaces(interp_data, lb[1], fb[1], original_scale=True)

""
geo_data.orientations

""
gp.plotting.plot_surfaces_3D(geo_data, v_l, s_l, alpha=1)

""
vertices = runner.trace("gempy_surfaces")[:]

###############################################################################
# Reading traces:

runner.trace("gempy_model")[0][0]

""
runner.trace("gempy_model")[1][0]

""
runner.trace('input_data')[0]

""
runner.trace('input_data')[3]

""
interp_data.geo_data_res.orientations

""
(runner.trace("gempy_model")[1][1] == runner.trace("gempy_model")[4][1]).sum(axis=1)

""
gp.plotting.plot_scalar_field(geo_data, runner.trace("gempy_model")[2][0][1], 5,
                              N=500, cmap='viridis')


""
# Function to plot the traces with the vertices
def plot_iterline(i, l, color="black", lw=0.5):
    f = vertices[i, l][:,1]==np.min(vertices[i, l][:,1])
   # f = a
    
    points = vertices[i, l][f]
    plt.plot(points[::5,0], points[::5,2], '-',
             #linestyle="solid",
             linewidth=lw, color=color, alpha=0.3)



""
# Some plotting options
params = {
    'axes.labelsize': 6,
    'font.size': 6,
    'legend.fontsize': 10,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'text.usetex': False,
    "axes.linewidth": 0.75,
    'xtick.major.size': 2,
    'xtick.major.width': 0.75,
    'ytick.major.size': 2,
    'ytick.major.width': 0.75,
}


def get_figsize(scale, textwidth=522, ratio=None):                      # Get this from LaTeX using \the\textwidth
    """Source: http://bkanuka.com/articles/native-latex-plots/"""
    inches_per_pt = 1.0 / 72.27                             # Convert pt to inch
    if ratio == None:
        ratio = (np.sqrt(5.0)-1.0)/2.0                    # Aesthetic ratio (you could change this)
    fig_width = textwidth * inches_per_pt * scale           # width in inches
    fig_height = fig_width * ratio                    # height in inches
    fig_size = [fig_width, fig_height]
    return fig_size

from matplotlib import rcParams
rcParams.update(params)

""
fig = plt.figure(figsize=get_figsize(0.75))
ax = plt.subplot()
    
for i in range(0,100,2):
    plot_iterline(i, 3, color="green")
    plot_iterline(i, 2, color="orange")
    plot_iterline(i, 1, color="red")
    # plot_iterline(i, 0, color="black", lw=2)
    
ax.set_xlabel("x")

plt.legend()
plt.ylabel("z")
plt.xlim(0,19)
ax.set_xticks(np.arange(0, 20, 5))
plt.ylim(-10,-0)
plt.plot([2., 13.3], [0,-10], color="gray", linewidth=4)
plt.grid(False)

#plt.title("Stochastic simulation of layer interfaces");

#plt.savefig("doc/figs/UQ.pdf", facecolor="white")

""
fig = plt.figure(figsize=get_figsize(0.75))
ax = plt.subplot()
    
for i in range(0,100,2):
    plot_iterline(i, 3, color="green")
    plot_iterline(i, 2, color="orange")
    plot_iterline(i, 1, color="red")
    # plot_iterline(i, 0, color="black", lw=2)
    
ax.set_xlabel("x")

plt.legend()
plt.ylabel("z")
plt.xlim(0,19)
ax.set_xticks(np.arange(0, 20, 5))
plt.ylim(-10,-0)
plt.plot([2., 13.3], [0,-10], color="gray", linewidth=4)
plt.grid(False)

#plt.title("Stochastic simulation of layer interfaces");

#plt.savefig("doc/figs/UQ.pdf", facecolor="white")

""
gp.plot_section(geo_data, runner.trace("gempy_model")[:][:,0,0,:][30], 5)
plt.savefig('original_model.pdf')

""
p_i = gp.pa.calculate_probability_lithology(runner.trace("gempy_model")[:][:,0,0,:])

""
gp.plot_section(geo_data,p_i[3], 5, cmap='viridis', norm=None)
plt.colorbar(orientation='horizontal')
#plt.savefig("doc/figs/prob.pdf", facecolor="white")

""
e = gp.pa.calcualte_ie_masked(p_i)

""
gp.plot_section(geo_data,e, 5, cmap='magma', norm=None)
plt.colorbar(orientation='horizontal')


#plt.savefig("doc/figs/entropy.pdf", facecolor="white")

###############################################################################
# ## Create syntetic gravity for likelihoods

geo_data_g = gp.create_data([-10e3,30e3,-10e3,20e3,-10e3,0],[50,50,50],
                         path_f = "input_data/paper_Orientations.csv",
                         path_i = "input_data/paper_Points.csv")
geo_data_g.add_interface(X=10, Y=4, Z=-7, formation='fault1')

gp.set_series(geo_data_g, {'fault_serie1': 'fault1','younger_serie' : 'Unconformity', 'older_serie': ('Layer1', 'Layer2')},
              order_formations= ['fault1', 'Unconformity', 'Layer2', 'Layer1'], verbose=2)

geo_data_g.modify_interface(9, Z = -6.4)

geo_data_g.interfaces[['X', 'Y', 'Z']] = geo_data_g.interfaces[['X', 'Y', 'Z']]*1000
geo_data_g.orientations[['X', 'Y', 'Z']] = geo_data_g.orientations[['X', 'Y', 'Z']]*1000

# Theano compilation
interp_data_g = gp.InterpolatorData(geo_data_g, u_grade=[3, 3, 3], dtype='float64', verbose=[],  output='gravity')

###############################################################################
# Change the initial data for the iteration that we want to compute

trace = 72
# modify input data values accordingly
interp_data_g.geo_data_res.interfaces[["X", "Y", "Z"]] = runner.trace("input_data")[72][0]

# Gx, Gy, Gz are just used for visualization. The theano function gets azimuth dip and polarity!!!
interp_data_g.geo_data_res.orientations[["X", "Y", "Z", 'dip', 'azimuth', 'polarity']] = runner.trace("input_data")[15][1]



""
# Set the specific parameters for the measurement grid of gravity:
gp.set_geophysics_obj(interp_data_g,  
                      [0.1e3,19.9e3,.1e3,.9e3, -10e3, 0], # Extent
                      [30,20])                            # Resolution 

# Setting desity and precomputations 
t = gp.precomputations_gravity(interp_data_g, 25,
                         [2.92e6, 3.1e6, 2.61e6, 2.92e6])

""
lith, fault, grav = gp.compute_model(interp_data_g, output='gravity')

""
# grav_u = np.load('real_grav.npy')

""
gp.plot_section(geo_data_g, lith[0], 5, direction='z',plot_data=True)
#annotate_plot(gp.get_data(geo_data_g, verbosity=2), 'annotations', 'X', 'Z', size = 20)
# ax = plt.gca()
# ax.set_xticks(np.linspace(0, 20, 50))
# ax.set_yticks(np.linspace(0, -10, 50))
#plt.grid()
fig = plt.gcf()
ax = plt.gca()
p = ax.imshow(grav.reshape(20,30), cmap='viridis', origin='lower', alpha=0.8, extent=[0,20e3,0,10e3])
# plt.xlim(-2e3,22e3)
# plt.ylim(-2e3,12e3)

plt.xlim(-2e3,22e3)
plt.ylim(-2e3,12e3)

plt.colorbar(p, orientation='horizontal')
#plt.savefig('Initial_grav.pdf')

""
gp.plot_section(geo_data_g, lith[0], 5, direction='z',plot_data=True)

plt.grid()
fig = plt.gcf()
ax = plt.gca()
p = ax.imshow(grav.reshape(20,30), cmap='viridis', origin='lower', alpha=0.8, extent=[0,20e3,0,10e3])
# plt.xlim(-2e3,22e3)
# plt.ylim(-2e3,12e3)

plt.xlim(-10e3,30e3)
plt.ylim(-10e3,20e3)

plt.colorbar(p)

""
# np.save('real_grav', grav)

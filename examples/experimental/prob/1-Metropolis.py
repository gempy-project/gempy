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
import pymc3 as pm

""
path_to_data = os.pardir+"/../data/input_data/jan_models/"

geo_data = gp.create_data([0,1000,0,1000,0,1000],resolution=[20,2,200], 
                        path_o = path_to_data + "model1_orientations.csv",
                        path_i = path_to_data + "model1_surface_points.csv") 

""
gp.plot.plot_data(geo_data)

""
gp.set_interpolation_data(geo_data, theano_optimizer='fast_compile')

""
gp.compute_model(geo_data)

""
gp.plot.plot_section(geo_data, 1, show_data=True)

""
# Column 10 of the cross-section above
well = geo_data.solutions.lith_block.reshape(geo_data.grid.regular_grid.resolution)[10,1,:]
np.round(well)

""
# Computing the thickness in meters
thickness = (np.round(well) == 2).sum()*geo_data.grid.regular_grid.dz
thickness

""
g = geo_data.surface_points.df.groupby('series')
g.groups

""
geo_data.modify_surface_points(g.groups[1], Z = 500)


"""
### Simple substraction likelihood. No GemPy involved:
"""

with pm.Model() as model:
    r2 = pm.Normal('rock2', 600, 50)
    r1 = pm.Normal('rock1', 400, 50)
   
    mu = pm.Deterministic('mu', r2-r1)
    a = pm.Normal('y', mu=mu, sd=20, observed=[200])
    trace = pm.sample(10000,
                      step = pm.Metropolis(),
                      compute_convergence_checks=True)

""
pm.traceplot(trace)


###############################################################################
# ### Creating custom theano functions
#
# Pymc3 only allows to use theano functions. GemPy has been written in theano - among other - for this particular reason. However, theano allows to create custom ops and hence call external functions. For this first notebook lets try to do so for a simple likelihood thickness:

def thickness(l2, l1):
    geo_data.modify_surface_points(g.groups[1], Z = l2)
    geo_data.modify_surface_points(g.groups[2], Z = l1)
    #gp.compute_model(geo_data)
    well = geo_data.solutions.lith_block.reshape(geo_data.grid.regular_grid.resolution)[10,1,:]
    thickness = (well == 2).sum()*geo_data.grid.regular_grid.dz
    return thickness
    
    

""
# %%timeit
# Testing the most comvoluted way ever to make a substraction
thickness(800, 200)

###############################################################################
# Now we need to create a theano op that performs the function above:

import theano.tensor as tt

class MuFromTheta(tt.Op):
    itypes = [tt.dvector]
    otypes = [tt.dscalar]

    def perform(self, node, inputs, outputs):
        theta, = inputs
        mu = thickness(theta[0], theta[1])
        outputs[0][0] = np.array(mu)

# The method allows for passing custom gradients too but we are not going to go down that road quite yet
    
#     def grad(self, inputs, g):
#         theta, = inputs
#         mu = self(theta)
#         thetamu = theta * mu
#         return [- g[0] * mu ** 2 / (1 + thetamu + tt.exp(-thetamu))]


""
tt_mu_from_theta = MuFromTheta()

""
ff = tt_mu_from_theta(theta)
mu.tag.test_value

""
# %%timeit
with pm.Model() as model:
    r2 = pm.Normal('rock2', 600, 50)
    r1 = pm.Normal('rock1', 400, 50)
    theta = tt.as_tensor_variable([r2, r1])
    mu = pm.Deterministic('mu', tt_mu_from_theta(theta))
    a = pm.Normal('y', mu=mu, sd=20, observed=[200])

""
with pm.Model() as model:
    r2 = pm.Normal('rock2', 600, 50)
    r1 = pm.Normal('rock1', 400, 50)
    theta = tt.as_tensor_variable([r2, r1])
    mu = pm.Deterministic('mu', tt_mu_from_theta(theta))
    a = pm.Normal('y', mu=mu, sd=20, observed=[200])
    trace = pm.sample(200, step = pm.Metropolis(), tune=50, cores=1, compute_convergence_checks=False)


""
pm.traceplot(trace)

""
pm.plot_posterior(trace)

###############################################################################
# ## Compiling gempy with PyMC3

import theano
theano.config.compute_test_value = 'ignore'

""
i = geo_data.interpolator.get_python_input_block()

""
geo_model_T = theano.OpFromGraph(geo_data.interpolator.theano_graph.input_parameters_loop,
                               geo_data.interpolator.theano_graph.compute_series(), inline=False,
                                 on_unused_input='warn',
                               name='geo_model')

""
rf = geo_data.rescaling.df.loc['values', 'rescaling factor']
centers = geo_data.rescaling.df.loc['values', 'centers']

""
# This is the new value for rock2
z_rock2 = 700

# We need to rescale
(z_rock2 - centers[2]) / rf + 0.5001

""
# Now we need to change the input of the z of rock 2
i[4][g.groups[1], 2] = (z_rock2 - centers[2]) / rf + 0.5001
i[4]

""
import theano
import theano.tensor as tt
theano.config.compute_test_value = 'warn'
# We convert a python variable to theano.shared
input_sh = []
for ii in i:
    input_sh.append(theano.shared(ii))

with pm.Model() as model:
    r2 = pm.Normal('rock2', 600, 50)
    r1 = pm.Normal('rock1', 400, 50)
    val2 = (r2 - centers[2]) / rf + 0.5001
    val1 = (r1 - centers[2]) / rf + 0.5001

    input_sh[4] = tt.set_subtensor(input_sh[4][g.groups[1], 2], val2)
    input_sh[4] = tt.set_subtensor(input_sh[4][g.groups[2], 2], val1)
    
    # we have to take the sol 0
    geo = geo_model_T(*input_sh)[0][0][0:8000]
    well = geo.reshape(geo_data.grid.regular_grid.resolution)[10,1,:]
    thickness = pm.Deterministic('thickness', tt.sum(tt.eq(well, 2)) * geo_data.grid.regular_grid.dz)
    thickness.name = 'thickness'
    a = pm.Normal('y', mu=thickness, sd=20, observed=120)
    b = pm.Metropolis()
    trace = pm.sample(10000, chains=1,
                     step =b,
                     compute_convergence_checks=True, live_plot=False)

""
pm.traceplot(trace)

""
a

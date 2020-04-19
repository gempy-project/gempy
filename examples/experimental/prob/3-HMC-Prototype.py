# -*- coding: utf-8 -*-
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


""
path_to_data = os.pardir+"/../data/input_data/jan_models/"

geo_data = gp.create_data([0,1000,0,1000,0,1000],resolution=[50,50,50], 
                        path_o = path_to_data + "model1_orientations.csv",
                        path_i = path_to_data + "model1_surface_points.csv") 

""
gp.plot.plot_data(geo_data)

""
dz = geo_data.grid.regular_grid.dz
dz

""
geo_data.surfaces.add_surfaces_values([0, 5, 0])

""
gp.set_interpolation_data(geo_data, theano_optimizer='fast_run', gradient=True)

""
gp.compute_model(geo_data, compute_mesh=False)


""
gp.compute_model(geo_data)
gp.plot.plot_section(geo_data, 1, show_data=True)

"""
## Compiling gempy with PyMC3
"""

import theano
import theano.tensor as T
theano.config.compute_test_value = 'ignore'

""
i = geo_data.interpolator.get_python_input_block()

""
geo_model_T = theano.OpFromGraph(geo_data.interpolator.theano_graph.input_parameters_loop,
                               geo_data.interpolator.theano_graph.compute_series(), inline=False,
                                 on_unused_input='ignore',
                               name='geo_model')

""
respect = geo_data.interpolator.theano_graph.input_parameters_loop[4]
m = geo_data.interpolator.theano_graph.compute_series()[0][1][0:125000]
w = m.reshape(geo_data.grid.regular_grid.resolution)[10,1,:]

""
th_f = theano.function(geo_data.interpolator.theano_graph.input_parameters_loop,
                         w,  on_unused_input='ignore')

""
gempy = th_f(*geo_data.interpolator.get_python_input_block())
plt.plot(gempy)

""
gempy.sum()

""
theano.config.compute_test_value = 'ignore'

th_f_g = theano.function(geo_data.interpolator.theano_graph.input_parameters_loop,
                         T.grad(w.sum(), 
                                respect), on_unused_input='ignore')

""
gempy_g = th_f_g(*geo_data.interpolator.get_python_input_block())
# 107 ms ± 395 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)


""
gempy_g

""
plt.plot(gempy_g[:, 2], 'o')

""
# geomodel rescaling parameters
rf = geo_data.rescaling.df.loc['values', 'rescaling factor']
centers = geo_data.rescaling.df.loc['values', 'centers']

# df group
g = geo_data.surface_points.df.groupby('id')

""
# Converting python numbers to theano shared
input_sh = []
for ii in i:
    input_sh.append(theano.shared(ii))

""
geo_data.interpolator.theano_graph.input_parameters_loop

""

theano.config.compute_test_value = 'ignore'

geo_model_T = theano.OpFromGraph(geo_data.interpolator.theano_graph.input_parameters_loop,
                               [geo_data.interpolator.theano_graph.compute_series()[0][1][0:125000]], inline=True,
                                 on_unused_input='warn',
                               name='geo_model')

""
geo_data.grid.regular_grid.resolution

""
import theano

import theano.tensor as tt
theano.config.compute_test_value = 'warn'
# We convert a python variable to theano.shared
input_sh = []
for ii in i:
    input_sh.append(theano.shared(ii))

with pm.Model() as model2:
    r2 = pm.Normal('rock2', 600, 50)
    r1 = pm.Normal('rock1', 400, 50)
    val2 = (r2 - centers[2]) / rf + 0.5001
    val1 = (r1 - centers[2]) / rf + 0.5001

    input_sh[4] = tt.set_subtensor(input_sh[4][g.groups[1], 2], val2)
    input_sh[4] = tt.set_subtensor(input_sh[4][g.groups[2], 2], val1)
    
    # we have to take the sol 0
    geo = geo_model_T(*input_sh)
    well = geo.reshape((50, 50, 50))[25,25,:]
   # thickness = pm.Deterministic('thickness', well.sum())
   # thickness.name = 'thickness'
    thickness = well.sum()
    a = pm.Normal('y', mu=thickness, sd=20, observed=120)
 #   b = pm.Metropolis()
    trace = pm.sample(1000, chains=1, tune=500,
    #                step =b,
                    compute_convergence_checks=True)
         #   live_plot=True)

""
pm.traceplot(trace)

""
trace.tree_size.sum()

""
trace.get_sampler_stats('mean_tree_accept')

""
(7*60+7)/2738

""


""


""
pm.traceplot(trace)

###############################################################################
# 1.62 iterations per second - Macbook
#
# 2.51 it/s - Ceres pre-optimization
#
# 3,73 it/s - Post-optimization

with model2:
    inference = pm.ADVI()
    approx = pm.fit(n=3000, method=inference)

""
trace = approx.sample(draws=5000)
pm.traceplot(trace)

""
model2.logpt

""
model2.profile(model2.logpt).summary()

""
model2.profile(pm.gradient(model2.logpt, model2.vars)).summary()

""


"""
6.3 - Joint inversion (preview).
================================
"""

 # These two lines are necessary only if GemPy is not installed
import sys, os
sys.path.append("../../gempy")
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cuda"

# Importing GemPy
import gempy as gp
from examples.tutorials.ch5_probabilistic_modeling.aux_functions.aux_funct import plot_geo_setting

# Embedding matplotlib figures in the notebooks
# %matplotlib inline

# Importing auxiliary libraries
import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm
import arviz as az


# %%
# Model definition
# ----------------
# 
# In the previous example we assume constant thickness to be able to
# reduce the problem to one dimension. This keeps the probabilistic model
# fairly simple since we do not need to deel with complex geometric
# structures. Unfortunaly, geology is all about dealing with complex three
# dimensional structures. In the moment data spread across the physical
# space, the probabilistic model will have to expand to relate data from
# different locations. In other words, the model will need to include
# either interpolations, regressions or some other sort of spatial
# functions. In this paper, we use an advance universal co-kriging
# interpolator. Further implications of using this method will be discuss
# below but for this lets treat is a simple spatial interpolation in order
# to keep the focus on the constraction of the probabilistic model.
# 

# %%
# This is to make it work in sphinx gallery
cwd = os.getcwd()
if not 'examples' in cwd:
    path_dir = os.getcwd() + '/examples/tutorials/ch5_probabilistic_modeling'
else:
    path_dir = cwd

# %%
geo_model = gp.load_model(r'2-layers', path=path_dir, recompile=True)

# %%
gp.compute_model(geo_model)
plot_geo_setting(geo_model)


# %% 
# geo_model = gp.create_model('2-layers')
# gp.init_data(geo_model, extent=[0, 12e3, -2e3, 2e3, 0, 4e3], resolution=[500, 1, 500])
#
# # %%
# geo_model.add_surfaces('surface 1')
# geo_model.add_surfaces('surface 2')
# geo_model.add_surfaces('basement')
# dz = geo_model.grid.regular_grid.dz
# geo_model.surfaces.add_surfaces_values([dz, 0, 0], ['dz'])
# geo_model.surfaces.add_surfaces_values(np.array([2.6, 2.4, 3.2]), ['density'])
#
# # %%
# geo_model.add_surface_points(3e3, 0, 3.05e3, 'surface 1')
# geo_model.add_surface_points(9e3, 0, 3.05e3, 'surface 1')
#
# geo_model.add_surface_points(3e3, 0, 1.02e3, 'surface 2')
# geo_model.add_surface_points(9e3, 0, 1.02e3, 'surface 2')
#
# geo_model.add_orientations(  6e3, 0, 4e3, 'surface 1', [0,0,1])
#
# # %%
# gp.plot.plot_data(geo_model)

# %% 
device_loc = np.array([[6e3, 0, 3700]])

# %% 
geo_model.set_centered_grid(device_loc,  resolution = [10, 10, 60], radius=4000)

# %%
gp.set_interpolator(geo_model, output=['gravity'], pos_density=2,  gradient=True,
                    theano_optimizer='fast_run')            

# %% 
gp.compute_model(geo_model, set_solutions=True, compute_mesh=False)

# %% 
geo_model.solutions.fw_gravity


# %%
# --------------
# 
# Bayesian Interence
# ~~~~~~~~~~~~~~~~~~
# 


# %%
# Setting gempy into a pymc function
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 

# %% 
geo_model._interpolator.theano_graph.sig_slope.set_value(150)

# %% 
geo_model._interpolator.theano_graph.input_parameters_loop[4]

# %% 
geo_model._interpolator.theano_graph.compute_type


# %%
# Test fw model gradient
# ^^^^^^^^^^^^^^^^^^^^^^
# 

# %% 
import theano
import theano.tensor as tt
theano.config.compute_test_value = 'ignore'
geo_model_T = theano.OpFromGraph(geo_model.interpolator.theano_graph.input_parameters_loop,
                                [theano.grad(geo_model.interpolator.theano_graph.theano_output()[12][0],
                                             geo_model.interpolator.theano_graph.input_parameters_loop[4])],
                                 inline=True,
                                 on_unused_input='ignore',
                                 name='forw_grav')

# %% 
i = geo_model.interpolator.get_python_input_block()
th_f = theano.function([], geo_model_T(*i), on_unused_input='warn')

# %% 
geo_model.interpolator.theano_graph.sig_slope.set_value(20)

# %% 
th_f()


# %%
# Setup Bayesian model
# --------------------
# 

# %% 
i = geo_model.interpolator.get_python_input_block()
theano.config.compute_test_value = 'ignore'
geo_model_T_grav = theano.OpFromGraph(geo_model.interpolator.theano_graph.input_parameters_loop,
                                [geo_model.interpolator.theano_graph.theano_output()[12]],
                                 inline=False,
                                 on_unused_input='ignore',
                                 name='forw_grav')

# %% 
geo_model_T_thick = theano.OpFromGraph(geo_model.interpolator.theano_graph.input_parameters_loop,
                                [geo_model.interpolator.theano_graph.compute_series()[0][1][0:250000]], inline=True,
                                 on_unused_input='ignore',
                                 name='geo_model')

# %% 
# We convert a python variable to theano.shared
input_sh = []
i = geo_model.interpolator.get_python_input_block()
for ii in i:
    input_sh.append(theano.shared(ii))

# We get the rescaling parameters:
rf = geo_model.rescaling.df.loc['values', 'rescaling factor'].astype('float32')
centers = geo_model.rescaling.df.loc['values', 'centers'].astype('float32')

# We create pandas groups by id to be able to modify several points at the same time:
g = geo_model.surface_points.df.groupby('id')
l = theano.shared(np.array([], dtype='float64'))

# %% 
g_obs_p = 1e3 * np.array([-0.3548658 , -0.35558686, -0.3563156 , -0.35558686, -0.3548658 ,
       -0.3534237 , -0.35201198, -0.3534237 , -0.3548658 , -0.3563401 ,
       -0.3548658 , -0.35558686, -0.3548658 , -0.3541554 , -0.3534569 ,
       -0.3527707 , -0.35424498, -0.35575098, -0.3572901 , -0.35575098,
       -0.35424498, -0.35575098, -0.35424498, -0.35575098, -0.35424498,
       -0.35575098, -0.35643718, -0.35713565, -0.35643718], dtype='float32')

y_obs_list = 1e3 * np.array([2.12, 2.06, 2.08, 2.05, 2.08, 2.09,
              2.19, 2.07, 2.16, 2.11, 2.13, 1.92])

# %% 
# Python input variables
i = geo_model.interpolator.get_python_input_block()
i

# %% 
# C/cuda input variables
input_sh

# %% 
## theano.config.compute_test_value = 'ignore'

with pm.Model() as model:
    
    depths = pm.Normal('depths', geo_model.surface_points.df['Z'],
                   np.array([200, 200, 200, 200]), shape=(4), dtype='float32')

    depths_r = (depths - centers[2])/rf + 0.5001
    
    input_sh[4] = tt.set_subtensor(input_sh[4][:, 2], depths_r)
    # input_sh[4] = depths_r
    grav = geo_model_T(*input_sh)
    geo = geo_model_T_thick(*input_sh)

    grav = pm.Deterministic('gravity', grav[0])
    well_1 = geo.reshape((500,1,500))[125, 0 ,:]
    well_2 = geo.reshape((500,1,500))[375, 0 ,:]

    thick_1 = pm.Deterministic('thick_1', well_1.sum())
    thick_2 = pm.Deterministic('thick_2', well_2.sum())
  
    sigma_grav = pm.Normal('sigma', mu = 250, sigma = 40)
    sigma_thick = pm.Gamma('sigma_thickness', mu = 300, sigma = 300)
    sigma_thick2 = pm.Gamma('sigma2_thickness', mu = 300, sigma = 300)

    obs_grav = pm.Normal('y', mu=grav, sd=sigma_grav, observed=g_obs_p)
    obs_thick_1 = pm.Normal('y2', mu=thick_1, sd=sigma_thick, observed=y_obs_list)
    obs_thick_2 = pm.Normal('y3', mu=thick_2, sd=sigma_thick2, observed=y_obs_list)



# %% 
pm.model_to_graphviz(model)


# %%
# Bayesian inference
# ~~~~~~~~~~~~~~~~~~
# 

# %% 
with model:
    trace = pm.sample(400, chains=1, tune=300,
                      #init='adapt_diag',
                      # trace= pm.backends.SQLite('Gravity1'),
                      discard_tuned_samples=False,
                      compute_convergence_checks=False,
                      )
   


# %%
# Predictive prior and posterior:
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 

# %% 
with model:
    prior = pm.sample_prior_predictive(1000)
    post = pm.sample_posterior_predictive(trace)


# %%
# Sampler stats:
# ^^^^^^^^^^^^^^
# 

# %% 
trace.get_sampler_stats('depth')

# %% 
trace.get_sampler_stats('step_size')

# %% 
trace.get_values('gravity')


# %% 
trace.get_values('depths')


# %%
# Save data:
# ~~~~~~~~~~
# 

# %% 
import arviz as az

data = az.from_pymc3(trace=trace,
                     prior=prior,
                     posterior_predictive=post)
data.to_netcdf('workshop1')

# %% 

data.posterior
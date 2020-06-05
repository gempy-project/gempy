"""
5.3 - Probability Density Transformation.
=========================================

"""

# Importing GemPy
import gempy as gp
from examples.tutorials.ch5_probabilistic_modeling.aux_functions.aux_funct import plot_geo_setting_well
from gempy.bayesian import plot_posterior as pp
from gempy.bayesian.plot_posterior import default_red, default_blue

# Importing auxiliary libraries
import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm
import arviz as az
import os

# %%
# Model definition
# ----------------
# 
# Same problem as before, let’s assume the observations are layer
# thickness measurements taken on an outcrop. Now, in the previous example
# we chose a prior for the mean arbitrarily:
# :math:`𝜇∼Normal(mu=10.0, sigma=5.0)`–something that made sense for these
# specific data set. If we do not have any further information, keeping
# an uninformative prior and let the data to dictate the final value of
# the inference is the sensible way forward. However, this also enable to
# add information to the system by setting informative priors.
# 
# Imagine we get a borehole with the tops of the two interfaces of
# interest. Each of this data point will be a random variable itself since
# the accuracy of the exact 3D location will be always limited. Notice
# that this two data points refer to depth not to thickness–the unit of
# the rest of the observations. Therefore, the first step would be to
# perform a transformation of the parameters into the observations space.
# Naturally in this example a simple subtraction will suffice.
# 
# Now we can define the probabilistic models:
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
plot_geo_setting_well(geo_model=geo_model)

# %%
y_obs = [2.12]
y_obs_list = [2.12, 2.06, 2.08, 2.05, 2.08, 2.09,
              2.19, 2.07, 2.16, 2.11, 2.13, 1.92]

np.random.seed(4003)

# %%
#
# # .. image:: /../../_static/computational_graph1.png
#

# %% 
with pm.Model() as model:
    mu_top = pm.Normal('$\mu_{top}$', 3.05, .2)
    sigma_top = pm.Gamma('$\sigma_{top}$', 0.3, 3)
    y_top = pm.Normal('y_{top}', mu=mu_top, sd=sigma_top, observed=[3.02])

    mu_bottom = pm.Normal('$\mu_{bottom}$', 1.02, .2)
    sigma_bottom = pm.Gamma('$\sigma_{bottom}$', 0.3, 3)
    y_bottom = pm.Normal('y_{bottom}', mu=mu_bottom, sd=sigma_bottom, observed=[1.02])

    mu_t = pm.Deterministic('$\mu_{thickness}$', mu_top - mu_bottom)
    sigma_thick = pm.Gamma('$\sigma_{thickness}$', 0.3, 3)
    y = pm.Normal('y_{thickness}', mu=mu_t, sd=sigma_thick, observed=y_obs_list)


# %%
# Sampling
# --------
# 

# %% 
with model:
    prior = pm.sample_prior_predictive(1000)
    trace = pm.sample(1000, discard_tuned_samples=False, cores=1)
    post = pm.sample_posterior_predictive(trace)

# %%
data = az.from_pymc3(trace=trace,
                     prior=prior,
                     posterior_predictive=post)
data

# %% 

az.plot_trace(data)
plt.show()

# %% 
# sphinx_gallery_thumbnail_number = 3
az.plot_density([data, data.prior], shade=.9, data_labels=["Posterior", "Prior"],
                var_names=[
                    '$\\mu_{top}$',
                    '$\\mu_{bottom}$',
                    '$\\mu_{thickness}$',
                    '$\\sigma_{top}$',
                    '$\\sigma_{bottom}$',
                    '$\\sigma_{thickness}$'
                ],
                colors=[default_red, default_blue], bw=5);
plt.show()

# %%

p = pp.PlotPosterior(data)

p.create_figure(figsize=(9, 5), joyplot=False, marginal=True, likelihood=False)
p.plot_marginal(var_names=['$\\mu_{top}$', '$\\mu_{bottom}$'],
                plot_trace=False, credible_interval=.70, kind='kde',
                marginal_kwargs={"bw": 1}
                )
plt.show()
# %%

p = pp.PlotPosterior(data)
p.create_figure(figsize=(9, 6), joyplot=True)

iteration = 500
p.plot_posterior(['$\\mu_{top}$', '$\\mu_{bottom}$'],
                 ['$\mu_{thickness}$', '$\sigma_{thickness}$'],
                 'y_{thickness}', iteration, marginal_kwargs={"credible_interval": 0.94,
                                                              'marginal_kwargs': {"bw": 1},
                                                              'joint_kwargs': {"bw": 1}})
plt.show()

# %%
az.plot_pair(data, divergences=False, var_names=['$\\mu_{top}$', '$\\mu_{bottom}$'])
plt.show()

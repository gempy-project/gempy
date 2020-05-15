"""
5 2 - Introduction PyMC3.
=========================

"""
# %%

# Importing GemPy
import gempy as gp

# Importing auxiliary libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm
import arviz as az

from gempy.bayesian import plot_posterior as pp
from importlib import reload
from matplotlib.ticker import StrMethodFormatter

# %%
# Model definition
# ----------------
# 

# %%
# This is to make it work in sphinx gallery
cwd = os.getcwd()
if not 'examples' in cwd:
    path_dir = os.getcwd()+'/examples/tutorials/ch5_probabilistic_modeling'
else:
    path_dir = cwd

# %%
geo_model = gp.load_model(r'/2-layers', path=path_dir, recompile=True)

# %%
geo_model.modify_surface_points(2, Z=1000)
gp.compute_model(geo_model)


# %%

def plot_geo_setting_well():
    device_loc = np.array([[6e3, 0, 3700]])
    p2d = gp.plot_2d(geo_model, show_topography=True, legend=False)

    well_1 = 3.41e3
    well_2 = 3.6e3
    p2d.axes[0].scatter([3e3], [well_1], marker='^', s=400, c='#71a4b3', zorder=10)
    p2d.axes[0].scatter([9e3], [well_2], marker='^', s=400, c='#71a4b3', zorder=10)
    p2d.axes[0].scatter(device_loc[:, 0], device_loc[:, 2], marker='x', s=400, c='#DA8886', zorder=10)

    p2d.axes[0].vlines(3e3, .5e3, well_1, linewidth=4, color='gray')
    p2d.axes[0].vlines(9e3, .5e3, well_2, linewidth=4, color='gray')
    p2d.axes[0].vlines(3e3, .5e3, well_1)
    p2d.axes[0].vlines(9e3, .5e3, well_2)
    p2d.axes[0].set_xlim(2900, 3100)
    plt.savefig('well.svg')
    plt.show()


# %%
plot_geo_setting_well()

# Thickness measurements
# ----------------------

# %% 
y_obs = [2.12]
y_obs_list = [2.12, 2.06, 2.08, 2.05, 2.08, 2.09,
              2.19, 2.07, 2.16, 2.11, 2.13, 1.92]
np.random.seed(4003)

# %%
# Normal-several points
# ~~~~~~~~~~~~~~~~~~~~~
# .. image:: /../../_static/computational_graph1.png
#

# %% 
with pm.Model() as model:
    mu = pm.Normal('$\mu$', 2.08, .07)
    sigma = pm.Gamma('$\sigma$', 0.3, 3)
    y = pm.Normal('$y$', mu, sigma, observed=y_obs_list)

# %% 
mu

# %% 
sigma

# %% 
y


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

# %% 
az.plot_trace(data)
plt.show()

# %%
# Raw observations:
# ^^^^^^^^^^^^^^^^^
# 

# %% 

reload(pp)
p = pp.PlotPosterior(data)
p.create_figure(figsize=(9, 3), joyplot=False, marginal=False)
p.plot_normal_likelihood('$\mu$', '$\sigma$', '$y$', iteration=-1, hide_bell=True)
p.likelihood_axes.set_xlim(1.90, 2.2)
p.likelihood_axes.xaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
for tick in p.likelihood_axes.get_xticklabels():
    tick.set_rotation(45)
plt.show()

# %%
# Final inference
# ^^^^^^^^^^^^^^^
# 

# %% 

reload(pp)
p = pp.PlotPosterior(data)
p.create_figure(figsize=(9, 3), joyplot=False, marginal=False)
p.plot_normal_likelihood('$\mu$', '$\sigma$', '$y$', iteration=-1, hide_lines=True)
p.likelihood_axes.set_xlim(1.70, 2.40)
p.likelihood_axes.xaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
for tick in p.likelihood_axes.get_xticklabels():
    tick.set_rotation(45)
plt.show()

# %%
# Joyplot
# ~~~~~~~
# 

# %% 
# %matplotlib inline
reload(pp)
p = pp.PlotPosterior(data)

p.create_figure(figsize=(9, 9), joyplot=True, marginal=False, likelihood=False, n_samples=31)
p.plot_joy(('$\mu$', '$\sigma$'), '$y$', iteration=14)
plt.show()

# %%
# Join probability
# ~~~~~~~~~~~~~~~~
# 

# %% 
# sphinx_gallery_thumbnail_number = 6
reload(pp)
p = pp.PlotPosterior(data)

p.create_figure(figsize=(9, 5), joyplot=False, marginal=True, likelihood=True)
p.plot_marginal(var_names=['$\mu$', '$\sigma$'],
                plot_trace=False, credible_interval=.93, kind='kde')

p.plot_normal_likelihood('$\mu$', '$\sigma$', '$y$', iteration=-1, hide_lines=True)
p.likelihood_axes.set_xlim(1.70, 2.40)
plt.show()

# %%
# Full plot
# ~~~~~~~~~
# 

# %%
reload(pp)
p = pp.PlotPosterior(data)

p.create_figure(figsize=(9, 5), joyplot=True, marginal=True, likelihood=True, n_samples=11)

p.plot_posterior(['$\mu$', '$\sigma$'], ['$\mu$', '$\sigma$'], '$y$', 1000,
                 marginal_kwargs={'plot_trace': False, 'credible_interval': .93, 'kind': 'kde'})
plt.show()
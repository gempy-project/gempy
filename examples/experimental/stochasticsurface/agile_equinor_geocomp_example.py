# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: sphinx
#       format_version: '1.1'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python [conda env:topology]
#     language: python
#     name: conda-env-topology-py
# ---

"""
# Stochastic Geomodelling in Python using GemPy
"""

###############################################################################
# ### Import

import warnings
warnings.filterwarnings('ignore')
import sys, os
sys.path.append("../../../")
import gempy as gp
import numpy as np
# %matplotlib inline
import matplotlib.pyplot as plt
import pickle
from gempy.utils import stochastic_surface as ss
from gempy.bayesian.posterior_analysis import calculate_probability_lithology, calculate_information_entropy
from gempy.assets import topology as tp
from tqdm import tqdm_notebook
import seaborn as sns

###############################################################################
# ### Create the Geomodel

geo_model = gp.create_model('Demo')
extent = [0,2000.,0,2000.,0,2000.]
gp.init_data(geo_model, extent,[50,50,50], 
      path_o = "simple_fault_model_orientations.csv",
      path_i = "simple_fault_model_points.csv", default_values=True); #%%

gp.map_series_to_surfaces(geo_model,
                            {"Fault_Series":'Main_Fault', 
                             "Strat_Series": ('Sandstone_2','Siltstone',
                                              'Shale', 'Sandstone_1', 'basement')}, remove_unused_series=True);

geo_model.set_is_fault(['Fault_Series'])

groups = pickle.load(open("surf_pts_groups.p", "rb"))
geo_model.surface_points.df["group"] = groups
geo_model.orientations.df["group"] = None

###############################################################################
# ## Visualizing the input data

gp.plot.plot_data(geo_model)

""
gp.plot.plot_data_3D(geo_model);

""
gp.set_interpolation_data(
    geo_model,
    compile_theano=True,
    theano_optimizer='fast_compile',
    verbose=[]
)

###############################################################################
# ### Compute the Geomodel

sol = gp.compute_model(geo_model, compute_mesh=True)

###############################################################################
# ### Plotting a section through the geomodel

gp.plot.plot_section(geo_model, 24)

###############################################################################
# ### Plotting the 3-D Geomodel 

gp.plot.plot_3D(geo_model, render_data=False)

""
from gempy.assets import topology as tp
graph_init, ctrs_init = tp.compute_topology(geo_model, 24, filter_rogue=True)

""
gp.plot.plot_section(geo_model, 24) 
gp.plot.plot_topology(geo_model, graph_init, ctrs_init)

###############################################################################
# ### Modeling horizon and fault uncertainties

smod.reset()

###############################################################################
# #### Uncertainty parametrization

stochastic_surfaces = []
surface_names = np.unique(geo_model.surface_points.df.group)

for surface in surface_names:
    ssurf = ss.StochasticSurfaceScipy(geo_model, surface, grouping="group")
    
    if "Fault" in surface:
        ssurf.parametrize_surfpts_individual(50, direction="X")
    else:
        ssurf.parametrize_surfpts_single(30, direction="Z")
        
    stochastic_surfaces.append(ssurf)
    
smod = ss.StochasticModel(geo_model, stochastic_surfaces)

###############################################################################
# #### Uncertainty simulation

sols = []
graphs = []

for i in tqdm_notebook(range(10)):
    smod.sample()
    smod.modify()
    
    gp.compute_model(geo_model, compute_mesh=True)
    
    g, ctrs = tp.compute_topology(geo_model, 24, filter_rogue=True)
    if len(g.nodes) in [9, 10, 11]:
        sols.append(geo_model.solutions.block_matrix)
        graphs.append(g)
    
sols = np.array(sols)

""
###############################################################################
# np.save("sols.npy", sols)

###############################################################################
# ### Visualizing Uncertainty

sols = np.load("sols.npy")

""
# grab the right data
lbs = sols[:, 1, 0, :].astype(int)
fbs = sols[:, 0, 0, :].astype(int)

# lithology probabilities
lb_probs = calculate_probability_lithology(lbs)
fb_probs = calculate_probability_lithology(fbs)
# information entropy
lb_ie = calculate_information_entropy(lb_probs)
fb_ie = calculate_information_entropy(fb_probs)

""
fig, axs = plt.subplots(ncols=2)

ies = [lb_ie, fb_ie]

vmin, vmax = np.min(ies), np.max(ies)
imkwargs = dict(origin="lower", cmap="inferno", vmin=vmin, vmax=vmax)

for ie, ax in zip(ies, axs):
    im = ax.imshow(ie.reshape(50, 50, 50)[:, 24, :].T, **imkwargs)


###############################################################################
# ### Estimating reservoir volume uncertainty

def calc_vol(sols, n):
    """Calculate the volumes for a specified layer
    across all geomodels."""
    vols = []
    for sol in sols:
        lb = sol[1].astype(int)
        vol = np.count_nonzero(lb==n)
        vols.append(vol)
    return vols


""
lb = sols[0][0].astype(int)
volumes = calc_vol(sols, 3)
ax = sns.distplot(volumes, bins=24)
ax.set_xlabel("Volume [$m^3$]")
ax.set_ylabel("Probability Density")
ax.set_title("Reservoir Volume Uncertainty")

""


""


""


""


""


""


""


""


""


""


""


""


""


""


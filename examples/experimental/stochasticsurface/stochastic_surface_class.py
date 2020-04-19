# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: sphinx
#       format_version: '1.1'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python [conda env:gempy]
#     language: python
#     name: conda-env-gempy-py
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

from typing import Tuple
import pandas as pd


""
def generate(n_interfaces: int, seed=42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    surface_points = pd.DataFrame()
    orientations = pd.DataFrame()
    
    for n in range(n_interfaces):
        x = np.random.rand(15) * 3
        y = np.random.rand(15)
        
        def func(x, y):
            return np.sin(x) + np.cos(y)
        
        z = func(x, y) + n
        
        surfpts = pd.DataFrame({"X": x, "Y": y, "Z": z})
        surfpts["formation"] = "Interface {}".format(n + 1)
        surface_points = surface_points.append(surfpts)
        
        orientpts = pd.DataFrame(
            {"X": [np.mean(surfpts.X)], 
             "Y": [np.mean(surfpts.Y)], 
             "Z": [np.mean(surfpts.Z)], 
             "azimuth": [90], "dip": [3], 
             "polarity": [1], "formation": ["Interface {}".format(n + 1)]})
        orientations = orientations.append(orientpts)
    return surface_points, orientations
    
surface_points, orientations = generate(3, seed=41)

""
surface_points.to_csv("surfpts.csv", index=False)
orientations.to_csv("orientpts.csv", index=False)

""
###############################################################################
# np.random.seed(42)
# x = np.random.rand(30) * 3
# y = np.random.rand(30)
#
# def func(x, y):
#     return np.sin(x) + np.cos(y)
#
# z = func(x, y)
#
#
#
# surfpts = pd.DataFrame({"X": x, "Y": y, "Z": z})
# surfpts["formation"] = "Interface 1"
# surfpts.to_csv("surfpts.csv", index=False)
#
# orientpts = pd.DataFrame({"X": [1.5], "Y": [0.5], "Z": [1.7], "azimuth": [90], "dip": [3], "polarity": [1], "formation": ["Interface 1"]})
# orientpts.to_csv("orientpts.csv", index=False)

""
###############################################################################
# import ipyvolume as ipv
# ipv.figure()
#
# ipv.scatter(x, y, z)
# ipv.show()

""
geo_model = gp.create_model('StochSurfTesting')
gp.init_data(geo_model, [surface_points.min().X * 0.95, surface_points.max().X * 1.05, 
                         surface_points.min().Y * 0.95, surface_points.max().Y * 1.05, 
                         surface_points.min().Z * 0.2, surface_points.max().Z * 1.2], 
             [50,50,50], path_i="surfpts.csv", path_o="orientpts.csv") 

""
gp.map_series_to_surfaces(geo_model, 
                            {"Default series": ["Interface {}".format(n + 1) for n in range(3)]})

""
gp.plot.plot_data(geo_model)

""
gp.set_interpolation_data(
    geo_model, output='geology', compile_theano=True, theano_optimizer='fast_compile'
)

""
sol = gp.compute_model(geo_model, compute_mesh=False)

""
gp.plot.plot_section(geo_model, 24)

""
###############################################################################
# gp.plot.plot_surfaces_3d_ipv(geo_model)

""
from gempy.utils import stochastic_surface as ss

""
surfaces = ["Interface {}".format(n + 1) for n in range(3)]

stochastic_surfaces = []
for surface in surfaces:
    ssurf = ss.StochasticSurfaceScipy(geo_model, surface)
    ssurf.parametrize_surfpts_single(0.15)
    stochastic_surfaces.append(ssurf)
    
smod = ss.StochasticModel(geo_model, stochastic_surfaces)

""
smod.sample()

""
smod.modify()

""
sol = gp.compute_model(geo_model, compute_mesh=True)
# gp.plot.plot_surfaces_3d_ipv(geo_model)

""
gp.plot.plot_section(geo_model, 24)

""


""


""


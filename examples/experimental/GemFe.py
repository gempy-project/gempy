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

import sys, os
sys.path.append("/Users/varga/PycharmProjects/surfe/_buildv15_3")

import surfepy as sp

 # These two lines are necessary only if GemPy is not installed
import sys, os
sys.path.append("../..")

# Importing GemPy
import gempy as gp

# Embedding matplotlib figures in the notebooks
# %matplotlib inline

# Importing auxiliary libraries
import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm

""
gp

""
path_to_data = os.pardir+"/data/input_data/jan_models/"

geo_data = gp.create_data([0,1000,0,1000,0,1000],resolution=[50,1,50], 
                        path_o = path_to_data + "model3_orientations.csv",
                        path_i = path_to_data + "model3_surface_points.csv") 

""
interfaces = geo_data.surface_points.df[['X', 'Y', 'Z', 'id']].values.astype(float)
#interfaces[:,3] = 0
interfaces

""
s = sp.Surfe_API(2)

""
s.SetRBFShapeParameter(1)

""
s.SetInterfaceConstraints(interfaces)

""
for i in interfaces:
    s.AddInterfaceConstraint(*i)

""
s.GetInterfaceConstraints()

""
orientations = geo_data.orientations.df[['X', 'Y', 'Z', 'G_x', 'G_y', 'G_z']].values
orientations

""
s.SetPlanarConstraints(orientations)
s.GetPlanarConstraints()

""
s.ComputeInterpolant()

""
sol_scalar = []
for i in geo_data.grid.values:
    sol_scalar.append(s.EvaluateInterpolantAtPoint(*i))

""
plt.contourf(np.array(sol_scalar).reshape(50,50).T, cmap='viridis')

""
gp.set_interpolation_data(geo_data)

""
gp.compute_model(geo_data)

""
gp.plot.plot_scalar_field(geo_data, 0)

""


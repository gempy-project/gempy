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

# These two lines are necessary only if gempy is not installed
import sys, os
sys.path.append("../..")

import theano
theano.config.optimizer = 'fast_run'

# Importing gempy
import gempy as gp

# Embedding matplotlib figures into the notebooks
# %matplotlib inline

# Aux imports
import numpy as np
import matplotlib.pyplot as plt
print(gp.__version__)

""
geo_model = gp.create_model('Greenstone')

""
# Importing the data from csv files and settign extent and resolution
geo_model = gp.init_data(geo_model, [696000,747000,6863000,6930000,-20000, 200],[50, 50, 50],
                         path_o = os.pardir+"/data/input_data/tut_SandStone/SandStone_Foliations.csv",
                         path_i = os.pardir+"/data/input_data/tut_SandStone/SandStone_Points.csv")

""
gp.plot.plot_data(geo_model, direction='z')

""
gp.map_series_to_surfaces(geo_model, {"EarlyGranite_Series": 'EarlyGranite', 
                         "BIF_Series":('SimpleMafic2', 'SimpleBIF'),
                          "SimpleMafic_Series":'SimpleMafic1', 'Basement':'basement'})

""
geo_model.add_surface_values([2.61,2.92,3.1,2.92,2.61])

""
gp.set_interpolation_data(geo_model,
                          compile_theano=True,
                          theano_optimizer='fast_compile',
                          verbose=[])

""
gp.compute_model(geo_model, set_solutions=True)

""
gp.plot.plot_section(geo_model, -1, direction='z', show_data=False)

""
gp.plot.plot_section(geo_model, 25, direction='x')

""
geo_model.solutions.values_matrix

""
geo_model.rescaling.df['centers']

""
gp.plot.plot_section(geo_model, 25, block = geo_model.solutions.values_matrix,  direction='y', show_data=True,
                    cmap='viridis', norm=None)

plt.colorbar()

""
np.save('greenstone_ver', geo_model.solutions.vertices)
np.save('greenstone_edges', geo_model.solutions.edges)

""


"""
### Saving the model
"""

gp.save_model(geo_model, path=os.pardir+'/data/gempy_models')

""
geo_model.meta.project_name

"""
1.4: Unconformity relationships
===============================

"""

# %%
# Importing gempy
import gempy as gp

# Aux imports
import numpy as np
import pandas as pd
import os

np.random.seed(1515)
pd.set_option('precision', 2)

# %%
# We import a model from an existing folder, representing a subduction
# zone with onlap relationships. The theano function is automatically
# recombined to allow changes.
# 

# %%
cwd = os.getcwd()
if not 'examples' in cwd:
    data_path = os.getcwd() + '/examples/'
else:
    data_path = cwd + '/../../'

geo_model = gp.load_model(r'Tutorial_ch1-8_Onlap_relations',
                          path=data_path + 'data/gempy_models', recompile=True)

# %% 
gp.plot_2d(geo_model)

# %%
geo_model.set_regular_grid([-200, 1000, -500, 500, -1000, 0], [50, 50, 50])
geo_model.set_topography(d_z=np.array([-600, -100]))

# %% 
s = gp.compute_model(geo_model, compute_mesh=True, debug=False)

# %% 
gp.plot_2d(geo_model, cell_number=25, show_data=True)

# %%
gp.plot_2d(geo_model, 2, regular_grid=geo_model.solutions.mask_matrix_pad[0],
           show_data=True, kwargs_regular_grid={'cmap': 'gray', 'norm': None})

gp.plot_2d(geo_model, 2, regular_grid=geo_model.solutions.mask_matrix_pad[1],
           show_data=True, kwargs_regular_grid={'cmap': 'gray', 'norm': None})

gp.plot_2d(geo_model, 2, regular_grid=geo_model.solutions.mask_matrix_pad[2],
           show_data=True, kwargs_regular_grid={'cmap': 'gray', 'norm': None})

gp.plot_2d(geo_model, 2, regular_grid=geo_model.solutions.mask_matrix_pad[3],
           show_data=True, kwargs_regular_grid={'cmap': 'gray', 'norm': None})


# %%
# sphinx_gallery_thumbnail_number = 7
p3d = gp.plot_3d(geo_model, show_surfaces=True, show_data=True, image=False,
                 show_topography=True,
                 kwargs_plot_structured_grid={'opacity': .2})

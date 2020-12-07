"""
Greenstone.
===========
"""

# Importing gempy
import gempy as gp

# Aux imports
import numpy as np
import matplotlib.pyplot as plt
import os

print(gp.__version__)

# %% 
geo_model = gp.create_model('Greenstone')

# %%

data_path = 'https://raw.githubusercontent.com/cgre-aachen/gempy_data/master/'

# Importing the data from csv files and settign extent and resolution
geo_model = gp.init_data(geo_model, [696000, 747000, 6863000, 6930000, -20000, 200], [50, 50, 50],
                         path_o=data_path + "/data/input_data/tut_SandStone/SandStone_Foliations.csv",
                         path_i=data_path + "/data/input_data/tut_SandStone/SandStone_Points.csv")

# %% 
gp.plot_2d(geo_model, direction=['z'])

# %% 
gp.map_stack_to_surfaces(geo_model, {"EarlyGranite_Series": 'EarlyGranite',
                                     "BIF_Series": ('SimpleMafic2', 'SimpleBIF'),
                                     "SimpleMafic_Series": 'SimpleMafic1', 'Basement': 'basement'})

# %% 
geo_model.add_surface_values([2.61, 2.92, 3.1, 2.92, 2.61])

# %% 
gp.set_interpolator(geo_model,
                    compile_theano=True,
                    theano_optimizer='fast_compile',
                    verbose=[])

# %% 
gp.compute_model(geo_model, set_solutions=True)

# %% 
gp.plot_2d(geo_model, cell_number=[-1], direction=['z'], show_data=False)

# %% 
gp.plot_2d(geo_model, cell_number=[25], direction='x')

# %% 
geo_model.solutions.values_matrix

# %% 
p2d = gp.plot_2d(geo_model, cell_number=[25], block=geo_model.solutions.values_matrix,
           direction=['y'], show_data=True,
           kwargs_regular_grid={'cmap': 'viridis', 'norm':None})

# %%
# sphinx_gallery_thumbnail_number = 5
gp.plot_3d(geo_model)

# %% 
np.save('greenstone_ver', geo_model.solutions.vertices)
np.save('greenstone_edges', geo_model.solutions.edges)

# %%
# Saving the model
# ~~~~~~~~~~~~~~~~
# 

# %% 
# gp.save_model(geo_model, path=os.pardir + '/data/gempy_models')
gp.save_model(geo_model)
"""
Chapter 1.8: Onlap relationships
--------------------------------

"""

# %% 
# These two lines are necessary only if gempy is not installed
import sys, os
sys.path.append("../../..")
#sys.path.insert(0, '/home/miguel/anaconda3/lib/python3.6/site-packages/scikit_image-0.15.dev0-py3.6-linux-x86_64.egg/')
import skimage
# Importing gempy
import gempy as gp
import matplotlib.pyplot as plt
# Embedding matplotlib figures into the notebooks
#%matplotlib inline


# Aux imports
import numpy as np
import pandas as pn
import matplotlib
import theano
import qgrid

#%matplotlib widget


# %%
# We import a model from an existing folder, representing a subduction
# zone with onlap relationships. The theano function is automatically
# recombiled to allow changes.
# 

# %% 
geo_model = gp.load_model('Tutorial_ch1-8_Onlap_relations', path= '../../data/gempy_models', recompile=False)

# %% 
gp.plot.plot_data(geo_model)

# %% 
gp.set_interpolation_data(geo_model, verbose=[])

# %% 
geo_model.set_regular_grid([-200,1000,-500,500,-1000,0], [50,50,50])

# %% 
geo_model.set_topography(d_z=np.array([-600,-100]))


# %%
# Now topography exist but not activated:
# 

# %% 
geo_model.grid.set_active('topography')

# %% 
s = gp.compute_model(geo_model, compute_mesh=True, debug=False)

# %% 
geo_model.solutions.geological_map

# %% 
gp.plot.plot_section(geo_model, 25, show_data=True)

# %% 
gp.plot.plot_section(geo_model, 2, block=geo_model.solutions.mask_matrix_pad[3].T, show_data=True,
                    )

# %% 
vtkp = gp.plot.plot_3D(geo_model, render_surfaces=True, render_data=True)

# %% 
vtkp.set_real_time_on()

# %% 

geo_model.set_topography(d_z=np.array([-600,-100]), plot_object=vtkp)

# %% 
vtkp.resume()

# %% 
geo_model.surfaces

# %% 
vtkp.resume()


# %%
# Save model if any changes were made:
# 

# %% 
# geo_model.save_model('Tutorial_ch1-8_Onlap_relations', path= '../data/gempy_models',)
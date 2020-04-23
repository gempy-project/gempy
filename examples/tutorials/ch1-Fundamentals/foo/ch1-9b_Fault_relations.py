"""
Chapter 1.9b: Fault relations
-----------------------------

"""

# %% 
# These two lines are necessary only if gempy is not installed
import sys, os
sys.path.append("../../..")

# Importing gempy
import gempy as gp

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
# We import a model from an existing folder.
# 

# %% 
geo_model = gp.load_model('Tutorial_ch1-9b_Fault_relations', path='../../../data/gempy_models')

# %% 
geo_model._surfaces

# %% 
geo_model._additional_data


# %%
# Displaying the input data:
# 

# %% 
gp.plot.plot_data(geo_model, direction='y')

# %% 
gp.plot.plot_data(geo_model, direction='x')

# %% 
gp.set_interpolation_data(geo_model)

# %% 
geo_model._stack

# %% 
geo_model._faults

# %% 
geo_model._faults.faults_relations_df

# %% 
gp.compute_model(geo_model, compute_mesh=False)

# %% 
gp.plot.plot_section(geo_model, 25,  show_data=True)

# %% 
gp.plot.plot_scalar_field(geo_model, 25, series=2)


# %%
# Offset parameter
# ~~~~~~~~~~~~~~~~
# 

# %% 
geo_model._interpolator.theano_graph.offset.set_value(1)
gp.compute_model(geo_model, compute_mesh=False)

# %% 
gp.plot.plot_section(geo_model, 25, block=geo_model.solutions.block_matrix[1, 0, :125000], show_data=True)

# %% 
gp.plot.plot_scalar_field(geo_model, 25, series=2)

# %% 
geo_model.solutions.scalar_field_matrix[1]

# %% 
geo_model.save_model('Tutorial_ch1-9b_Fault_relations')


# %%
# Finding the faults intersection:
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# Sometimes we need to find the voxels that containt the each fault. To do
# so we can use gempys functionality to find interfaces as follows. Lets
# use the first fault as an example:
# 

# %% 
gp.plot.plot_section(geo_model, 25, block=geo_model.solutions.block_matrix[0, 0, :125000], show_data=True)

# %% 
# Importing the function to find the interface
from gempy.utils.input_manipulation import find_interfaces_from_block_bottoms
import matplotlib.pyplot as plt

# Remember the fault block is stored on:
geo_model.solutions.block_matrix[0, 0, :125000]

# %% 
# Now we can find where is the intersection of the values 1 by calling the following function. This will return
# Trues on those voxels on the intersection
intersection = find_interfaces_from_block_bottoms(
    geo_model.solutions.block_matrix[0, 0, :125000].reshape(50,50,50), 1,  shift= 1)

# %% 
# We can manually plotting together to see exactly what we have done
ax = gp.plot.plot_section(geo_model, 25, block=geo_model.solutions.block_matrix[0, 0, :125000], show_data=True)
plt.imshow(intersection[:, 25, :].T, origin='bottom', extent=(0,1000,-1000,0), alpha=.5)

"""
1.5: Fault relations
====================

"""

# %%
# Importing gempy
import gempy as gp

# Aux imports
import numpy as np
import pandas as pd
import os

# Importing the function to find the interface
from gempy.utils.input_manipulation import find_interfaces_from_block_bottoms
import matplotlib.pyplot as plt

np.random.seed(1515)
pd.set_option('precision', 2)

# %%
# We import a model from an existing folder.
# 

# %%
cwd = os.getcwd()
if 'examples' not in cwd:
    data_path = os.getcwd() + '/examples/'
else:
    data_path = cwd + '/../../'

geo_model = gp.load_model('Tutorial_ch1-9a_Fault_relations',
                          path=data_path + 'data/gempy_models', recompile=True)

# %% 
geo_model.faults.faults_relations_df

# %% 
geo_model.faults

# %% 
geo_model.surfaces

# %% 
gp.compute_model(geo_model, compute_mesh=False)

# %% 
geo_model.solutions.lith_block

# %% 
geo_model.solutions.block_matrix[0]

# %%
gp.plot_2d(geo_model, cell_number=[25], show_data=True)

# Graben example
# --------------


# %%
geo_model_graben = gp.load_model('Tutorial_ch1-9b_Fault_relations',
                                 path=data_path + 'data/gempy_models', recompile=True)

# %%
geo_model_graben.surfaces

# %%
geo_model_graben.additional_data

# %%
# Displaying the input data:
#

# %%
gp.plot_2d(geo_model_graben, direction='y')

# %%
gp.plot_2d(geo_model_graben, direction='x')


# %%
geo_model_graben.stack

# %%
geo_model_graben.faults

# %%
geo_model_graben.faults.faults_relations_df

# %%
gp.compute_model(geo_model_graben)

# %%
gp.plot_2d(geo_model_graben, cell_number=[25], show_data=True)

# %%
# sphinx_gallery_thumbnail_number = 5
gp.plot_3d(geo_model_graben, image=True)

# %%
gp.plot_2d(geo_model_graben, cell_number=[25], show_scalar=True, series_n=0)

gp.plot_2d(geo_model_graben, cell_number=[25], show_scalar=True, series_n=1)

# %%
# Offset parameter (Experimental)
# -------------------------------
#

# %%
geo_model_graben._interpolator.theano_graph.offset.set_value(1)
gp.compute_model(geo_model_graben, compute_mesh=False)


# %%
gp.plot_2d(geo_model_graben, block=geo_model_graben.solutions.block_matrix[1, 0, :125000],
           show_data=True)

# %%
gp.plot_2d(geo_model_graben, series_n=2, show_scalar=True)

# %%
geo_model_graben.solutions.scalar_field_matrix[1]

# %%
# Finding the faults intersection:
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Sometimes we need to find the voxels that contain the each fault. To do
# so we can use gempy's functionality to find interfaces as follows. Lets
# use the first fault as an example:
#

# %%
gp.plot_2d(geo_model_graben,
           regular_grid=geo_model_graben.solutions.block_matrix[0, 0, :125000],
           show_data=True)

# %%
# Remember the fault block is stored on:
geo_model_graben.solutions.block_matrix[0, 0, :125000]

# %%
# Now we can find where is the intersection of the values 1 by calling the following function.
# This will return Trues on those voxels on the intersection
intersection = find_interfaces_from_block_bottoms(
    geo_model_graben.solutions.block_matrix[0, 0, :125000].reshape(50, 50, 50), 1, shift=1)

# %%
# We can manually plotting together to see exactly what we have done
ax = gp.plot_2d(geo_model_graben,
                block=geo_model_graben.solutions.block_matrix[0, 0, :125000],
                show_data=True, show=False)

plt.imshow(intersection[:, 25, :].T, origin='bottom', extent=(0, 1000, -1000, 0), alpha=.5)
plt.show()
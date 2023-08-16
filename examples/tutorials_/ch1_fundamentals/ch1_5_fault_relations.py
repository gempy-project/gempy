"""
1.5: Fault relations
====================

"""

# %%
# Importing gempy
import gempy as gp
import gempy_viewer as gpv

# Aux imports
import numpy as np
import pandas as pd
import os

# Importing the function to find the interface
from gempy_plugins.utils.input_manipulation import find_interfaces_from_block_bottoms
import matplotlib.pyplot as plt

np.random.seed(1515)
pd.set_option('display.precision', 2)

# %%
# We import a model from an existing folder.
# 

# %%
data_path = os.path.abspath('../../')

geo_model: gp.data.GeoModel = gp.create_geomodel(
    project_name='Faults_relations',
    extent=[0, 1000, 0, 1000, -1000, -400],
    resolution=[20, 20, 20],
    number_octree_levels=1,  # * For this model is better not to use octrees because we want to see what is happening in the scalar fields
    importer_helper=gp.data.ImporterHelper(
        path_to_orientations=data_path + "/data/input_data/tut-ch1-5/tut_ch1-5_orientations.csv",
        path_to_surface_points=data_path + "/data/input_data/tut-ch1-5/tut_ch1-5_points.csv",
    )
)

print(geo_model)
# %%
# One fault model
# ---------------

# %% 
# Setting the structural frame

fault1: gp.data.StructuralElement = geo_model.structural_frame.get_element_by_name("fault1")
fault2: gp.data.StructuralElement = geo_model.structural_frame.get_element_by_name("fault2")

# Remove the faults from the default group
default_group: gp.data.StructuralGroup = geo_model.structural_frame.get_group_by_name("default_formation")
default_group.elements.remove(fault1)
default_group.elements.remove(fault2)

# Add a new group for the fault
gp.add_structural_group(
    model=geo_model,
    group_index=0,
    structural_group_name="fault_series_1",
    elements=[fault1],
    structural_relation=gp.data.StackRelationType.FAULT,
    fault_relations=gp.data.FaultsRelationSpecialCase.OFFSET_ALL
)

print(geo_model.structural_frame)

# %% 

geo_model.transform.apply_anisotropy(gp.data.GlobalAnisotropy.NONE)
if False:
    gp.compute_model(geo_model)
    # %%
    print(geo_model.solutions.raw_arrays.block_matrix[0])  # This contains the block values for the fault1
    print(geo_model.solutions.raw_arrays.block_matrix[1])  # This contains the block values for the formations
    # %%
    gpv.plot_2d(geo_model, show_data=True)
    gpv.plot_3d(geo_model, show_data=True, kwargs_plot_structured_grid={'opacity': .2})

# TODO: Add example of offsetting just one fault

# Graben example
# --------------

# %%
gp.add_structural_group(
    model=geo_model,
    group_index=1,
    structural_group_name="fault_series_2",
    elements=[fault2],
    structural_relation=gp.data.StackRelationType.FAULT,
    fault_relations=gp.data.FaultsRelationSpecialCase.OFFSET_ALL
)
print(geo_model.structural_frame)

from gempy_engine.core.data.kernel_classes.solvers import Solvers
geo_model.interpolation_options.kernel_options.kernel_solver = Solvers.SCIPY_CG
geo_model.interpolation_options.kernel_options.compute_condition_number = True
gp.compute_model(geo_model)

# %%
gpv.plot_2d(geo_model, show_data=True)
gpv.plot_3d(geo_model, show_data=True, image=True, kwargs_plot_structured_grid={'opacity': .2})

# %%
gpv.plot_2d(geo_model, show_scalar=True, show_lith=False,series_n=0)
gpv.plot_2d(geo_model, show_scalar=True, show_lith=False,series_n=1)
gpv.plot_2d(geo_model, show_scalar=True, show_lith=False,series_n=2)

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

plt.imshow(intersection[:, 25, :].T, origin='lower', extent=(0, 1000, -1000, 0), alpha=.5)
plt.show()

gp.save_model(geo_model)

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
import os

np.random.seed(1515)

# %%
# We import a model from an existing folder.
# 

# %%
data_path = os.path.abspath('../../')

geo_model: gp.data.GeoModel = gp.create_geomodel(
    project_name='Faults_relations',
    extent=[0, 1000, 0, 1000, -1000, -400],
    resolution=[20, 20, 20],
    refinement=6,  # * For this model is better not to use octrees because we want to see what is happening in the scalar fields
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

geo_model.input_transform.apply_anisotropy(gp.data.GlobalAnisotropy.NONE)
gp.compute_model(geo_model)
# %%
print(geo_model.solutions.raw_arrays.block_matrix[0])  # This contains the block values for the fault1
print(geo_model.solutions.raw_arrays.block_matrix[1])  # This contains the block values for the formations
# %%
gpv.plot_2d(geo_model, show_data=True)
gpv.plot_3d(geo_model, show_data=True, kwargs_plot_structured_grid={'opacity': .2})


# %5
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

gp.compute_model(geo_model)

# %%
gpv.plot_2d(geo_model, show_data=True)
gpv.plot_3d(geo_model, show_data=True, image=True, kwargs_plot_structured_grid={'opacity': .2})

# %%
gpv.plot_2d(geo_model, show_scalar=True, show_lith=False, series_n=0)
gpv.plot_2d(geo_model, show_scalar=True, show_lith=False, series_n=1)
gpv.plot_2d(geo_model, show_scalar=True, show_lith=False, series_n=2)

# %%
# Finite Faults
# -------------



# %%
# Faults relations
# ----------------

# %%
# Let's split the formations in two groups

gp.add_structural_group(
    model=geo_model,
    group_index=2,
    structural_group_name="series_1",
    elements=[
        geo_model.structural_frame.get_element_by_name("rock4"),
        geo_model.structural_frame.get_element_by_name("rock3")
    ],
    structural_relation=gp.data.StackRelationType.ERODE
)

default_group.elements.remove(geo_model.structural_frame.get_element_by_name("rock4"))
default_group.elements.remove(geo_model.structural_frame.get_element_by_name("rock3"))

gp.set_fault_relation(
    frame=geo_model.structural_frame,
    rel_matrix=np.array([
        [0, 1, 1, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ]
    )
)
print(geo_model.structural_frame)

# %%
gp.compute_model(geo_model)

# %%
gpv.plot_2d(geo_model, show_data=True)
gpv.plot_3d(geo_model, show_data=True, image=False, kwargs_plot_structured_grid={'opacity': .2})

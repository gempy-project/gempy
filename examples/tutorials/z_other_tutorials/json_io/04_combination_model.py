# %%
"""
Combination Model with JSON I/O
==============================

This example demonstrates how to create a model combining faults and unconformities using GemPy's JSON I/O functionality.
The model is based on the g07_combination.py example, featuring a folded domain with an unconformity and a fault.

Part 1: Create and save the initial model structure
Part 2: Load the model, compute it, and visualize results
"""

import gempy as gp
import gempy_viewer as gpv
import numpy as np
import os
from gempy_engine.core.data.stack_relation_type import StackRelationType
from gempy_engine.config import AvailableBackends
from gempy.modules.json_io.json_operations import JsonIO

# %% 
# Part 1: Create and save the initial model structure
# -------------------------------------------------

# Define paths
data_path = 'https://raw.githubusercontent.com/cgre-aachen/gempy_data/master/'
path_to_data = data_path + "/data/input_data/jan_models/"
json_file = "combination_model.json"
computed_json_file = "combination_model_computed.json"

# Create the model with data import
model = gp.create_geomodel(
    project_name='Combination Model',
    extent=[0, 2500, 0, 1000, 0, 1000],
    resolution=[20, 20, 20],
    refinement=6,
    importer_helper=gp.data.ImporterHelper(
        path_to_orientations=path_to_data + "model7_orientations.csv",
        path_to_surface_points=path_to_data + "model7_surface_points.csv"
    )
)

# Set metadata
model.meta.creation_date = "2024-03-24"
model.meta.last_modification_date = "2024-03-24"

# Map geological series to surfaces
gp.map_stack_to_surfaces(
    gempy_model=model,
    mapping_object={
        "Fault_Series": ('fault',),
        "Strat_Series1": ('rock3',),
        "Strat_Series2": ('rock2', 'rock1'),
    }
)

# Set structural relations
model.structural_frame.structural_groups[0].structural_relation = StackRelationType.FAULT
model.structural_frame.fault_relations = np.array([
    [0, 1, 1],
    [0, 0, 0],
    [0, 0, 0]
])

# Set colors for visualization
model.structural_frame.get_element_by_name("fault").color = "#015482"
model.structural_frame.get_element_by_name("rock3").color = "#9f0052"
model.structural_frame.get_element_by_name("rock2").color = "#ffbe00"
model.structural_frame.get_element_by_name("rock1").color = "#728f02"

# Set interpolation options
model.interpolation_options.number_octree_levels_surface = 5

# Save the model data to a JSON file
JsonIO.save_model_to_json(model, json_file)
print(f"\nSaved initial model to: {os.path.abspath(json_file)}")

# %% 
# Part 2: Load the model and compute
# ---------------------------------

print("\nLoading model from JSON...")
model = JsonIO.load_model_from_json(json_file)

print("\nModel Metadata:")
print(f"Name: {model.meta.name}")
print(f"Creation Date: {model.meta.creation_date}")
# TODO: This does not update here when running. In 03 you have a current date time setter
print(f"Last Modified: {model.meta.last_modification_date}")

print("\nStructural Groups:")
print(model.structural_frame)

# Compute the model
print("\nComputing the model...")
s = gp.compute_model(
    gempy_model=model,
    engine_config=gp.data.GemPyEngineConfig(
        backend=AvailableBackends.numpy
    )
)

# Save the computed model
# TODO: This is identical to the file saved before computation (no results stored)
JsonIO.save_model_to_json(model, computed_json_file)
print(f"\nSaved computed model to: {os.path.abspath(computed_json_file)}")

#%%

# Plot the results
print("\nGenerating plots...")


# 2D plots
gpv.plot_2d(model, direction='y', show_results=False)
gpv.plot_2d(model, direction='y', show_data=True, show_boundaries=True)
gpv.plot_2d(model, direction='x', show_data=True)

# Plot the blocks accounting for fault blocks
gpv.plot_2d(
    model=model,
    override_regular_grid=model.solutions.raw_arrays.litho_faults_block,
    show_data=True, kwargs_lithology={'cmap': 'Set1', 'norm': None}
)

# 3D plot
gpv.plot_3d(model)

print("\nDone! The model has been:")
print("1. Created and saved to:", json_file)
print("2. Loaded from JSON")
print("3. Computed")
print("4. Saved with computation results to:", computed_json_file)
print("5. Visualized with various plots") 
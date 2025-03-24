"""
Tutorial: Loading a Model with Multiple Series and Faults using JSON I/O
=====================================================================

This tutorial demonstrates how to load a geological model with multiple series and faults using GemPy's JSON I/O functionality.
The model consists of two layers (rock1, rock2) and a fault that offsets them.
"""

# %%
# Import necessary libraries
import matplotlib
# matplotlib.use('Agg')  # Use non-interactive backend
import gempy as gp
import gempy_viewer as gpv
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
from gempy_engine.core.data.stack_relation_type import StackRelationType
from gempy.modules.json_io.json_operations import JsonIO  # Updated import path
from datetime import datetime

# %%
# Define the model data
model_data = {
    "metadata": {
        "name": "Multiple Series and Faults Model",
        "creation_date": datetime.now().isoformat(),
        "last_modification_date": datetime.now().isoformat(),
        "owner": "GemPy Team"
    },
    "surface_points": [
        {"x": 0, "y": 200, "z": 600, "id": 0, "nugget": 0.00002},  # rock1
        {"x": 0, "y": 500, "z": 600, "id": 0, "nugget": 0.00002},  # rock1
        {"x": 0, "y": 800, "z": 600, "id": 0, "nugget": 0.00002},  # rock1
        {"x": 200, "y": 200, "z": 600, "id": 0, "nugget": 0.00002},  # rock1
        {"x": 200, "y": 500, "z": 600, "id": 0, "nugget": 0.00002},  # rock1
        {"x": 200, "y": 800, "z": 600, "id": 0, "nugget": 0.00002},  # rock1
        {"x": 800, "y": 200, "z": 200, "id": 0, "nugget": 0.00002},  # rock1
        {"x": 800, "y": 500, "z": 200, "id": 0, "nugget": 0.00002},  # rock1
        {"x": 800, "y": 800, "z": 200, "id": 0, "nugget": 0.00002},  # rock1
        {"x": 1000, "y": 200, "z": 200, "id": 0, "nugget": 0.00002},  # rock1
        {"x": 1000, "y": 500, "z": 200, "id": 0, "nugget": 0.00002},  # rock1
        {"x": 1000, "y": 800, "z": 200, "id": 0, "nugget": 0.00002},  # rock1
        {"x": 0, "y": 200, "z": 800, "id": 1, "nugget": 0.00002},  # rock2
        {"x": 0, "y": 800, "z": 800, "id": 1, "nugget": 0.00002},  # rock2
        {"x": 200, "y": 200, "z": 800, "id": 1, "nugget": 0.00002},  # rock2
        {"x": 200, "y": 800, "z": 800, "id": 1, "nugget": 0.00002},  # rock2
        {"x": 800, "y": 200, "z": 400, "id": 1, "nugget": 0.00002},  # rock2
        {"x": 800, "y": 800, "z": 400, "id": 1, "nugget": 0.00002},  # rock2
        {"x": 1000, "y": 200, "z": 400, "id": 1, "nugget": 0.00002},  # rock2
        {"x": 1000, "y": 800, "z": 400, "id": 1, "nugget": 0.00002},  # rock2
        {"x": 500, "y": 500, "z": 500, "id": 2, "nugget": 0.00002},  # fault
        {"x": 450, "y": 500, "z": 600, "id": 2, "nugget": 0.00002},  # fault
        {"x": 500, "y": 200, "z": 500, "id": 2, "nugget": 0.00002},  # fault
        {"x": 450, "y": 200, "z": 600, "id": 2, "nugget": 0.00002},  # fault
        {"x": 500, "y": 800, "z": 500, "id": 2, "nugget": 0.00002},  # fault
        {"x": 450, "y": 800, "z": 600, "id": 2, "nugget": 0.00002}  # fault
    ],
    "orientations": [
        {"x": 100, "y": 500, "z": 800, "G_x": 0, "G_y": 0, "G_z": 1, "id": 1, "nugget": 0.00002, "polarity": 1},  # rock2
        {"x": 100, "y": 500, "z": 600, "G_x": 0, "G_y": 0, "G_z": 1, "id": 0, "nugget": 0.00002, "polarity": 1},  # rock1
        {"x": 900, "y": 500, "z": 400, "G_x": 0, "G_y": 0, "G_z": 1, "id": 1, "nugget": 0.00002, "polarity": 1},  # rock2
        {"x": 900, "y": 500, "z": 200, "G_x": 0, "G_y": 0, "G_z": 1, "id": 0, "nugget": 0.00002, "polarity": 1},  # rock1
        {"x": 500, "y": 500, "z": 500, "G_x": 0.866, "G_y": 0, "G_z": 0.5, "id": 2, "nugget": 0.00002, "polarity": 1}  # fault
    ],
    "series": [
        {
            "name": "series1",
            "surfaces": ["rock1", "rock2"],
            "structural_relation": "ERODE",
            "colors": ["#015482", "#9f0052"]
        },
        {
            "name": "fault_series",
            "surfaces": ["fault"],
            "structural_relation": "FAULT",
            "colors": ["#ffbe00"]
        }
    ],
    "grid_settings": {
        "regular_grid_resolution": [50, 50, 50],
        "regular_grid_extent": [0, 1000, 0, 1000, 0, 1000],
        "octree_levels": None
    },
    "interpolation_options": {
        "kernel_options": {
            "range": 1.7,
            "c_o": 10
        },
        "mesh_extraction": True,
        "number_octree_levels": 1
    },
    "fault_relations": [[0, 1], [0, 0]],  # Fault series affects series1
    "id_name_mapping": {
        "name_to_id": {
            "rock1": 0,
            "rock2": 1,
            "fault": 2
        }
    }
}

# %%
# Save the model data to a JSON file
tutorial_dir = Path(__file__).parent
json_file = tutorial_dir / "multiple_series_faults.json"
with open(json_file, "w") as f:
    json.dump(model_data, f, indent=4)

# %%
# Load the model from JSON
model = JsonIO.load_model_from_json(str(json_file))

# Print metadata to verify it's properly loaded
print("\nModel Metadata:")
print(f"Name: {model.meta.name}")
print(f"Creation Date: {model.meta.creation_date}")
print(f"Last Modification Date: {model.meta.last_modification_date}")
print(f"Owner: {model.meta.owner}")

# Print structural groups
print("\nStructural Groups:")
print(model.structural_frame.structural_groups)

# %%
# Set fault relations
# Create a 2x2 matrix for fault relations (2 series: Fault_Series, Strat_Series)
# 1 means the fault affects the series, 0 means it doesn't
model.structural_frame.fault_relations = np.array([[0, 1], [0, 0]], dtype=bool)  # Using NumPy array with boolean type

# Explicitly set the structural relation for the fault series
model.structural_frame.structural_groups[1].structural_relation = StackRelationType.FAULT

# Set the fault series as a fault
gp.set_is_fault(
    frame=model,
    fault_groups=['fault_series']
)

# %%
# Compute the geological model
gp.compute_model(model)

# %%
# Save the computed model to a new JSON file
computed_json_file = tutorial_dir / "multiple_series_faults_computed.json"
JsonIO.save_model_to_json(model, str(computed_json_file))
print(f"\nSaved computed model to: {computed_json_file}")

# %%
# Load the computed model back to verify metadata is preserved
reloaded_model = JsonIO.load_model_from_json(str(computed_json_file))
print("\nReloaded Model Metadata:")
print(f"Name: {reloaded_model.meta.name}")
print(f"Creation Date: {reloaded_model.meta.creation_date}")
print(f"Last Modification Date: {reloaded_model.meta.last_modification_date}")
print(f"Owner: {reloaded_model.meta.owner}")

# %%
# Plot the model
# Plot the initial geological model in the y direction without results
fig, ax = plt.subplots(figsize=(10, 6))
gpv.plot_2d(model, direction=['y'], show_results=False, ax=ax)
plt.title("Initial Geological Model (y direction)")
plt.savefig('initial_model_y.png')
plt.close()

# Plot the result of the model in the x and y direction with data and without boundaries
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
gpv.plot_2d(model, direction=['x'], show_data=True, show_boundaries=False, ax=ax1)
ax1.set_title("Model with Data (x direction)")
gpv.plot_2d(model, direction=['y'], show_data=True, show_boundaries=False, ax=ax2)
ax2.set_title("Model with Data (y direction)")
plt.tight_layout()
plt.savefig('model_with_data.png')
plt.close()

# Plot the scalar field of the fault
fig, ax = plt.subplots(figsize=(10, 6))
gpv.plot_2d(model, show_scalar=True, show_lith=False, series_n=0, ax=ax)
plt.title("Fault Scalar Field")
plt.savefig('fault_scalar_field.png')
plt.close() 
# %%

"""
Tutorial: Loading a Model with Multiple Series and Faults using JSON I/O
=====================================================================

This tutorial demonstrates how to load a geological model with multiple series and faults using GemPy's JSON I/O functionality.
The model consists of two layers (rock1, rock2) and a fault that offsets them.
"""

# %%
# Import necessary libraries
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import gempy as gp
import gempy_viewer as gpv
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
from gempy_engine.core.data.stack_relation_type import StackRelationType
from gempy.modules.json_io.json_operations import JsonIO  # Updated import path

# %%
# Define the model data
model_data = {
    "metadata": {
        "name": "multiple_series_faults",
        "creation_date": "2024-03-19",
        "last_modification_date": "2024-03-19",
        "owner": "tutorial"
    },
    "surface_points": [
        # fault surface points (previously rock1 points)
        {"x": 0.0, "y": 200.0, "z": 600.0, "id": 2, "nugget": 0.00002},
        {"x": 0.0, "y": 500.0, "z": 600.0, "id": 2, "nugget": 0.00002},
        {"x": 0.0, "y": 800.0, "z": 600.0, "id": 2, "nugget": 0.00002},
        {"x": 200.0, "y": 200.0, "z": 600.0, "id": 2, "nugget": 0.00002},
        {"x": 200.0, "y": 500.0, "z": 600.0, "id": 2, "nugget": 0.00002},
        {"x": 200.0, "y": 800.0, "z": 600.0, "id": 2, "nugget": 0.00002},
        {"x": 800.0, "y": 200.0, "z": 200.0, "id": 2, "nugget": 0.00002},
        {"x": 800.0, "y": 500.0, "z": 200.0, "id": 2, "nugget": 0.00002},
        {"x": 800.0, "y": 800.0, "z": 200.0, "id": 2, "nugget": 0.00002},
        {"x": 1000.0, "y": 200.0, "z": 200.0, "id": 2, "nugget": 0.00002},
        {"x": 1000.0, "y": 500.0, "z": 200.0, "id": 2, "nugget": 0.00002},
        {"x": 1000.0, "y": 800.0, "z": 200.0, "id": 2, "nugget": 0.00002},
        # rock2 surface points
        {"x": 0.0, "y": 200.0, "z": 800.0, "id": 1, "nugget": 0.00002},
        {"x": 0.0, "y": 800.0, "z": 800.0, "id": 1, "nugget": 0.00002},
        {"x": 200.0, "y": 200.0, "z": 800.0, "id": 1, "nugget": 0.00002},
        {"x": 200.0, "y": 800.0, "z": 800.0, "id": 1, "nugget": 0.00002},
        {"x": 800.0, "y": 200.0, "z": 400.0, "id": 1, "nugget": 0.00002},
        {"x": 800.0, "y": 800.0, "z": 400.0, "id": 1, "nugget": 0.00002},
        {"x": 1000.0, "y": 200.0, "z": 400.0, "id": 1, "nugget": 0.00002},
        {"x": 1000.0, "y": 800.0, "z": 400.0, "id": 1, "nugget": 0.00002},
        # rock1 surface points (previously fault points)
        {"x": 500.0, "y": 500.0, "z": 500.0, "id": 0, "nugget": 0.00002},
        {"x": 450.0, "y": 500.0, "z": 600.0, "id": 0, "nugget": 0.00002},
        {"x": 500.0, "y": 200.0, "z": 500.0, "id": 0, "nugget": 0.00002},
        {"x": 450.0, "y": 200.0, "z": 600.0, "id": 0, "nugget": 0.00002},
        {"x": 500.0, "y": 800.0, "z": 500.0, "id": 0, "nugget": 0.00002},
        {"x": 450.0, "y": 800.0, "z": 600.0, "id": 0, "nugget": 0.00002},
    ],
    "orientations": [
        # rock2 orientation (upper layer at x=100)
        {"x": 100.0, "y": 500.0, "z": 800.0, "G_x": 0.0, "G_y": 0.0, "G_z": 1.0, "id": 1, "nugget": 0.01, "polarity": 1},
        # rock1 orientation (lower layer at x=100)
        {"x": 100.0, "y": 500.0, "z": 600.0, "G_x": 0.0, "G_y": 0.0, "G_z": 1.0, "id": 2, "nugget": 0.01, "polarity": 1},
        # fault orientation (at x=500)
        {"x": 500.0, "y": 500.0, "z": 500.0, "G_x": 0.8, "G_y": 0.0, "G_z": 0.6, "id": 0, "nugget": 0.01, "polarity": 1},
        # rock2 orientation (upper layer at x=900)
        {"x": 900.0, "y": 500.0, "z": 400.0, "G_x": 0.0, "G_y": 0.0, "G_z": 1.0, "id": 1, "nugget": 0.01, "polarity": 1},
        # rock1 orientation (lower layer at x=900)
        {"x": 900.0, "y": 500.0, "z": 200.0, "G_x": 0.0, "G_y": 0.0, "G_z": 1.0, "id": 2, "nugget": 0.01, "polarity": 1},
    ],
    "series": [
        {
            "name": "Fault_Series",
            "surfaces": ["fault"],
            "structural_relation": "FAULT",
            "color": "#015482"  # Blue color for fault
        },
        {
            "name": "Strat_Series",
            "surfaces": ["rock2", "rock1"],
            "structural_relation": "ERODE",
            "colors": ["#ffbe00", "#9f0052"]  # Yellow for rock2, Pink for rock1
        }
    ],
    "grid_settings": {
        "regular_grid_resolution": [90, 30, 30],  # Increased resolution for better visualization
        "regular_grid_extent": [0, 1000, 0, 1000, 0, 1000],
        "octree_levels": None
    },
    "interpolation_options": {}
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

# Print structural groups
print("\nStructural Groups:")
print(model.structural_frame.structural_groups)

# %%
# Set fault relations
# Create a 2x2 matrix for fault relations (2 series: Fault_Series, Strat_Series)
# 1 means the fault affects the series, 0 means it doesn't
model.structural_frame.fault_relations = np.array([
    [0, 1],  # Fault_Series affects Strat_Series
    [0, 0]   # Strat_Series doesn't affect any series
])

# Explicitly set the structural relation for the fault series
model.structural_frame.structural_groups[0].structural_relation = StackRelationType.FAULT

# Set the fault series as a fault
gp.set_is_fault(
    frame=model,
    fault_groups=['Fault_Series']
)

# %%
# Compute the geological model
gp.compute_model(model)

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
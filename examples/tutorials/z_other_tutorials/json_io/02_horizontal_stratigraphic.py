"""
Tutorial: Loading a Horizontal Stratigraphic Model using JSON I/O
===============================================================

This tutorial demonstrates how to load a horizontal stratigraphic model using GemPy's JSON I/O functionality.
The model consists of two horizontal layers (rock1 and rock2) with surface points and orientations.
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
from datetime import datetime

# %%
# Define the model data
model_data = {
    "metadata": {
        "name": "Horizontal Stratigraphic Model",
        "creation_date": "2024-03-24",
        "last_modification_date": datetime.now().strftime("%Y-%m-%d"),
        "owner": "GemPy Team"
    },
    "surface_points": [
        {"x": 0.0, "y": 0.0, "z": 0.0, "id": 1, "nugget": 0.0},
        {"x": 1.0, "y": 0.0, "z": 0.0, "id": 1, "nugget": 0.0},
        {"x": 0.0, "y": 1.0, "z": 0.0, "id": 1, "nugget": 0.0},
        {"x": 1.0, "y": 1.0, "z": 0.0, "id": 1, "nugget": 0.0},
        {"x": 0.0, "y": 0.0, "z": 1.0, "id": 2, "nugget": 0.0},
        {"x": 1.0, "y": 0.0, "z": 1.0, "id": 2, "nugget": 0.0},
        {"x": 0.0, "y": 1.0, "z": 1.0, "id": 2, "nugget": 0.0},
        {"x": 1.0, "y": 1.0, "z": 1.0, "id": 2, "nugget": 0.0}
    ],
    "orientations": [
        {"x": 0.5, "y": 0.5, "z": 0.0, "G_x": 0.0, "G_y": 0.0, "G_z": 1.0, "id": 1, "nugget": 0.0, "polarity": 1},
        {"x": 0.5, "y": 0.5, "z": 1.0, "G_x": 0.0, "G_y": 0.0, "G_z": 1.0, "id": 2, "nugget": 0.0, "polarity": 1}
    ],
    "series": [
        {
            "name": "series1",
            "surfaces": ["layer1", "layer2"],
            "structural_relation": "ERODE",
            "colors": ["#ff0000", "#00ff00"]
        }
    ],
    "grid_settings": {
        "regular_grid_resolution": [10, 10, 10],
        "regular_grid_extent": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
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
    "fault_relations": None,
    "id_name_mapping": {
        "name_to_id": {
            "layer1": 1,
            "layer2": 2
        }
    }
}

# %%
# Save the model data to a JSON file
tutorial_dir = Path(__file__).parent
json_file = tutorial_dir / "horizontal_stratigraphic.json"
with open(json_file, "w") as f:
    json.dump(model_data, f, indent=4)

# %%
# Load the model from JSON
model = gp.modules.json_io.JsonIO.load_model_from_json(str(json_file))

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
"""
Tutorial: Loading a Horizontal Stratigraphic Model using JSON I/O
===============================================================

This tutorial demonstrates how to load a horizontal stratigraphic model using GemPy's JSON I/O functionality.
The model consists of two horizontal layers (rock1 and rock2) with surface points and orientations.
"""

# %%
# Import necessary libraries
import gempy as gp
import gempy_viewer as gpv
import numpy as np
import json
from pathlib import Path

# %%
# Define the model data
model_data = {
    "metadata": {
        "name": "horizontal_stratigraphic",
        "creation_date": "2024-03-19",
        "last_modification_date": "2024-03-19",
        "owner": "tutorial"
    },
    "surface_points": [
        # rock1 surface points
        {"x": 500.0, "y": 500.0, "z": 500.0, "id": 0, "nugget": 0.00002},
        {"x": 500.0, "y": 500.0, "z": 600.0, "id": 0, "nugget": 0.00002},
        {"x": 500.0, "y": 500.0, "z": 700.0, "id": 0, "nugget": 0.00002},
        {"x": 500.0, "y": 500.0, "z": 800.0, "id": 0, "nugget": 0.00002},
        {"x": 500.0, "y": 500.0, "z": 900.0, "id": 0, "nugget": 0.00002},
        # rock2 surface points
        {"x": 500.0, "y": 500.0, "z": 200.0, "id": 1, "nugget": 0.00002},
        {"x": 500.0, "y": 500.0, "z": 300.0, "id": 1, "nugget": 0.00002},
        {"x": 500.0, "y": 500.0, "z": 400.0, "id": 1, "nugget": 0.00002},
        {"x": 500.0, "y": 500.0, "z": 500.0, "id": 1, "nugget": 0.00002},
        {"x": 500.0, "y": 500.0, "z": 600.0, "id": 1, "nugget": 0.00002},
    ],
    "orientations": [
        # rock1 orientations
        {"x": 500.0, "y": 500.0, "z": 500.0, "G_x": 0.0, "G_y": 0.0, "G_z": 1.0, "id": 0, "nugget": 0.01, "polarity": 1},
        {"x": 500.0, "y": 500.0, "z": 700.0, "G_x": 0.0, "G_y": 0.0, "G_z": 1.0, "id": 0, "nugget": 0.01, "polarity": 1},
        {"x": 500.0, "y": 500.0, "z": 900.0, "G_x": 0.0, "G_y": 0.0, "G_z": 1.0, "id": 0, "nugget": 0.01, "polarity": 1},
        # rock2 orientations
        {"x": 500.0, "y": 500.0, "z": 200.0, "G_x": 0.0, "G_y": 0.0, "G_z": 1.0, "id": 1, "nugget": 0.01, "polarity": 1},
        {"x": 500.0, "y": 500.0, "z": 400.0, "G_x": 0.0, "G_y": 0.0, "G_z": 1.0, "id": 1, "nugget": 0.01, "polarity": 1},
        {"x": 500.0, "y": 500.0, "z": 600.0, "G_x": 0.0, "G_y": 0.0, "G_z": 1.0, "id": 1, "nugget": 0.01, "polarity": 1},
    ],
    "series": [
        {
            "name": "Strat_Series",
            "surfaces": ["rock2", "rock1"]
        }
    ],
    "grid_settings": {
        "regular_grid_resolution": [10, 10, 10],
        "regular_grid_extent": [0, 1000, 0, 1000, 0, 1000],
        "octree_levels": None
    },
    "interpolation_options": {}
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
# Plot the model
# Plot the initial geological model in the y direction without results
gpv.plot_2d(model, direction=['y'], show_results=False)

# Plot the result of the model in the x and y direction with data and without boundaries
gpv.plot_2d(model, direction=['x'], show_data=True, show_boundaries=False)
gpv.plot_2d(model, direction=['y'], show_data=True, show_boundaries=False) 
"""
Tutorial: Minimal JSON I/O with default metadata, interpolation options, and grid settings
This tutorial demonstrates how to use the JSON I/O functionality with minimal input,
relying on default metadata values, interpolation options, and grid settings.
"""

# %%
import numpy as np
from datetime import datetime

import gempy as gp
import gempy_viewer as gpv
import json
import pyvista as pv

from gempy.modules.json_io.json_operations import JsonIO  # Updated import path


# %%
# Create the corresponding minimal JSON model
model_data = {
    "surface_points": [
        {
            "x": 50.0,
            "y": 0.0,
            "z": -20.0,
            "id": 0,
        }
    ],
    "orientations": [
        {
            "x": 50.0,
            "y": 0.0,
            "z": -20.0,
            "G_x": 1.0,
            "G_y": 0.0,
            "G_z": 1.0,
            "id": 0,
            "polarity": 1
        }
    ],
    "grid_settings": {
        "regular_grid_resolution": [100, 2, 100],
        "regular_grid_extent": [0, 150, -10, 10, -100, 0]
    }
}

# %%
# Save the minimal model to JSON
with open("minimal_model.json", "w") as f:
    json.dump(model_data, f, indent=4)

# Load the model from JSON
geo_model = JsonIO.load_model_from_json("minimal_model.json")

# Compute the geological model
gp.compute_model(geo_model)

p2d = gpv.plot_2d(geo_model)
# %%

# Print the model metadata (should use default values)
print("\nModel Metadata:")
print(f"Name: {geo_model.meta.name}")
print(f"Creation Date: {geo_model.meta.creation_date}")
print(f"Last Modification Date: {geo_model.meta.last_modification_date}")
print(f"Owner: {geo_model.meta.owner}")

# Print the interpolation options (should use default values)
print("\nInterpolation Options:")
print(f"Range: {geo_model.interpolation_options.kernel_options.range}")
print(f"Mesh Extraction: {geo_model.interpolation_options.mesh_extraction}")

# Print the grid settings (should use default values)
print("\nGrid Settings:")
print(f"Resolution: {geo_model.grid._dense_grid.resolution}")
print(f"Extent: {geo_model.grid._dense_grid.extent}")

# Print the structural groups
print("\nStructural Groups:")
for group in geo_model.structural_frame.structural_groups:
    print(group)

# Save the loaded model to verify the metadata, interpolation options, and grid settings are preserved
JsonIO.save_model_to_json(geo_model, "minimal_model_loaded.json")

# %%

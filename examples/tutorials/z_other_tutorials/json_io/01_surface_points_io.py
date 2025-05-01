"""
Tutorial for JSON I/O operations in GemPy - Surface Points
=======================================================

This tutorial demonstrates how to save and load surface points data using JSON files in GemPy.
"""

import json
import numpy as np
import gempy as gp
from gempy.modules.json_io import JsonIO

# Create a sample surface points dataset
# ------------------------------------

# Create some sample surface points
x = np.array([0, 1, 2, 3, 4])
y = np.array([0, 1, 2, 3, 4])
z = np.array([0, 1, 2, 3, 4])
ids = np.array([0, 0, 1, 1, 2])  # Three different surfaces
nugget = np.array([0.00002, 0.00002, 0.00002, 0.00002, 0.00002])

# Create name to id mapping
name_id_map = {f"surface_{id}": id for id in np.unique(ids)}

# Create a SurfacePointsTable
surface_points = gp.data.SurfacePointsTable.from_arrays(
    x=x,
    y=y,
    z=z,
    names=[f"surface_{id}" for id in ids],
    nugget=nugget,
    name_id_map=name_id_map
)

# Create a JSON file with the surface points data
# ---------------------------------------------

# Create the JSON structure
json_data = {
    "metadata": {
        "name": "sample_model",
        "creation_date": "2024-03-19",
        "last_modification_date": "2024-03-19",
        "owner": "tutorial"
    },
    "surface_points": [
        {
            "x": float(x[i]),
            "y": float(y[i]),
            "z": float(z[i]),
            "id": int(ids[i]),
            "nugget": float(nugget[i])
        }
        for i in range(len(x))
    ],
    "orientations": [],
    "faults": [],
    "series": [],
    "grid_settings": {
        "regular_grid_resolution": [10, 10, 10],
        "regular_grid_extent": [0, 4, 0, 4, 0, 4],
        "octree_levels": None
    },
    "interpolation_options": {}
}

# Save the JSON file
with open("sample_surface_points.json", "w") as f:
    json.dump(json_data, f, indent=4)

# Load the surface points from JSON
# -------------------------------

# Load the model from JSON
loaded_surface_points = JsonIO._load_surface_points(json_data["surface_points"])

# Verify the loaded data
print("\nOriginal surface points:")
print(surface_points)
print("\nLoaded surface points:")
print(loaded_surface_points)

# Verify the data matches
print("\nVerifying data matches:")
print(f"X coordinates match: {np.allclose(surface_points.xyz[:, 0], loaded_surface_points.xyz[:, 0])}")
print(f"Y coordinates match: {np.allclose(surface_points.xyz[:, 1], loaded_surface_points.xyz[:, 1])}")
print(f"Z coordinates match: {np.allclose(surface_points.xyz[:, 2], loaded_surface_points.xyz[:, 2])}")
print(f"IDs match: {np.array_equal(surface_points.ids, loaded_surface_points.ids)}")
print(f"Nugget values match: {np.allclose(surface_points.nugget, loaded_surface_points.nugget)}")

# Print the name_id_maps to compare
print("\nName to ID mappings:")
print(f"Original: {surface_points.name_id_map}")
print(f"Loaded: {loaded_surface_points.name_id_map}") 
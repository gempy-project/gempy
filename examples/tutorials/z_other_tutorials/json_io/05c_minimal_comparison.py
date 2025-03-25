"""
Tutorial: Compare minimal GemPy model with JSON representation
This tutorial demonstrates how to create a minimal GemPy model and compare it with its JSON representation.
"""

# %%
import gempy as gp
import gempy_viewer as gpv
import json
import numpy as np
from gempy.modules.json_io.json_operations import JsonIO

# %%
# Set up the basic model
geo_model: gp.data.GeoModel = gp.create_geomodel(
    project_name='Model1',
    extent=[0, 150, -10, 10, -100, 0],
    resolution=[100, 2, 100],
    structural_frame=gp.data.StructuralFrame.initialize_default_structure()
)

# Add a surface point
gp.add_surface_points(
    geo_model=geo_model,
    x=[50],
    y=[0],
    z=[-20],
    elements_names=['surface1']
)

# Add an orientation
gp.add_orientations(
    geo_model=geo_model,
    x=[50],
    y=[0],
    z=[-68],
    elements_names=['surface1'],
    pole_vector=[[1, 0, 1]]
)

# Set interpolation options
geo_model.interpolation_options.kernel_options.range = 10.
geo_model.interpolation_options.kernel_options.c_o = 5.
geo_model.interpolation_options.mesh_extraction = True
geo_model.interpolation_options.number_octree_levels = 2

geo_model.update_transform(gp.data.GlobalAnisotropy.NONE)
gp.compute_model(geo_model, engine_config=gp.data.GemPyEngineConfig())

# %%
# Plot the model
p2d = gpv.plot_2d(geo_model)

# %%
# Create the corresponding minimal JSON model
json_model_data = {
    "surface_points": [
        {
            "x": 50.0,
            "y": 0.0,
            "z": -20.0,
            "id": 0
        }
    ],
    "orientations": [
        {
            "x": 50.0,
            "y": 0.0,
            "z": -68.0,
            "G_x": 1.0,
            "G_y": 0.0,
            "G_z": 1.0,
            "id": 0
        }
    ],
    "grid_settings": {
        "regular_grid_resolution": [100, 2, 100],
        "regular_grid_extent": [0, 150, -10, 10, -100, 0]
    },
    "interpolation_options": {
        "kernel_options": {
            "range": 10.0,
            "c_o": 5.0
        },
        "mesh_extraction": True,
        "number_octree_levels": 2
    },
    "series": [
        {
            "name": "default_formations",
            "surfaces": ["surface1"],
            "structural_relation": "ERODE"
        }
    ]
}

# Save the JSON model
with open("minimal_model.json", "w") as f:
    json.dump(json_model_data, f, indent=4)

# Load the model from JSON
json_geo_model = JsonIO.load_model_from_json("minimal_model.json")

# Compute the JSON model
gp.compute_model(json_geo_model, engine_config=gp.data.GemPyEngineConfig())

# %%
# Compare the models
print("\nComparing GemPy and JSON models:")
print("\n1. Surface Points:")
print("GemPy model:")
print(geo_model.surface_points_copy)
print("\nJSON model:")
print(json_geo_model.surface_points_copy)

print("\n2. Orientations:")
print("GemPy model:")
print(geo_model.orientations_copy)
print("\nJSON model:")
print(json_geo_model.orientations_copy)

print("\n3. Grid Settings:")
print("GemPy model:")
print(f"Resolution: {geo_model.grid._dense_grid.resolution}")
print(f"Extent: {geo_model.grid._dense_grid.extent}")
print("\nJSON model:")
print(f"Resolution: {json_geo_model.grid._dense_grid.resolution}")
print(f"Extent: {json_geo_model.grid._dense_grid.extent}")

print("\n4. Interpolation Options:")
print("GemPy model:")
print(f"Range: {geo_model.interpolation_options.kernel_options.range}")
print(f"C_o: {geo_model.interpolation_options.kernel_options.c_o}")
print(f"Mesh Extraction: {geo_model.interpolation_options.mesh_extraction}")
print(f"Number Octree Levels: {geo_model.interpolation_options.number_octree_levels}")
print("\nJSON model:")
print(f"Range: {json_geo_model.interpolation_options.kernel_options.range}")
print(f"C_o: {json_geo_model.interpolation_options.kernel_options.c_o}")
print(f"Mesh Extraction: {json_geo_model.interpolation_options.mesh_extraction}")
print(f"Number Octree Levels: {json_geo_model.interpolation_options.number_octree_levels}")

print("\n5. Structural Groups:")
print("GemPy model:")
for group in geo_model.structural_frame.structural_groups:
    print(group)
print("\nJSON model:")
for group in json_geo_model.structural_frame.structural_groups:
    print(group)

# %%
# Plot both models side by side
p2d_gempy = gpv.plot_2d(geo_model, title="GemPy Model")
p2d_json = gpv.plot_2d(json_geo_model, title="JSON Model")

# %% 
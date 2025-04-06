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
        "creation_date": "2024-03-24",  # Fixed creation date
        "last_modification_date": datetime.now().strftime("%Y-%m-%d"),  # Current date only
        "owner": "GemPy Team"
    },
    "surface_points": [
        # rock1 points
        {"x": 0, "y": 200, "z": 600, "id": 0, "nugget": 0.00002},
        {"x": 0, "y": 500, "z": 600, "id": 0, "nugget": 0.00002},
        {"x": 0, "y": 800, "z": 600, "id": 0, "nugget": 0.00002},
        {"x": 200, "y": 200, "z": 600, "id": 0, "nugget": 0.00002},
        {"x": 200, "y": 500, "z": 600, "id": 0, "nugget": 0.00002},
        {"x": 200, "y": 800, "z": 600, "id": 0, "nugget": 0.00002},
        {"x": 800, "y": 200, "z": 200, "id": 0, "nugget": 0.00002},
        {"x": 800, "y": 500, "z": 200, "id": 0, "nugget": 0.00002},
        {"x": 800, "y": 800, "z": 200, "id": 0, "nugget": 0.00002},
        {"x": 1000, "y": 200, "z": 200, "id": 0, "nugget": 0.00002},
        {"x": 1000, "y": 500, "z": 200, "id": 0, "nugget": 0.00002},
        {"x": 1000, "y": 800, "z": 200, "id": 0, "nugget": 0.00002},
        # rock2 points
        {"x": 0, "y": 200, "z": 800, "id": 1, "nugget": 0.00002},
        {"x": 0, "y": 800, "z": 800, "id": 1, "nugget": 0.00002},
        {"x": 200, "y": 200, "z": 800, "id": 1, "nugget": 0.00002},
        {"x": 200, "y": 800, "z": 800, "id": 1, "nugget": 0.00002},
        {"x": 800, "y": 200, "z": 400, "id": 1, "nugget": 0.00002},
        {"x": 800, "y": 800, "z": 400, "id": 1, "nugget": 0.00002},
        {"x": 1000, "y": 200, "z": 400, "id": 1, "nugget": 0.00002},
        {"x": 1000, "y": 800, "z": 400, "id": 1, "nugget": 0.00002},
        # fault points
        {"x": 500, "y": 500, "z": 500, "id": 2, "nugget": 0.00002},
        {"x": 450, "y": 500, "z": 600, "id": 2, "nugget": 0.00002},
        {"x": 500, "y": 200, "z": 500, "id": 2, "nugget": 0.00002},
        {"x": 450, "y": 200, "z": 600, "id": 2, "nugget": 0.00002},
        {"x": 500, "y": 800, "z": 500, "id": 2, "nugget": 0.00002},
        {"x": 450, "y": 800, "z": 600, "id": 2, "nugget": 0.00002}
    ],
    "orientations": [
        # rock2 orientations
        {"x": 100, "y": 500, "z": 800, "G_x": 0, "G_y": 0, "G_z": 1, "id": 1, "nugget": 0.00002, "polarity": 1},
        {"x": 900, "y": 500, "z": 400, "G_x": 0, "G_y": 0, "G_z": 1, "id": 1, "nugget": 0.00002, "polarity": 1},
        # rock1 orientations
        {"x": 100, "y": 500, "z": 600, "G_x": 0, "G_y": 0, "G_z": 1, "id": 0, "nugget": 0.00002, "polarity": 1},
        {"x": 900, "y": 500, "z": 200, "G_x": 0, "G_y": 0, "G_z": 1, "id": 0, "nugget": 0.00002, "polarity": 1},
        # fault orientation (60-degree dip)
        {"x": 500, "y": 500, "z": 500, "G_x": 0.866, "G_y": 0, "G_z": 0.5, "id": 2, "nugget": 0.00002, "polarity": 1}
    ],
    "series": [
        {
            "name": "Fault_Series",
            "surfaces": ["fault"],
            "structural_relation": "FAULT",
            "colors": ["#ffbe00"]
        },
        {
            "name": "Strat_Series",
            "surfaces": ["rock2", "rock1"],
            "structural_relation": "ERODE",
            "colors": ["#015482", "#9f0052"]
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
    "fault_relations": [[0, 1], [0, 0]],  # Fault series affects Strat_Series
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
# Compute the geological model
gp.compute_model(model)

# Print scalar field values to verify they are calculated
print("\nScalar Field Values (Fault Series):")
print(f"Shape: {model.solutions.raw_arrays.scalar_field_matrix.shape}")
print(f"Min value: {model.solutions.raw_arrays.scalar_field_matrix[0].min()}")
print(f"Max value: {model.solutions.raw_arrays.scalar_field_matrix[0].max()}")
print(f"Mean value: {model.solutions.raw_arrays.scalar_field_matrix[0].mean()}")

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
# Create plots with proper configuration
# Plot 1: Cross-section in Y direction (XZ plane)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)
gpv.plot_2d(
    model,
    cell_number=25,  # Middle of the model
    direction='y',
    show_data=True,
    show_boundaries=True,
    show_results=True,
    ax=ax
)
plt.title("Geological Model - Y Direction (XZ plane)")
plt.savefig('model_y_direction.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot 2: Cross-section in X direction (YZ plane)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)
gpv.plot_2d(
    model,
    cell_number=25,  # Middle of the model
    direction='x',
    show_data=True,
    show_boundaries=True,
    show_results=True,
    ax=ax
)
plt.title("Geological Model - X Direction (YZ plane)")
plt.savefig('model_x_direction.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot 3: Scalar field of the fault
plt.figure(figsize=(10, 8))
ax = plt.gca()

# Get scalar field values and reshape to grid
scalar_field = model.solutions.raw_arrays.scalar_field_matrix[0].reshape(50, 50, 50)

# Plot middle slice in Y direction
middle_slice = scalar_field[:, 25, :]
im = ax.imshow(middle_slice.T, 
               extent=[0, 1000, 0, 1000],
               origin='lower',
               cmap='RdBu',
               aspect='equal')

# Add colorbar
plt.colorbar(im, ax=ax, label='Scalar Field Value')

# Plot surface points
fault_element = model.structural_frame.get_element_by_name("fault")
if fault_element and fault_element.surface_points is not None:
    fault_points_coords = fault_element.surface_points.xyz

    # Filter points near the slice (Y around 500)
    mask = np.abs(fault_points_coords[:, 1] - 500) < 100
    filtered_points = fault_points_coords[mask]

    if len(filtered_points) > 0:
        ax.scatter(filtered_points[:, 0], filtered_points[:, 2], 
                  c='red', s=50, label='Surface Points')
        ax.legend()

ax.set_xlabel('X')
ax.set_ylabel('Z')
ax.set_title('Fault Scalar Field - Y Direction (Middle Slice)')

# Save plot
plt.savefig('fault_scalar_field.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nPlot saved as fault_scalar_field.png")

# Plot 4: 3D visualization (optional)
PLOT_3D = False  # Set to True to enable 3D plotting

if PLOT_3D:
    try:
        import pyvista as pv
        p = pv.Plotter(notebook=False, off_screen=True)
        gpv.plot_3d(
            model,
            show_data=True,
            show_surfaces=True,
            show_boundaries=True,
            plotter=p
        )
        p.screenshot('model_3d.png', transparent_background=False)
        p.close()
    except Exception as e:
        print(f"Could not create 3D plot: {e}")
# %%

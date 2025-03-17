"""
Model 1 - Horizontal stratigraphic
==================================

This script demonstrates how to create a basic model of horizontally stacked layers using GemPy,
a Python-based, open-source library for implicit geological modeling.
"""

# Import necessary libraries
import gempy as gp
import gempy_viewer as gpv


# sphinx_gallery_thumbnail_number = 2

# %%
# Generate the model
# Define the path to data
data_path = 'https://raw.githubusercontent.com/cgre-aachen/gempy_data/master/'
# Create a GeoModel instance
data = gp.create_geomodel(
    project_name='horizontal',
    extent=[0, 1000, 0, 1000, 0, 1000],
    refinement=6,
    importer_helper=gp.data.ImporterHelper(
        path_to_orientations=data_path + "/data/input_data/jan_models/model1_orientations.csv",
        path_to_surface_points=data_path + "/data/input_data/jan_models/model1_surface_points.csv"
    )
)
# Map geological series to surfaces
gp.map_stack_to_surfaces(
    gempy_model=data,
    mapping_object={"Strat_Series": ('rock2', 'rock1')}
)
# Compute the geological model
gp.compute_model(data)
geo_data = data

# %%
# Plot the initial geological model in the y direction without results
gpv.plot_2d(geo_data, direction=['y'], show_results=False)

# Plot the result of the model in the x and y direction with data and without boundaries
gpv.plot_2d(geo_data, direction=['x'], show_data=True, show_boundaries=False)
gpv.plot_2d(geo_data, direction=['y'], show_data=True, show_boundaries=False)

# %%
# Export to VTK model
p = gpv.plot_3d(geo_data, show_data=True, show_results=True, show_boundaries=True,image=True)
p.surface_poly['rock1'].save('rock1.vtk') # Save the vtk file for formation 1
p.surface_poly['rock2'].save('rock2.vtk') # Save the vtk file for formation 2
p.orientations_mesh.save('orientations.vtk') # Save the vtk file for the orientations
p.surface_points_mesh.save('surface_points.vtk') # Save the vtk file for the surface points
box = p.regular_grid_actor.GetMapper().GetInput() # Get the vtk file for the regular grid
box.save('box.vtk')

# %%
import pyvista as pv
pv.read('rock1.vtk').plot(show_edges=False)
pv.read('rock2.vtk').plot(show_edges=False)
pv.read('orientations.vtk').plot(show_edges=False)
pv.read('surface_points.vtk').plot(show_edges=False)
pv.read('box.vtk').plot(show_edges=False)

# %%
# Export vertices of mesh
from builtins import range
import vtk
import pandas as pd
def generate_normals(polydata):
    normal_generator = vtk.vtkPolyDataNormals()
    normal_generator.SetInputData(polydata)
    normal_generator.ComputePointNormalsOn()
    normal_generator.ComputeCellNormalsOff()
    normal_generator.Update()
    return normal_generator.GetOutput()

def get_vertices_and_normals(mesh):

    surface_mesh = mesh.extract_surface()
    polydata = surface_mesh

    # Generate normals if not present
    polydata_with_normals = generate_normals(polydata)

    # Get points (vertices)
    points = polydata_with_normals.GetPoints()
    vertices = []
    for i in range(points.GetNumberOfPoints()):
        vertices.append(points.GetPoint(i))

    # Get normals
    normals_array = polydata_with_normals.GetPointData().GetNormals()
    normals = []
    for i in range(normals_array.GetNumberOfTuples()):
        normals.append(normals_array.GetTuple(i))

    return vertices, normals

def save_to_excel(vertices, normals, vertices_file, normals_file):
    # Create DataFrames
    vertices_df = pd.DataFrame(vertices, columns=['X', 'Y', 'Z'])
    normals_df = pd.DataFrame(normals, columns=['x', 'y', 'z'])

    # Save to Excel files
    vertices_df.to_excel(vertices_file, index=False)
    normals_df.to_excel
mesh = p.surface_poly['rock1']
vertices, normals = get_vertices_and_normals(mesh)
vertices_df = pd.DataFrame(vertices, columns=['X', 'Y', 'Z'])
normals_df = pd.DataFrame(normals, columns=['x', 'y', 'z'])
# Save to Excel filesthe
vertices_file = "rock1_vertices.xlsx"
normals_file = "rock1_norms.xlsx"
save_to_excel(vertices, normals, vertices_file, normals_file)
pd.read_excel(vertices_file)
pd.read_excel(normals_file)

# %%
# Convert the DataFrame to an XYZ file
def dataframe_to_xyz(df, filename):
    with open(filename, 'w') as f:
        for index, row in df.iterrows():
            f.write(f"{row['X']} {row['Y']} {row['Z']}\n")

# Specify the filename
filename = "output.xyz"

# Call the function to write the DataFrame to an XYZ file
dataframe_to_xyz(vertices_df, filename)
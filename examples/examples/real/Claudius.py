"""
Geomodeling benchmark: the "Claudius"-Model
========

This model is part of a geomodeling benchmaring effort. More information (and, hopefully, publication) coming.
"""

# %%
import sys, os

# Importing gempy
import gempy as gp
import gempy_viewer as gpv

# Aux imports
import numpy as np
import pandas as pn

# %%
# Loading data from repository:
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# With pandas we can do it directly from the web and with the right args
# we can directly tidy the data in gempy style:
# 

# %% 

data_path = os.path.abspath('../../data/input_data/Claudius')

reduce_data_by = 30

dfs = []
for letter in 'ABCD':
    dfs.append(
        pn.read_csv(
            filepath_or_buffer=f"{data_path}/{letter}Points.csv",
            sep=';',
            names=['X', 'Y', 'Z', 'surface', 'cutoff'],
            header=0
        )[::reduce_data_by]
    )

# Add fault:
dfs.append(
    pn.read_csv(
        filepath_or_buffer=f"{data_path}/Fault.csv",
        names=['X', 'Y', 'Z', 'surface'],
        header=0,
        sep=','
    )
)

surface_points = pn.concat(dfs, sort=True)
surface_points['surface'] = surface_points['surface'].astype('str')
surface_points.reset_index(inplace=True, drop=False)
surface_points.tail()

# %%
surface_points.dtypes

# %%
# How many points are per surface
# 

# %% 
surface_points.groupby('surface').count()

# %%
# Now we do the same with the orientations:
# 

# %% 
dfs = []

for surf in ['0', '330']:
    o = pn.read_csv(
        filepath_or_buffer=f"{data_path}/Dips.csv",
        sep=';',
        names=['X', 'Y', 'Z', 'G_x', 'G_y', 'G_z', '-'],
        header=1
    )

    # Orientation needs to belong to a surface. This is mainly to categorize to which series belong and to
    # use the same color
    o['surface'] = surf
    dfs.append(o)
orientations = pn.concat(dfs, sort=True)
orientations.reset_index(inplace=True, drop=False)

orientations.tail()

# %%
orientations.dtypes

# %%
# Data initialization:
# ~~~~~~~~~~~~~~~~~~~~
# 
# Suggested size of the axis-aligned modeling box: Origin: 548800 7816600
# -8400 Maximum: 552500 7822000 -11010
# 
# Suggested resolution: 100m x 100m x -90m (grid size 38 x 55 x 30)
# 

# %% 
# Number of voxels:
np.array([38, 55, 30]).prod()

surface_points_table: gp.data.SurfacePointsTable = gp.data.SurfacePointsTable.from_arrays(
    x=surface_points['X'].values,
    y=surface_points['Y'].values,
    z=surface_points['Z'].values,
    names=surface_points['surface'].values
)

orientations_table: gp.data.OrientationsTable = gp.data.OrientationsTable.from_arrays(
    x=orientations['X'].values,
    y=orientations['Y'].values,
    z=orientations['Z'].values,
    G_x=orientations['G_x'].values,
    G_y=orientations['G_y'].values,
    G_z=orientations['G_z'].values,
    names=orientations['surface'].values,
    name_id_map=surface_points_table.name_id_map  # ! Make sure that ids and names are shared
)

structural_frame: gp.data.StructuralFrame = gp.data.StructuralFrame.from_data_tables(
    surface_points=surface_points_table,
    orientations=orientations_table
)

geo_model: gp.data.GeoModel = gp.create_geomodel(
    project_name='Claudius',
    extent=[548800, 552500, 7816600, 7822000, -11010, -8400],
    resolution=[38, 55, 30],
    refinement=5,
    structural_frame=structural_frame
)

group_fault = gp.data.StructuralGroup(
    name='Fault1',
    elements=[geo_model.structural_frame.structural_elements.pop(-2)],
    structural_relation=gp.data.StackRelationType.FAULT,
    fault_relations=gp.data.FaultsRelationSpecialCase.OFFSET_ALL
)

geo_model.structural_frame.get_group_by_name("default_formation").elements.pop(-1)

# Insert the fault group into the structural frame:
geo_model.structural_frame.insert_group(0, group_fault)

gp.set_is_fault(
    frame=geo_model.structural_frame,
    fault_groups=[geo_model.structural_frame.get_group_by_name('Fault1')]
)

print(geo_model)

# %%
# We are going to increase the smoothness (nugget) of the data to increase
# the conditional number of the matrix:
# 

# %% 
gp.modify_surface_points(geo_model, nugget=0.01)

# %%
# Also the original poles are pointing downwards. We can change the
# direction by calling the following:
# 

# %%
gp.modify_orientations(geo_model, polarity=-1)

# %%
# We need an orientation per series/fault. The faults does not have
# orientation so the easiest is to create an orientation from the surface
# points availablle:
# 

element = geo_model.structural_frame.get_element_by_name("Claudius_fault")
new_orientations: gp.data.OrientationsTable = gp.create_orientations_from_surface_points_coords(
    xyz_coords=element.surface_points.xyz
)
gp.add_orientations(
    geo_model=geo_model,
    x=new_orientations.data['X'],
    y=new_orientations.data['Y'],
    z=new_orientations.data['Z'],
    pole_vector=new_orientations.grads,
    elements_names="Claudius_fault"
)

# %% 
gpv.plot_2d(geo_model, direction='y')

# %%
# We will need to separate with surface belong to each series:
# 

gp.map_stack_to_surfaces(
    gempy_model=geo_model,
    mapping_object={
        'Default series': ('0', '60', '250'),
        'Fault': 'Claudius_fault',
        'Uncomformity': '330',
    }
)
# %%
# So far we did not specify which series/faults are actula faults:
# 

# %%
gp.set_is_fault(
    frame=geo_model.structural_frame,
    fault_groups=[geo_model.structural_frame.get_group_by_name('Fault')]
)

geo_model.structural_frame

# %%
geo_model.interpolation_options.kernel_options.range = 1
gp.compute_model(
    geo_model,
    gp.data.GemPyEngineConfig(
        backend=gp.data.AvailableBackends.numpy,
        use_gpu=False,
        dtype='float64'
    )
)

# %% 
sect = ['mid']

gpv.plot_2d(geo_model, cell_number=sect, series_n=1, show_scalar=True, direction='x')

# %% 
gpv.plot_2d(geo_model, cell_number=sect, show_data=True, direction='x')

# %% 
gpv.plot_2d(geo_model, cell_number=[28], series_n=0, direction='y', show_scalar=True)
gpv.plot_2d(geo_model, cell_number=[28], series_n=1, direction='y', show_scalar=True)
gpv.plot_2d(geo_model, cell_number=[28], series_n=2, direction='y', show_scalar=True)

# %% 
gpv.plot_2d(geo_model, cell_number=[28], show_data=True, direction='y')

# %%

# sphinx_gallery_thumbnail_number = 8
gpv.plot_3d(geo_model, show_lith=True, show_data=True, show_boundaries=True)

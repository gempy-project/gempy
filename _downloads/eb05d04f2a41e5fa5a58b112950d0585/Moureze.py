"""
Geomodeling benchmark: the "Moureze"-Model
==========================================

This model is part of a geomodeling benchmaring effort. More information (and, hopefully, publication) coming.
"""

import os

# %% 
# These two lines are necessary only if gempy is not installed

# Importing gempy
import gempy as gp
import gempy_viewer as gpv

# Aux imports
import numpy as np
import pandas as pd

from gempy_engine.config import AvailableBackends

# %%
# Loading surface points from repository:
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# With pandas we can do it directly from the web and with the right args
# we can directly tidy the data in gempy style:
# 

# %% 

data_path = os.path.abspath('../../data/input_data/Moureze')
Moureze_points = pd.read_csv(
    filepath_or_buffer=data_path + '/Moureze_Points.csv',
    sep=';',
    names=['X', 'Y', 'Z', 'G_x', 'G_y', 'G_z', '_'],
    header=0,
)

Sections_EW = pd.read_csv(
    filepath_or_buffer=data_path + '/Sections_EW.csv',
    sep=';',
    names=['X', 'Y', 'Z', 'ID', '_'], header=1
).dropna()

Sections_NS = pd.read_csv(
    filepath_or_buffer=data_path + '/Sections_NS.csv',
    sep=';',
    names=['X', 'Y', 'Z', 'ID', '_'], header=1
).dropna()

# %%
# Extracting the orientatins:
# 

# %% 
mask_surfpoints = Moureze_points['G_x'] < -9999
surface_points = Moureze_points[mask_surfpoints][::10]
orientations = Moureze_points[~mask_surfpoints][::10]

# %%
# Giving an arbitrary value name to the surface
# 

# %% 
surface_points['surface'] = '0'
orientations['surface'] = '0'

# %% 
surface_points.tail()

# %% 
orientations.tail()

# %%
# Data initialization:
# ~~~~~~~~~~~~~~~~~~~~
# 
# Suggested size of the axis-aligned modeling box:
# 
# Origin: -5 -5 -200
# 
# Maximum: 305 405 -50
# 
# Suggested resolution: 2m (grid size 156 x 206 x 76)
# 


# %%
# Only using one orientation because otherwhise it gets a mess
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 

# %% 
# Number voxels
np.array([156, 206, 76]).prod()

# %% 
resolution_requ = [156, 206, 76]
resolution = [77, 103, 38]
resolution_low = [45, 51, 38]


surface_points_table: gp.data.SurfacePointsTable = gp.data.SurfacePointsTable.from_arrays(
    x=surface_points['X'].values,
    y=surface_points['Y'].values,
    z=surface_points['Z'].values,
    names=surface_points['surface'].values.astype(str)
)

orientations_table: gp.data.OrientationsTable = gp.data.OrientationsTable.from_arrays(
    x=orientations['X'].values,
    y=orientations['Y'].values,
    z=orientations['Z'].values,
    G_x=orientations['G_x'].values,
    G_y=orientations['G_y'].values,
    G_z=orientations['G_z'].values,
    names=orientations['surface'].values.astype(str),
    name_id_map=surface_points_table.name_id_map  # ! Make sure that ids and names are shared
)

structural_frame: gp.data.StructuralFrame = gp.data.StructuralFrame.from_data_tables(
    surface_points=surface_points_table,
    orientations=orientations_table
)

geo_model: gp.data.GeoModel = gp.create_geomodel(
    project_name='Moureze',
    extent=[-5, 305, -5, 405, -200, -50],
    # resolution=resolution_low,
    refinement=5,
    structural_frame=structural_frame
)

# %%
# Now we can see how the data looks so far:
# 

# %% 
gpv.plot_2d(geo_model, direction='y')


# %%
# The default range is always the diagonal of the extent. Since in this
# model data is very close we will need to reduce the range to 5-10% of
# that value:
# 

# %% 
geo_model.interpolation_options.kernel_options.range *= 0.2
geo_model.interpolation_options.evaluation_options.verbose = True
geo_model.interpolation_options.evaluation_options.octree_error_threshold = 1.5
geo_model.interpolation_options.evaluation_options.number_octree_levels_surface = 5

# %%
gp.compute_model(
    gempy_model=geo_model,
    engine_config=gp.data.GemPyEngineConfig(
        use_gpu=False, 
        dtype='float32',
        backend=AvailableBackends.PYTORCH
    )
)


# %%
# Time
# ~~~~
# 
# 300k voxels 3.5k points
# ^^^^^^^^^^^^^^^^^^^^^^^
# 
# -  Nvidia 2080: 500 ms ± 1.3 ms per loop (mean ± std. dev. of 7 runs, 1
#    loop each), Memory 1 Gb
# -  CPU 14.2 s ± 82.4 ms per loop (mean ± std. dev. of 7 runs, 1 loop
#    each), Memory: 1.3 Gb
# 
# 2.4 M voxels, 3.5k points
# ^^^^^^^^^^^^^^^^^^^^^^^^^
# 
# -  CPU 2min 33s ± 216 ms per loop (mean ± std. dev. of 7 runs, 1 loop
#    each) Memory: 1.3 GB
# -  Nvidia 2080: 1.92 s ± 6.74 ms per loop (mean ± std. dev. of 7 runs, 1
#    loop each) 1 Gb
# 
# 2.4 M voxels, 3.5k points 3.5 k orientations
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# 
# -  Nvidia 2080: 2.53 s ± 1.31 ms per loop (mean ± std. dev. of 7 runs, 1
#    loop each)
# 

# %% 

gpv.plot_2d(geo_model, cell_number='mid', series_n=0, show_scalar=True)
# %% 
gpv.plot_2d(geo_model, cell_number='mid', show_data=True, direction='y')

# %%
# sphinx_gallery_thumbnail_number = 4
gpv.plot_3d(geo_model)

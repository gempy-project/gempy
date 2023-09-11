"""
Alesmodel: Plotting sections and maps.
======================================

"""

import gempy as gp
import gempy_viewer as gpv
import numpy as np
import matplotlib.pyplot as plt
import os

# %%
cwd = os.getcwd()
if 'examples' not in cwd:
    data_path = os.getcwd() + '/examples'
else:
    data_path = cwd + '/../..'

path_interf = data_path + "/data/input_data/AlesModel/2018_interf.csv"
path_orient = data_path + "/data/input_data/AlesModel/2018_orient_clust_n_init5_0.csv"
path_dem = data_path + "/data/input_data/AlesModel/_cropped_DEM_coarse.tif"

# %% 
geo_model: gp.data.GeoModel = gp.create_geomodel(
    project_name='Claudius',
    extent=[729550.0, 751500.0, 1913500.0, 1923650.0, -1800.0, 800.0],
    resolution=[100, 100, 100],
    number_octree_levels=6,
    importer_helper=gp.data.ImporterHelper(
        path_to_orientations=path_orient,
        path_to_surface_points=path_interf,
    )
)

# %% 
gp.set_section_grid(
    grid=geo_model.grid,
    section_dict={
        'section1': ([732000, 1916000], [745000, 1916000], [200, 150])
    }
)

# %% 
# sorting of lithologies
gp.map_stack_to_surfaces(
    gempy_model=geo_model,
    mapping_object={
        'fault_left'     : 'fault_left',
        'fault_right'    : 'fault_right',
        'fault_lr'       : 'fault_lr',
        'Trias_Series'   : ('TRIAS', 'LIAS'),
        'Carbon_Series'  : 'CARBO',
        'Basement_Series': 'basement'
    },
    remove_unused_series=True
)

# %% 
# TODO: Update Colors
# colordict = {'LIAS'      : '#015482', 'TRIAS': '#9f0052', 'CARBO': '#ffbe00', 'basement': '#728f02',
#              'fault_left': '#2a2a2a', 'fault_right': '#545454', 'fault_lr': '#a5a391'}
# geo_model.surfaces.colors.change_colors(colordict)

# %% 
a = gpv.plot_2d(geo_model, direction='y')

# %% 
gpv.plot_section_traces(geo_model)

# %%
# Faults
# ''''''
# 

# %% 
gp.set_is_fault(
    frame=geo_model.structural_frame,
    fault_groups=[
        geo_model.structural_frame.get_group_by_name('fault_left'),
        geo_model.structural_frame.get_group_by_name('fault_right'),
        geo_model.structural_frame.get_group_by_name('fault_lr')
    ],
    change_color=True
)


# %%
# Topography
# ~~~~~~~~~~
# 

# %% 
# TODO:
# geo_model.set_topography(source='gdal', filepath=path_dem)

# %% 
print(geo_model.structural_frame)
geo_model.structural_frame


# %% 
_ = gp.compute_model(geo_model, engine_config=gp.data.GemPyEngineConfig(use_gpu=True))

# %% 
# BUG: Plot topography has to be Ture
gpv.plot_2d(geo_model, cell_number=[4], direction=['y'], show_topography=False, show_data=True)

# %% 
# gpv.plot_2d(geo_model, section_names=['topography'], show_data=False, show_boundaries=False)

# %%
# sphinx_gallery_thumbnail_number = 5
gpv.plot_3d(geo_model)

"""
Alesmodel: Plotting sections and maps.
======================================

# %%
# .. admonition:: Explanation
#
#       This model is quite unstable in general and requires float64 to find a solution. The lack of data on
#       one of the corners for the TRIAS and LIAS series makes that the model bends in an unrealistic
#       way and erodes CARBO that disappears on that section. The easy way to solve this is to add more data in that area but
#       I leave as it is since I did no constructed the model.
#

"""

import gempy as gp
import gempy_viewer as gpv
import os
import numpy as np


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
    refinement=6,
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
# Change colors
geo_model.structural_frame.get_element_by_name("LIAS").color = "#015482"
geo_model.structural_frame.get_element_by_name("TRIAS").color = "#9f0052"
geo_model.structural_frame.get_element_by_name("CARBO").color = "#ffbe00"

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
gp.set_topography_from_file(
    grid=geo_model.grid,
    filepath=path_dem,
    crop_to_extent=[729550.0, 751500.0, 1913500.0, 1923650.0]
)


gpv.plot_3d(geo_model, show_topography=True, ve=1, image=True)

# %%
carbo = geo_model.structural_frame.get_group_by_name("Carbon_Series")

# %%
geo_model.interpolation_options.number_octree_levels_surface = 4
geo_model.interpolation_options.kernel_options.range = .8
gp.modify_surface_points(
    geo_model=geo_model,
    elements_names=["CARBO", "LIAS", "TRIAS"],
    nugget=0.01
)

# %% 
print(geo_model.structural_frame)
geo_model.structural_frame

# %% 
_ = gp.compute_model(
    geo_model,
    engine_config=gp.data.GemPyEngineConfig(
        use_gpu=True,
        dtype="float64"
    ))


gpv.plot_2d(geo_model, show_topography=False, section_names=['topography'], show_lith=True)

# %% 
gpv.plot_2d(geo_model, cell_number=[4], direction=['y'], show_topography=True, show_data=True)
gpv.plot_2d(geo_model, cell_number=[-4], direction=['y'], show_topography=True, show_data=True)

# %%
# sphinx_gallery_thumbnail_number = -1
gpv.plot_3d(geo_model, show_lith=True, show_topography=False, kwargs_plot_structured_grid={'opacity': 0.2})

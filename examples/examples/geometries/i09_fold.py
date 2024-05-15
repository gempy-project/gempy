# %%
# Import minmal requirements
import gempy as gp
import gempy_viewer as gpv

# %%
# Some ways for input data
# data_path = 'https://raw.githubusercontent.com/cgre-aachen/gempy_data/master/'
# path_to_data = data_path + "/data/input_data/jan_models/"

# Look at csv file --> Github

data_path = '../../'
path_to_data = data_path + "/data/input_data/jan_models/"
# %%
# Create a GeoModel instance
geo_model = gp.create_geomodel(
    project_name='tutorial_model',
    extent=[0, 2500, 0, 1000, 0, 1110],
    refinement=5,
    # resolution=[40, 40, 40],
    importer_helper=gp.data.ImporterHelper(
        path_to_orientations=path_to_data + "tutorial_model_orientations.csv",
        path_to_surface_points=path_to_data + "tutorial_model_surface_points.csv"
    )
)
# %%
# Displaying simple data cross section
# gpv.plot_2d(geo_model)
# %%
# Map geological series to surfaces
gp.map_stack_to_surfaces(
    gempy_model=geo_model,
    mapping_object={
            "Strat_Series1": ('rock3'),
            "Strat_Series2": ('rock2', 'rock1'),
    }
)
# %%
# geo_model.interpolation_options._number_octree_levels_surface=4
# %%

# %%

# %%
geo_model.update_transform(auto_anisotropy=gp.data.GlobalAnisotropy.CUBE)

import numpy as np

# refinement = 3
geo_model.interpolation_options.kernel_options.range = 1
geo_model.interpolation_options.mesh_extraction = False
geo_model.interpolation_options.compute_scalar_gradient = True
# geo_model.interpolation_options.number_octree_levels_surface = 5
# geo_model.grid.regular_grid.resolution = np.array([2 ** refinement] * 3)
gp.compute_model(
    gempy_model=geo_model,
    engine_config=gp.data.GemPyEngineConfig(backend=gp.data.AvailableBackends.PYTORCH)
)

gpv.plot_2d(geo_model, show_scalar=False, series_n=1)
# %% md

# %%
gpv.plot_2d(geo_model, show_scalar=True, series_n=1)

# %%
# gpv.plot_3d(geo_model, show_lith=False)
# %%
# some ways for manually adding data
'''
gp.add_surface_points(
    geo_model=geo_model,
    x=[458, 612],
    y=[0, 0],
    z=[-107, -14],
    elements_names=['surface1', 'surface1']
)
'''
# %%
# Displaying data cross section
# gpv.plot_2d(geo_model)
# %%

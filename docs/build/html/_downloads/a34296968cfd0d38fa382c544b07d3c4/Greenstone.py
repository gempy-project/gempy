"""
Greenstone.
===========
"""
import os

# Importing gempy
import gempy as gp
import gempy_viewer as gpv

print(gp.__version__)

# %%

data_path = os.path.abspath('../../data/input_data/tut_SandStone')

# Importing the data from csv

geo_model: gp.data.GeoModel = gp.create_geomodel(
    project_name='Greenstone',
    extent=[696000, 747000, 6863000, 6930000, -20000, 200],  # * Here we define the extent of the model
    refinement=6,
    # * Here we define the number of octree levels. If octree levels are defined, the resolution is ignored.
    importer_helper=gp.data.ImporterHelper(
        path_to_orientations=data_path + "/SandStone_Foliations.csv",
        path_to_surface_points=data_path + "/SandStone_Points.csv",
        hash_surface_points=None,
        hash_orientations=None
    )
)

# %% 
gpv.plot_2d(geo_model, direction=['z'])

# %% 
gp.map_stack_to_surfaces(
    gempy_model=geo_model,
    mapping_object={
        "EarlyGranite_Series": 'EarlyGranite',
        "BIF_Series": ('SimpleMafic2', 'SimpleBIF'),
        "SimpleMafic_Series": 'SimpleMafic1', 'Basement': 'basement'
    }
)

# %% 
gp.compute_model(geo_model)

# %% 
gpv.plot_2d(geo_model, cell_number=[-1], direction=['z'], show_data=False)

# %% 
gpv.plot_2d(geo_model, cell_number=['mid'], direction='x')

# %%
# sphinx_gallery_thumbnail_number = -1
gpv.plot_3d(geo_model)

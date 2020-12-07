"""
Alesmodel: Plotting sections and maps.
======================================

"""

import gempy as gp
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
resolution = [100, 100, 100]
extent = np.array([729550.0, 751500.0, 1913500.0, 1923650.0, -1800.0, 800.0])
geo_model = gp.create_model('Alesmodel')
gp.init_data(geo_model, extent=extent, resolution=resolution,
             path_i=path_interf,
             path_o=path_orient)

# %% 
sdict = {'section1': ([732000, 1916000], [745000, 1916000], [200, 150])}
geo_model.set_section_grid(sdict)

# %% 
# sorting of lithologies
gp.map_stack_to_surfaces(geo_model, {'fault_left': ('fault_left'),
                                     'fault_right': ('fault_right'),
                                     'fault_lr': ('fault_lr'),
                                     'Trias_Series': ('TRIAS', 'LIAS'),
                                     'Carbon_Series': ('CARBO'),
                                     'Basement_Series': ('basement')}, remove_unused_series=True)

# %% 
colordict = {'LIAS': '#015482', 'TRIAS': '#9f0052', 'CARBO': '#ffbe00', 'basement': '#728f02',
             'fault_left': '#2a2a2a', 'fault_right': '#545454', 'fault_lr': '#a5a391'}
geo_model.surfaces.colors.change_colors(colordict)

# %% 
a = gp.plot_2d(geo_model, direction='y')

# %% 
geo_model.rescaling

# %% 
gp.plot.plot_section_traces(geo_model)

# %%
# Faults
# ''''''
# 

# %% 
geo_model.set_is_fault(['fault_right', 'fault_left', 'fault_lr'], change_color=True)

# %% 
gp.set_interpolator(geo_model,
                    output=['geology'], compile_theano=True,
                    theano_optimizer='fast_run', dtype='float64',
                    verbose=[])

# %%
# Topography
# ~~~~~~~~~~
# 

# %% 
geo_model.set_topography(source='gdal', filepath=path_dem)

# %% 
geo_model.surfaces

# %% 
_ = gp.compute_model(geo_model, compute_mesh=True, compute_mesh_options={'rescale': False})

# %% 
gp.plot_2d(geo_model, cell_number=[4], direction=['y'], show_topography=True,
           show_data=True)

# %% 
gp.plot_2d(geo_model, section_names=['topography'], show_data=False,
           show_boundaries=False)

# %%
# sphinx_gallery_thumbnail_number = 5
gp.plot_3d(geo_model)

# %%
# np.save('Ales_vert3', geo_model.solutions.vertices)
# np.save('Ales_edges3', geo_model.solutions.edges)

# %% 
# gp.plot.plot_ar(geo_model)

gp.save_model(geo_model)
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: sphinx
#       format_version: '1.1'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

#import warnings
import sys
sys.path.append("../..")

import gempy as gp
import numpy as np
import matplotlib.pyplot as plt
import os

""
path_interf = os.pardir+"/data/input_data/AlesModel/2018_interf.csv"
path_orient = os.pardir+"/data/input_data/AlesModel/2018_orient_clust_n_init5_0.csv"
path_dem = os.pardir+"/data/input_data/AlesModel/_cropped_DEM_coarse.tif"

""
resolution = [100, 100, 100]
extent = np.array([729550.0, 751500.0, 1913500.0, 1923650.0, -1800.0, 800.0])
geo_model = gp.create_model('Alesmodel')
gp.init_data(geo_model, extent = extent, resolution = resolution,
                        path_i = path_interf,
                        path_o = path_orient)

""
sdict = {'section1':([732000, 1916000],[745000,1916000],[200,150])}
geo_model.grid.create_section_grid(sdict)

""
print(len(geo_model.orientations.df))
print(len(geo_model.surface_points.df))
print(len(geo_model.surfaces.df))

""
#sorting of lithologies
gp.map_series_to_surfaces(geo_model,{'fault_left':('fault_left'),
                        'fault_right':('fault_right'),
                        'fault_lr':('fault_lr'),
                        'Trias_Series':('TRIAS','LIAS'), 
                        'Carbon_Series':('CARBO'), 
                        'Basement_Series':('basement')},remove_unused_series=True)

""
colordict = {'LIAS':'#015482', 'TRIAS': '#9f0052', 'CARBO':'#ffbe00','basement':'#728f02',
            'fault_left':'#2a2a2a','fault_right':'#545454', 'fault_lr': '#a5a391'}
geo_model.surfaces.colors.change_colors(colordict)

""
a = gp.plot.plot_data(geo_model,direction='y')

""
geo_model.rescaling

""
gp.plot.plot_section_traces(geo_model, contour_lines=True, show_all_data=False)

"""
##### Faults
"""

geo_model.set_is_fault(['fault_right', 'fault_left', 'fault_lr'], change_color=True)

""
gp.set_interpolation_data(geo_model,
                          output=['geology'], compile_theano=True,
                          theano_optimizer='fast_run', dtype='float64',
                          verbose=[])

###############################################################################
# ### Topography

geo_model.set_topography(source='gdal', filepath=path_dem)

""
from scipy.spatial import Delaunay
tri=Delaunay(geo_model.grid.topography.values[:,:2])
f = tri.simplices
f

""
geo_model.grid.topography.values_3D.shape, geo_model.grid.topography.values_3D_res.shape, geo_model.grid.topography.values.shape

""
geo_model.surfaces

""
_=gp.compute_model(geo_model, compute_mesh=True, compute_mesh_options={'rescale': True})

""
geo_model.interpolator.theano_graph.n_surfaces_per_series.get_value()

""
geo_model.interpolator.print_theano_shared()

""
gp.plot.plot_section(geo_model, 4, direction='y',show_topo=True, show_data=True, show_faults=True, show_all_data=True)

""
gp.plot.plot_map(geo_model, show_data=False, contour_lines=False)

""
#np.save('Ales_vert3', geo_model.solutions.vertices)
#np.save('Ales_edges3', geo_model.solutions.edges)

""
gp.plot.plot_ar(geo_model)

""


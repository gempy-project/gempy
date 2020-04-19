# -*- coding: utf-8 -*-
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

"""
### Loading packages
"""

# These two lines are necessary only if gempy is not installed
import sys, os
sys.path.append("../../..")
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cuda"

import theano
# Importing gempy
import gempy as gp

# Embedding matplotlib figures into the notebooks
# %matplotlib inline

# Aux imports
import numpy as np
import pandas as pn
import matplotlib.pyplot as plt

###############################################################################
# ### Loading surface points from repository:
#
# With pandas we can do it directly from the web and with the right args we can directly tidy the data in gempy style:

Moureze_points = pn.read_csv('https://raw.githubusercontent.com/Loop3D/ImplicitBenchmark/master/Moureze/Moureze_Points.csv', sep=';',
                         names=['X', 'Y', 'Z', 'G_x', 'G_y', 'G_z', '_'], header=0, )
Sections_EW = pn.read_csv('https://raw.githubusercontent.com/Loop3D/ImplicitBenchmark/master/Moureze/Sections_EW.csv', sep=';',
                         names=['X', 'Y', 'Z', 'ID', '_'], header=1).dropna()
Sections_NS = pn.read_csv('https://raw.githubusercontent.com/Loop3D/ImplicitBenchmark/master/Moureze/Sections_NS.csv', sep=';',
                         names=['X', 'Y', 'Z', 'ID', '_'], header=1).dropna()

###############################################################################
# Extracting the orientatins:

mask_surfpoints = Moureze_points['G_x'] < -9999
surfpoints = Moureze_points[mask_surfpoints]
orientations = Moureze_points[~mask_surfpoints]

###############################################################################
# Giving an arbitrary value name to the surface

surfpoints['surface'] = 0.0
orientations['surface'] = 0.0

""
surfpoints.tail()

""
orientations.tail()

###############################################################################
# ### Data initialization:
#
#
# Suggested size of the axis-aligned modeling box:
#
# Origin: -5 -5 -200
#
# Maximum: 305 405 -50
#
# Suggested resolution: 2m (grid size 156 x 206 x 76)

###############################################################################
# ### Only using one orientation because otherwhise it gets a mess

# Number voxels
np.array([156, 206, 76]).prod()

""
resolution_requ = [156, 206, 76]
resolution = [77, 103, 38]
geo_model = gp.create_model('Moureze')
geo_model = gp.init_data(geo_model, 
                         extent=[-5,305,-5,405,-200, -50], resolution=resolution_requ,
                         surface_points_df=surfpoints, orientations_df=orientations,
                         surface_name='surface',
                         add_basement=True)

###############################################################################
# Now we can see how the data looks so far:

gp.plot.plot_data(geo_model, direction='y')

""
gp.set_interpolation_data(geo_model, 
                          theano_optimizer='fast_run')

###############################################################################
# The default range is always the diagonal of the extent. Since in this model data is very close we will need to reduce the range to 5-10% of that value:

val=.1
geo_model.interpolator.theano_graph.a_T.set_value(val)
geo_model.interpolator.theano_graph.a_T_surface.set_value(val)


""
gp.compute_model(geo_model, set_solutions=True, sort_surfaces=False)

###############################################################################
# ### Time
# #### 300k voxels 3.5k points
# - Nvidia 2080: 500 ms ± 1.3 ms per loop (mean ± std. dev. of 7 runs, 1 loop each), Memory 1 Gb
# - CPU  14.2 s ± 82.4 ms per loop (mean ± std. dev. of 7 runs, 1 loop each), Memory: 1.3 Gb
#
# #### 2.4 M voxels, 3.5k points
#
# - CPU 2min 33s ± 216 ms per loop (mean ± std. dev. of 7 runs, 1 loop each) Memory: 1.3 GB
# - Nvidia 2080:  1.92 s ± 6.74 ms per loop (mean ± std. dev. of 7 runs, 1 loop each) 1 Gb
#
# #### 2.4 M voxels, 3.5k points 3.5 k orientations
# - Nvidia 2080: 2.53 s ± 1.31 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

gp.plot.plot_scalar_field(geo_model, 16, series=0, direction='x')

""
gp.plot.plot_section(geo_model,16, show_data=True, direction='y')

""
# gp.plot.plot_3D(geo_model, render_data=False)

###############################################################################
#
# ![](./Moureze.png)

###############################################################################
# ### Export data:
#
# The solution is stored in a numpy array of the following shape. Axis 0 are the scalar fields of each correspondent series/faults in the following order (except basement): 

geo_model.series

###############################################################################
# For the surfaces, there are two numpy arrays, one with vertices and the other with triangles. Axis 0 is each surface in the order:

geo_model.surfaces

""
# np.save('Moureze_scalar', geo_model.solutions.scalar_field_matrix)
# np.save('Moureze_ver', geo_model.solutions.vertices)
# np.save('Moureze_edges', geo_model.solutions.edges)
# gp.plot.export_to_vtk(geo_model, 'Moureze')

""
def write_property_to_gocad_voxet(propertyfilename, propertyvalues):
    """
    This function writes a numpy array into the right format for a gocad
    voxet property file. This assumet there is a property already added to the .vo file,
    and is just updating the file.
    propertyfile - string giving the path to the file to write
    propertyvalues - numpy array nz,ny,nx ordering and in float format
    """
    propertyvalues = propertyvalues.astype('>f4') #big endian
#     array = propertyvalues.newbyteorder()
    propertyvalues.tofile(propertyfilename)


""
write_property_to_gocad_voxet('moureze_sf_gempy',
                              geo_model.solutions.scalar_field_matrix[0].reshape([156, 206, 76]).ravel('F'))

""


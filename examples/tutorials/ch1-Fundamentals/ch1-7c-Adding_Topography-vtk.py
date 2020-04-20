"""
Adding topography to geological models
======================================

"""

import sys
sys.path.append("../../..")

import gempy as gp
import numpy as np
import matplotlib.pyplot as plt
import os


######################################################################
# 1. The common procedure to set up a model:
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 

geo_model = gp.create_model('Tutorial_ch1-7_Single_layer_topo')

data_path= '../..'

gp.init_data(geo_model, extent=[450000, 460000, 70000,80000,-1000,500],resolution = (50,50,50),
                         path_i = data_path+"/data/input_data/tut-ch1-7/onelayer_interfaces.csv",
                         path_o = data_path+"/data/input_data/tut-ch1-7/onelayer_orient.csv")


# use happy spring colors! 
geo_model.surfaces.colors.change_colors({'layer1':'#ff8000','basement':'#88cc60'})

# %matplotlib inline
gp.map_series_to_surfaces(geo_model, {'series':('layer1','basement')})


######################################################################
# 2. Adding topography
# ~~~~~~~~~~~~~~~~~~~~
# 


######################################################################
# 2.b create fun topography
# ^^^^^^^^^^^^^^^^^^^^^^^^^
# 

geo_model.set_topography(d_z=np.array([0,200]))

gp.plot.plot_data(geo_model)


######################################################################
# 2 a. Load from raster file
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
# 

fp = data_path+"/data/input_data/tut-ch1-7/bogota.tif"

geo_model.set_topography()

vtkp = gp.plot.plot_3D(geo_model)

vtkp.resume()

geo_model.grid.topography.load_from_gdal(filepath=fp)

geo_model.set_topography(source='gdal',filepath=fp)

plt.imshow(geo_model.grid.topography.topo.dem_zval)
plt.colorbar()

gp.set_interpolation_data(geo_model,
                          compile_theano=True,
                          theano_optimizer='fast_compile')

gp.compute_model(geo_model)

vtkp = gp.plot.plot_3D(geo_model)

vtkp.set_real_time_on()

vtkp.update_model()

geo_model.set_topography(d_z=np.array([0,100]), fd=0.9, plot_object= vtkp)

gp.plot.plot_section(geo_model, 25, direction='y', block=geo_model.grid.regular_grid.mask_topo,
                show_topo=False)

gp.plot.plot_section(geo_model, 25, direction='x', block=geo_model.grid.regular_grid.mask_topo,
                show_topo=False)

"""
Chapter 1.6: Creating models from scratch
-----------------------------------------

"""

# These two lines are necessary only if gempy is not installed
import sys, os
sys.path.append("../../..")

# Importing gempy
import gempy as gp

# Embedding matplotlib figures into the notebooks
#%matplotlib inline


# Aux imports
import numpy as np
import pandas as pn
import matplotlib.pyplot as plt
import theano
import qgrid

#gp.save_model(geo_model, 'CreateModelTuto5', path=os.pardir+'/data/gempy_models')

data_path= '../..'
if False:
    geo_model = gp.load_model('Tutorial_ch1-6_CreatingModels', path=data_path+'/data/gempy_models')
else:
    geo_model = gp.create_model('Tutorial_ch1-6_CreatingModels')
    gp.init_data(geo_model, [0, 1000, 0, 1000, -1000, 0], [50, 50, 50])
    geo_model.set_default_surfaces()
    geo_model.set_default_orientation()
    geo_model.add_surface_points(400, 300, -500, 'surface1')
    geo_model.add_surface_points(600, 300, -500, 'surface1')



######################################################################
# Some default values but to make the model a bit faster but they are not
# necessary:
# 

gp.set_interpolation_data(geo_model, theano_optimizer='fast_run',  verbose=[])

geo_model.additional_data

gp.compute_model(geo_model, debug=False,compute_mesh=False, sort_surfaces=False)

gp.plot.plot_section(geo_model, cell_number=25,
                         direction='x', show_data=True)


gp.plot.plot_scalar_field(geo_model, 25, direction='x', series=0)

vtk_object = gp.plot.plot_3D(geo_model, render_surfaces=True, silent=True)

vtk_object.real_time =True

geo_model.modify_surface_points(0, X=-500,
                               plot_object=vtk_object)


######################################################################
# Passing the vtk object to qgrid
# -------------------------------
# 

gp.activate_interactive_df(geo_model, vtk_object)


######################################################################
# It is important to get df with get to update the models sinde the
# ``activate_interactive`` method is called
# 

geo_model.qi.get('orientations')

geo_model.qi.get('surface_points')

geo_model.qi.get('surfaces')

geo_model.qi.get('series')

geo_model.qi.get('faults')

geo_model.qi.get('faults_relations')


######################################################################
# Finite Fault parameters
# -----------------------
# 

geo_model.interpolator.theano_graph.not_l.set_value(1.)
vtk_object.update_model()

geo_model.interpolator.theano_graph.ellipse_factor_exponent.set_value(50)

vtk_object.update_model()


######################################################################
# Topography
# ~~~~~~~~~~
# 

geo_model.set_topography(d_z=np.array([0,-600]))

geo_model.grid.active_grids

gp.compute_model(geo_model)

gp.plot.plot_section(geo_model)

gp.plot.plot_map(geo_model)

vtk_object.render_topography()

np.unique(geo_model.surface_points.df['id'])

geo_model.surface_points
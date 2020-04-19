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
## Chapter 1.10 Modifying kriging parameters
"""

# These two lines are necessary only if gempy is not installed
import sys, os
sys.path.append("../..")

# Importing gempy
import gempy as gp

# Embedding matplotlib figures into the notebooks
# #%matplotlib inline


# Aux imports
import numpy as np
import pandas as pn
import matplotlib.pyplot as plt
import theano
import qgrid

# #%matplotlib widget


""
gp.save_model(geo_model, 'Tutorial2-1')

""
if False:
    geo_model = gp.load_model('Tutorial2-1')
else:
    geo_model = gp.create_model('Tutorial2-1')
    gp.init_data(geo_model, [0, 1000, 0, 1000, -1000, 0], [50, 50, 50])
    geo_model.set_default_surfaces()
#     geo_model.set_default_orientation()
#     geo_model.add_surface_points(400, 300, -500, 'surface1')
#     geo_model.add_surface_points(600, 300, -500, 'surface1')


""
gp.set_interpolation_data(geo_model, theano_optimizer='fast_compile',  verbose=[])

""
geo_model.additional_data

""
vtk_object = gp.plot.plot_3D(geo_model)

""
vtk_object.set_real_time_on()

""
geo_model.additional_data

""
geo_model.modify_kriging_parameters('range', 50)

""
geo_model.interpolator.theano_graph.a_T.get_value()

""
vtk_object.update_model()

""
vtk_object.resume()

""


""
gp.activate_interactive_df(geo_model, vtk_object)

""
geo_model.qi.get('surface_points')

""
geo_model.qi.get('orientations')

""
geo_model.modify_kriging_parameters('range', 5000)
geo_model.modify_kriging_parameters('drift equations', np.array([0]))
geo_model.interpolator.set_initial_results()
geo_model.rescaling.set_rescaled_orientations()
vtk_object.update_model()

""
geo_model.interpolator.theano_graph.a_T.get_value()

""
geo_model.interpolator.theano_graph.c_o_T.get_value()

""
gp.compute_model(geo_model, debug=False,compute_mesh=True, sort_surfaces=False)


""
gp.plot.plot_scalar_field(geo_model, 25, direction='x', series=0)
plt.colorbar()

""
geo_model.additional_data.kriging_data

""


""
gp.plot.plot_section(geo_model, cell_number=25, block_type=geo_model.solutions.lith_block,
                         direction='x', plot_data=True)


""


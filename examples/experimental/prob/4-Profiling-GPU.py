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

 # These two lines are necessary only if GemPy is not installed
import sys, os
sys.path.append("../../..")
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cuda"
# Importing GemPy
import gempy as gp

# Embedding matplotlib figures in the notebooks
# %matplotlib inline

# Importing auxiliary libraries
import numpy as np
import matplotlib.pyplot as plt
#import pymc3 as pm

""
path_to_data = os.pardir+"/../data/input_data/jan_models/"

geo_data = gp.create_data([0,1000,0,1000,0,1000],resolution=[100,100,100], 
                        path_o = path_to_data + "model1_orientations.csv",
                        path_i = path_to_data + "model1_surface_points.csv") 

""
gp.plot.plot_data(geo_data)

""
dz = geo_data.grid.regular_grid.dz
dz

""
geo_data.surfaces.add_surfaces_values([0, 5, 0])

""
geo_data.interpolator.theano_graph.block_matrix.dtype

""

import theano
theano.config.floatX

""
gp.set_interpolation_data(geo_data, theano_optimizer='fast_compile', gradient=False)

""
gp.compute_model(geo_data, compute_mesh=False)


""
gp.compute_model(geo_data)
gp.plot.plot_section(geo_data, 1, show_data=True)

""
geo_data.interpolator.grid.values_r

""
import theano
import theano.tensor as T

grid_sh = theano.shared(geo_data.interpolator.get_python_input_block()[6])

""
geo_data.interpolator.theano_graph.grid_val_T = grid_sh

""
geo_data.interpolator.theano_graph.grid_val_T

""


""
th_f_2 = theano.function(geo_data.interpolator.theano_graph.input_parameters_loop,
                        geo_data.interpolator.theano_graph.compute_a_series(
    len_i_0=0, len_i_1=geo_data.interpolator.theano_graph.len_series_i[1],
    len_f_0=0, len_f_1=geo_data.interpolator.theano_graph.len_series_o[1],
    len_w_0=0, len_w_1=geo_data.interpolator.theano_graph.len_series_w[1],
    n_form_per_serie_0=0, n_form_per_serie_1=geo_data.interpolator.theano_graph.n_surfaces_per_series[1],
    u_grade_iter=3,
    compute_weight_ctr=np.array(True), compute_scalar_ctr=np.array(True), compute_block_ctr=np.array(True),
    is_finite=np.array(False), is_erosion=np.array(True), is_onlap=np.array(False),
    n_series=0,
    block_matrix=geo_data.interpolator.theano_graph.block_matrix,
    weights_vector=geo_data.interpolator.theano_graph.weights_vector,
    scalar_field_matrix=geo_data.interpolator.theano_graph.scalar_fields_matrix,
    sfai=geo_data.interpolator.theano_graph.sfai,
    mask_matrix=geo_data.interpolator.theano_graph.mask_matrix
                 ),
                         on_unused_input='ignore',
                        profile=True)

""
th_f_2(*geo_data.interpolator.get_python_input_block())[1]

""
th_f_2.profile.summary()

""
theano.config.allow_gc, theano.config.scan.allow_gc
